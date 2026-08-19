//! The query language: how an agent binds its state to its state managers.
//!
//! ```text
//! position          a field of the state this agent is producing right now
//! prev!(position)   that field as of the end of this agent's previous step
//! agent!(Body1)     another agent's most recently committed state
//! <query>.<field>   a field of whatever <query> evaluates to
//! (a, b,)           a tuple; a consumed query is always one, and a produced query
//!                   may be, for a manager that returns a value per element
//! ```
//!
//! The grammar, written by hand because it is small enough to read in one sitting:
//!
//! ```text
//! query   := primary ("." name)*
//! primary := name "!" macro | name | "(" query ("," query)* ","? ")"
//! macro   := "(" query ")"     // prev!(..)
//!          | "(" name ")"      // agent!(..)
//! ```

use crate::Result;
use std::fmt;

/// A parsed query.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Query {
    /// Read from the previous step instead of the one being built.
    Prev(Box<Query>),
    /// The most recently committed state of another agent.
    Agent(String),
    /// A field of whatever `base` evaluates to.
    Access { base: Box<Query>, field: String },
    /// A field of the state of the agent the query is running for.
    Base(String),
    /// Several queries at once. A state manager's arguments, in order.
    Tuple(Vec<Query>),
}

impl Query {
    /// Parse a query.
    pub fn parse(source: &str) -> Result<Self> {
        Parser::parse(source).map_err(|err| format!("could not parse query `{}`: {err}", source.trim()))
    }
}

impl fmt::Display for Query {
    /// Writes the query back out in the syntax it was parsed from, so that error
    /// messages quote queries the way the model author wrote them.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Prev(query) => write!(f, "prev!({query})"),
            Self::Agent(id) => write!(f, "agent!({id})"),
            Self::Access { base, field } => write!(f, "{base}.{field}"),
            Self::Base(field) => write!(f, "{field}"),
            Self::Tuple(queries) => {
                write!(f, "(")?;
                for (index, query) in queries.iter().enumerate() {
                    if index > 0 {
                        write!(f, " ")?;
                    }
                    // The comma always trails: `(velocity,)` is a one-element
                    write!(f, "{query},")?;
                }
                write!(f, ")")
            }
        }
    }
}

/// A token, and the byte offset it starts at.
type Spanned<'a> = (Token<'a>, usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Token<'a> {
    /// An identifier: a field name, an agent id, or the name of a macro.
    Name(&'a str),
    Bang,
    Dot,
    Comma,
    Open,
    Close,
}

impl fmt::Display for Token<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Name(name) => write!(f, "`{name}`"),
            Self::Bang => write!(f, "`!`"),
            Self::Dot => write!(f, "`.`"),
            Self::Comma => write!(f, "`,`"),
            Self::Open => write!(f, "`(`"),
            Self::Close => write!(f, "`)`"),
        }
    }
}

/// A recursive-descent parser over a query.
struct Parser<'a> {
    tokens: Vec<Spanned<'a>>,
    /// How far into `tokens` the parser has read.
    position: usize,
}

impl<'a> Parser<'a> {
    fn parse(source: &'a str) -> Result<Query> {
        let mut parser = Self {
            tokens: tokenize(source)?,
            position: 0,
        };

        let query = parser.query()?;
        match parser.peek() {
            None => Ok(query),
            Some((token, offset)) => Err(format!(
                "expected the end of the query, found {token} at character {offset}"
            )),
        }
    }

    fn query(&mut self) -> Result<Query> {
        let mut query = self.primary()?;
        // Field access binds tighter than anything else and is left-associative:
        // `agent!(Body1).position.x` is `((agent!(Body1)).position).x`.
        while self.eat(Token::Dot) {
            query = Query::Access {
                base: Box::new(query),
                field: self.name()?.to_string(),
            };
        }
        Ok(query)
    }

    fn primary(&mut self) -> Result<Query> {
        match self.advance()? {
            (Token::Name(name), offset) => {
                if self.eat(Token::Bang) {
                    self.macro_call(name, offset)
                } else {
                    Ok(Query::Base(name.to_string()))
                }
            }
            (Token::Open, _) => self.tuple(),
            (token, offset) => Err(format!("expected a query, found {token} at character {offset}")),
        }
    }

    /// One of the `name!`-shaped queries. They are the only place the language
    /// reaches outside the state the agent is building right now.
    fn macro_call(&mut self, name: &str, offset: usize) -> Result<Query> {
        match name {
            "prev" => {
                self.expect(Token::Open)?;
                let query = self.query()?;
                self.expect(Token::Close)?;
                Ok(Query::Prev(Box::new(query)))
            }
            "agent" => {
                self.expect(Token::Open)?;
                let agent = self.name()?.to_string();
                self.expect(Token::Close)?;
                Ok(Query::Agent(agent))
            }
            _ => Err(format!(
                "there is no `{name}!` query (at character {offset}); \
                 the query language has `prev!(..)` and `agent!(..)`"
            )),
        }
    }

    /// A tuple, having consumed its opening parenthesis.
    fn tuple(&mut self) -> Result<Query> {
        let mut queries = vec![self.query()?];
        // A comma may or may not be followed by another query: the last one in
        // the tuple is allowed to trail.
        let mut trailing = false;
        while self.eat(Token::Comma) {
            trailing = true;
            if self.peek_token() == Some(Token::Close) {
                break;
            }
            trailing = false;
            queries.push(self.query()?);
        }

        // A trailing comma is the only thing that tells `(velocity,)` apart from
        // a query that merely happens to be parenthesized.
        if let (1, false, Some((Token::Close, offset))) = (queries.len(), trailing, self.peek()) {
            return Err(format!(
                "a tuple of one query needs a trailing comma, like `(velocity,)` \
                 (at character {offset})"
            ));
        }
        self.expect(Token::Close)?;
        Ok(Query::Tuple(queries))
    }

    fn peek(&self) -> Option<Spanned<'a>> {
        self.tokens.get(self.position).copied()
    }

    fn peek_token(&self) -> Option<Token<'a>> {
        self.peek().map(|(token, _)| token)
    }

    fn advance(&mut self) -> Result<Spanned<'a>> {
        let token = self
            .peek()
            .ok_or_else(|| "the query ends before it is finished".to_string())?;
        self.position += 1;
        Ok(token)
    }

    /// Consume `token` if it is next.
    fn eat(&mut self, token: Token<'_>) -> bool {
        if self.peek_token() == Some(token) {
            self.position += 1;
            return true;
        }
        false
    }

    fn expect(&mut self, token: Token<'_>) -> Result<()> {
        match self.peek() {
            Some((found, _)) if found == token => {
                self.position += 1;
                Ok(())
            }
            Some((found, offset)) => Err(format!("expected {token}, found {found} at character {offset}")),
            None => Err(format!("expected {token}, found the end of the query")),
        }
    }

    fn name(&mut self) -> Result<&'a str> {
        match self.advance()? {
            (Token::Name(name), _) => Ok(name),
            (token, offset) => Err(format!("expected a name, found {token} at character {offset}")),
        }
    }
}

/// Split a query into tokens. Whitespace separates tokens and is otherwise
/// ignored, which is what lets agents lay their queries out over several lines.
fn tokenize(source: &str) -> Result<Vec<Spanned<'_>>> {
    let mut tokens = Vec::new();
    let mut characters = source.char_indices().peekable();

    while let Some((offset, character)) = characters.next() {
        let token = match character {
            character if character.is_whitespace() => continue,
            '(' => Token::Open,
            ')' => Token::Close,
            '.' => Token::Dot,
            ',' => Token::Comma,
            '!' => Token::Bang,
            character if character.is_ascii_alphabetic() || character == '_' => {
                let mut end = offset + character.len_utf8();
                while let Some(&(next, character)) = characters.peek() {
                    if !character.is_ascii_alphanumeric() && character != '_' {
                        break;
                    }
                    end = next + character.len_utf8();
                    characters.next();
                }
                match source.get(offset..end) {
                    Some(name) => Token::Name(name),
                    None => return Err(format!("invalid name at character {offset}")),
                }
            }
            character => {
                return Err(format!("unexpected character `{character}` at character {offset}"));
            }
        };
        tokens.push((token, offset));
    }

    Ok(tokens)
}
