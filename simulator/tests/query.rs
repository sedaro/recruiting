//! Parsing the query language.
//!
//! `Query::parse` and the AST it produces are the whole of what a model author
//! touches, so this needs nothing that is not public — and nothing that is not Rust,
//! which makes it the one test in the crate that runs without an interpreter.
#![allow(clippy::expect_used)]

use sedaro_nano_simulator::query::Query;

#[test]
fn parses_a_consumed_query() {
    // NOTE: This test gives an example input/output pair for the parser.
    let input = "(prev!(timeStep), agent!(Body1).position,)";
    let expected_output = Query::Tuple(vec![
        Query::Prev(Box::new(Query::Base("timeStep".to_string()))),
        Query::Access {
            base: Box::new(Query::Agent("Body1".to_string())),
            field: "position".to_string(),
        },
    ]);

    let query = Query::parse(input).expect("Could not parse input!");

    assert_eq!(query, expected_output);
    // A query prints back out in the syntax it was parsed from.
    assert_eq!(query.to_string(), input);
    // A name may hold an underscore, the way a model written in Python usually spells one.
    assert_eq!(Query::parse("time_step"), Ok(Query::Base("time_step".to_string())));
}

#[test]
fn rejects_a_query_that_is_not_one() {
    // NOTE: The errors a model author actually hits. Each says what was wrong and
    // where, because a query is written in a string and the compiler cannot help.
    let cases = [
        ("(velocity)", "a tuple of one query needs a trailing comma"),
        ("root!", "there is no `root!` query"),
        ("nope!(x)", "there is no `nope!` query"),
        ("prev!(", "the query ends before it is finished"),
        ("velocity.", "the query ends before it is finished"),
        ("$", "unexpected character `$` at character 0"),
    ];

    for (input, expected) in cases {
        let error = Query::parse(input).expect_err(input);
        assert!(error.contains(expected), "parsing `{input}` said: {error}");
    }
}
