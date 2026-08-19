# Sedaro Nano
The tiniest possible mockup of our system

## Goal
This mini-project is a chance to show off your personal strengths and how you scope a problem, structure a solution, and reason about tradeoffs. Aim for a submission that's cohesive and impressive.

Submissions are evaluated on whether they:
- Showcase the skills this role needs. You don't have to contribute everywhere.
- Meaningfully improve the sample project and demonstrate sound product judgment.
- Tackle something challenging that matches the level for which you're applying.
- Meet the quality bar you'd hold for an MR on a production team: code you'd be proud to ship and comfortable handing a teammate for review.
- Are explained clearly and concisely, respecting our time.

The project is due within **7 days** of receipt and we expect you to spend no more than **6 hours** on it. If you have any questions, issues, or if you get stuck, please contact Kacie at `kacie.neurohr@sedaro.com`.

## Submission
Please submit a `.zip` file including:
- The **code** (excluding temporary or .gitignored files and directories, such as `__pycache__`, `node_modules`, `target`, or any local virtual environments)
- **Instructions** for setting up and running your solution
- **A video** of your solution in action explaining your key design decisions (**required** - submissions without a video will not be reviewed)
- A **write-up** explaining your changes and why you made them
- A description of the tools you used and how you used them, including any AI assistants or agents.

If you end up getting to a solution that you aren't happy with or that is a dead end, document why and we will call that good enough. A write-up of why a solution is insufficient and how you might approach it differently often tells us what we need to know.

Once you have completed your solution, email it to `kacie.neurohr@sedaro.com` and the other email(s) listed in the original instructions. To avoid your submission being blocked by our mail server, we recommend sharing the `.zip` file using a Google Drive link (or similar sharing service). Please notify Kacie in a separate email that you have submitted your solution so we can confirm receipt.

## Forking Policy

**We are happy for you to fork this repository but we ask that you keep the fork private so it remains hidden from other candidates.**

## Choosing a Project
Included in this directory is a tiny mockup of Sedaro's system. Though it technically comprises a full-stack app, there are _many_ areas in which it could be improved. Review the files that make up Sedaro Nano to figure out how it works, then choose a project that shows off your unique strengths. The prompt is intentionally open-ended to allow creative solutions.

Here are some suggestions to get you thinking:

### Frontend/full-stack
- Improve interactivity, for example live-streaming the simulation and allowing users to control the playback speed during the simulation
- Improve alignment with accessibility standards
- Elegantly support creating and managing many agents
- Support running a series of simulations with varying parameters and displaying the results

### Backend
- Add unique patterns of user engagement
- Create a more scalable storage solution than a JSON string in one DB row
- Do some statistical analysis on the data
- Set up background jobs to preprocess data
- Incorporate computational optimizations (e.g. linear programming)

### DevOps
- Integrate observability tooling and use it to performance profile the application
- Improve the availability of the application using clustering and infrastructure as code
- Write the "supreme pizza" version of a CI/CD pipeline
- Analyze and minimize the attack surface of the application without constraining development

### Workflows
- Set up background jobs to preprocess data
- Integrate observability tooling and use it to performance profile the application
- Improve the availability of the application using clustering and infrastructure as code
- Create a more scalable storage solution than a JSON string in one DB row

### Modeling & Simulation
- Improve the numerical stability of the simulation functions
- Implement additional modeling and simulation scope
- Analyze the sensitivity to initial conditions

### Compiler & Runtime
- Improve the performance or scalability of either buildtime or runtime
- Add new capabilities or semantics to the simulator or query language
- Improve QA for either developers or users of the simulator
- Strengthen the simulator's interfaces or guarantees

![](./files/screenshot2.png)

## Setup
Clone this repository.
- Please note that **only** cloning via HTTPS is supported
- Please **do not** commit changes to any branch of this repository. If you would like to use git, you may fork this repository to create a private repo of your own

Choose one of the options below to run the app at http://localhost:3030 and API at http://localhost:8000.

The `LOG_LEVEL` environment variable configures the logging level.

### Docker Compose
Recommended for non-Rust work.

- Install [Docker](https://www.docker.com/products/docker-desktop/)
- Run `./bin/main` to run the app with Docker Compose
- Non-Rust changes reload automatically
- Rust changes require re-running `./bin/main`

### Process Compose
Recommended for Rust work.

- Either:
   - Install [process-compose](https://f1bonacc1.github.io/process-compose/installation/), [Rust](https://rust-lang.org/tools/install/), [uv](https://docs.astral.sh/uv/getting-started/installation/), and [Node 24](https://nodejs.org/en/download)
      - Run `./bin/setup`
   - Install [Nix](https://docs.determinate.systems/determinate-nix/) and [direnv](https://search.nixos.org/packages?channel=26.05&query=direnv#show=direnv)
      - Run `direnv allow`
- Run `./bin/build` (or `./bin/build --release`)
- Run `./bin/main --dev` to run the app with Process Compose
- Non-Rust changes reload automatically
- Rust changes reload automatically after `./bin/build` commands

After `./bin/build`s, you can also run a simulation directly with the `./bin/run` script.

## Tutorial
In the initial version, the first body is not affected by the gravitational force of the second. See [app/query_tutorial.md](./app/query_tutorial.md) for guidance on fixing this, and a brief introduction to the nano query language.
