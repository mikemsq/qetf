# QuantETF

QuantETF is a small, modular research and production platform for ETF based quantitative strategies.

It is designed for two modes:
- **Research**: explore datasets, define investable universes, prototype signals, run backtests, compare strategies.
- **Production recommendations**: run a scheduled pipeline that outputs a **trade recommendation packet** (what to buy and sell, target weights, and an audit trail). This project does not include automated trading.

The codebase is structured so that you can swap components (data sources, alpha models, risk models, portfolio construction, rebalancing schedules) without rewriting the backtest engine or production runtime.


## Goals

- Make strategy research reproducible with versioned datasets and config driven experiments.
- Provide clean interfaces for the core building blocks of a quant workflow.
- Support backtests that are realistic enough to avoid obvious traps (lookahead, data leakage, missing costs).
- Produce production outputs that are easy to inspect, share, and audit.

Non-goals (for now):
- Live order execution, brokerage integration, OMS/EMS.
- Ultra low latency, intraday trading simulation.
- Perfect point in time holdings for every ETF (depends on data vendor).


## Mental model

A complete run, whether for research or production, looks like this:

1. **Pick a dataset snapshot** (a versioned, curated view of the raw data).
2. **Build an investable universe** for each rebalance date.
3. **Compute features** as-of each rebalance date (no peeking).
4. **Score the universe** using one or more alpha models.
5. **Estimate risk** (covariance and exposures) as-of the same date.
6. **Construct a portfolio** (target weights) subject to constraints and costs.
7. **Simulate** holdings and returns (backtest) or **emit recommendations** (production).


## Repository Structure

### Code
- **`src/quantetf/`** - Main library organized by domain:
  - `data/` - Ingestion, storage, validation, point-in-time access
  - `universe/` - Universe builders and filters
  - `features/` - Feature engineering
  - `alpha/` - Alpha models (momentum, etc.)
  - `risk/` - Risk models and covariance
  - `portfolio/` - Portfolio construction and transaction costs
  - `backtest/` - Event-driven backtest engine
  - `evaluation/` - Metrics and reporting
  - `production/` - Production runtime
  - `cli/` - Command-line interface
  - `utils/` - Shared utilities

### Data
- **`data/`** - Data storage hierarchy:
  - `raw/` - Immutable ingested data
  - `curated/` - Cleaned, normalized data
  - `snapshots/` - Versioned, reproducible datasets for backtests

### Configuration
- **`configs/`** - YAML configs for strategies, universes, schedules, costs

### Development
- **`scripts/`** - Utility scripts (ingest, snapshot, backtest)
- **`notebooks/`** - Jupyter notebooks for research
- **`tests/`** - Unit tests, integration tests, golden tests
- **`artifacts/`** - Output from runs (metrics, plots, recommendations)

### Multi-Agent Workflow
- **`handoffs/`** - Agent-to-agent task handoff files
- **`session-notes/`** - Detailed session history

### Documentation

See **Documentation Guide** section below for file purposes.


## Core abstractions

QuantETF is intentionally interface first. The important interfaces you will see across modules include:

- **Data**: ingestors/connectors, curated stores, snapshot selection by `DatasetVersion`.
- **UniverseProvider**: returns the eligible ETF list for a given as-of date.
- **FeatureComputer**: produces a feature matrix as-of a date for a universe.
- **AlphaModel**: produces scores (or expected returns) for a universe as-of a date.
- **RiskModel**: produces covariance and exposure summaries as-of a date.
- **PortfolioConstructor**: converts alpha + risk + constraints into target weights.
- **TransactionCostModel**: estimates costs from turnover and liquidity proxies.
- **BacktestEngine**: orchestrates a run on a schedule and returns a result bundle.
- **ProductionRuntime**: runs “today”, writes a recommendation packet, and records a manifest.

The key design rule is that all of these components accept an explicit `as_of` date and a `dataset_version`, so you can enforce point in time behavior and reproducibility.


## How to work with this project

### Research workflow (typical)
- Ingest data, curate it, and create a dataset snapshot.
- Define a universe and strategy config.
- Run a backtest and inspect artifacts in `artifacts/`.

### Production workflow (typical)
- On a schedule (weekly, monthly, or custom), run the production pipeline.
- The output is a dated recommendation packet (CSV or JSON plus a small manifest) that can be reviewed and then executed manually.


## Recommended next steps (practical order)

1. **Decide on data sources and the canonical schema**
   - Choose at least one price and total return source.
   - Decide how you will represent adjusted prices, distributions, and survivorship.
   - Define “instrument identifiers” rules (ticker mapping, symbol changes).

2. **Create a tiny end-to-end vertical slice**
   - Use a small fixed universe (for example 20 ETFs).
   - Implement one ingestion connector, one snapshot, one alpha model (momentum), one portfolio rule (equal weight top X), and a simple backtest.
   - Add at least 3 golden tests:
     - schedule generation and rebalance dates
     - portfolio weights sum to 1.0 and respect constraints
     - equity curve matches expected results on a tiny synthetic dataset

3. **Add a feasibility tool**
   - Implement oracle upper bound and random baseline simulations for the same X and Y schedule.
   - This makes it easy to answer “is outperformance even possible under these constraints?” quickly.

4. **Upgrade realism gradually**
   - Transaction costs, turnover penalties, liquidity filters.
   - Risk controls (beta target, volatility targeting, concentration limits).
   - Walk-forward evaluation and parameter selection hygiene.

5. **Production hardening**
   - Data freshness checks, run manifests (code hash, config hash, dataset version).
   - Logging, alerting, and a “diff vs last run” summary.
   - A stable recommendation output schema consumed by downstream tools.


## Documentation Guide

This project uses a clear separation of concerns for documentation:

### For All Sessions
- **[CLAUDE_CONTEXT.md](CLAUDE_CONTEXT.md)** - Coding standards, patterns, and best practices
  - What: Python style guide, common patterns, financial data guidelines
  - Use: Reference when writing code

### For Understanding the Project
- **[PROJECT_BRIEF.md](PROJECT_BRIEF.md)** - Vision, goals, and architecture
  - What: Overall project goals, success criteria, phases, technical decisions
  - Use: Understand what we're building and why

### For Current Status
- **[PROGRESS_LOG.md](PROGRESS_LOG.md)** - Current status and recent activity
  - What: Current phase, recent decisions, last 7 days of work
  - Use: Know where we are and what's happening now

### For Development Process
- **[AGENT_WORKFLOW.md](AGENT_WORKFLOW.md)** - Multi-agent development process
  - What: How specialized agents work in parallel, task lifecycle, handoff patterns
  - Use: Understand the development workflow

- **[TASKS.md](TASKS.md)** - Task queue for parallel development
  - What: Ready/blocked/completed tasks with dependencies and priorities
  - Use: Find work to do

- **[handoffs/](handoffs/)** - Detailed task specifications
  - What: Context, steps, acceptance criteria for specific tasks
  - Use: Pick up and implement a task

### For History
- **[SESSION_INDEX.md](SESSION_INDEX.md)** - Complete session history
  - What: Chronological record of all development sessions
  - Use: Find historical context and past decisions

- **[session-notes/](session-notes/)** - Detailed session notes
  - What: In-depth notes from each development session
  - Use: Deep dive into specific sessions

-----

## Multi-Agent Development Workflow

This project uses a **distributed multi-agent workflow** for development:

- Multiple specialized agents work in parallel from shared documentation
- Tasks are managed in [TASKS.md](TASKS.md) with clear dependencies
- Each task has a detailed handoff file in [handoffs/](handoffs/)
- No context loss between sessions - each agent starts fresh

**For developers:**
1. Read [AGENT_WORKFLOW.md](AGENT_WORKFLOW.md) to understand the process
2. Check [TASKS.md](TASKS.md) for available tasks
3. Pick up a task and read its handoff file
4. Follow [CLAUDE_CONTEXT.md](CLAUDE_CONTEXT.md) coding standards
5. Update session notes as you go

**Benefits:**
- 3x speedup through parallel execution
- Clear task handoffs eliminate confusion
- Resilient to context limits and session interruptions
- Scalable - add more agents as needed
