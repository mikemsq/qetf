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


## Repository structure

- `configs/`
  - Strategy, universe, schedule, and transaction cost configuration files.
  - These are intended to be the main inputs to research sweeps and production runs.
- `data/`
  - `raw/` immutable ingested data
  - `curated/` normalized clean data in a canonical schema
  - `snapshots/` versioned, reproducible "as-of" datasets used by backtests
- `artifacts/`
  - Output bundles from runs (metrics, plots, positions, recommendation packets).
- `notebooks/`
  - Exploration notebooks that import the library code from `src/`.
- `handoffs/`
  - Agent-to-agent handoff files for parallel development workflow.
  - See [AGENT_WORKFLOW.md](AGENT_WORKFLOW.md) for details.
- `scripts/`
  - Convenience scripts for common tasks (ingest, build snapshots, run backtests).
- `src/quantetf/`
  - The actual library, organized by domain modules:
    - `data/` ingestion connectors, storage, and quality checks
    - `universe/` universe builders and eligibility filters
    - `features/` feature definitions and a lightweight feature store
    - `alpha/` alpha model interfaces and example models
    - `risk/` covariance and exposure modeling
    - `portfolio/` constraints, allocators, transaction cost models
    - `backtest/` backtest engine and accounting
    - `evaluation/` metrics, comparisons, and reporting helpers
    - `production/` scheduled runtime that emits recommendation packets
    - `cli/` command entrypoints (thin wrappers around library calls)
    - `utils/` shared utilities
- `tests/`
  - Unit tests and smoke tests. Over time, this should include “golden tests” with fixed expected outputs.


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


## Using AI agents to implement this project

AI agents work best when the system is described as a set of contracts, examples, and acceptance tests.

### Recommended approach

1. **Write specs first, then ask agents to implement**
   - Add a `docs/` folder with short, concrete specs:
     - `ARCHITECTURE.md` describing the pipeline and module boundaries
     - `DATA_SCHEMA.md` with tables, fields, and example rows
     - `CONFIG_SPEC.md` describing YAML fields with examples
     - `BACKTEST_SPEC.md` defining the event timeline, fill assumptions, and accounting rules
     - `OUTPUT_SCHEMA.md` defining the recommendation packet formats
   - Keep each spec less than a few pages. Link them from this README.

2. **Drive implementation through tests and “golden artifacts”**
   - Provide a small deterministic dataset in `tests/fixtures/`.
   - Commit expected outputs (positions, returns, recommendation packet) for that dataset.
   - Ask agents to make tests pass, then expand scope.

3. **Decompose work into small, interface scoped tasks**
   - Examples:
     - “Implement `UniverseProvider` for YAML defined static universes.”
     - “Implement momentum alpha with leakage guard and unit tests.”
     - “Implement a transaction cost model based on turnover, plus tests.”
     - “Implement backtest accounting for equal weight portfolios.”
   - Each task should have a definition of done:
     - inputs, outputs, edge cases, and test expectations.

4. **Make agent context explicit**
   - Add an `AGENTS.md` file with:
     - code style rules (typing, docstrings, logging)
     - how to run tests, how to add fixtures
     - which module owns which responsibilities
     - “no lookahead” rules and how to enforce them

### High value artifacts for agents

- **Interface contracts**: typed dataclasses for key objects (Universe, FeatureFrame, AlphaScores, RiskModelOutput, TargetWeights).
- **Mermaid diagrams**: a pipeline flow and module dependency graph.
- **Example configs**: at least one complete strategy config that runs end to end.
- **Synthetic dataset + golden results**: small, deterministic, and checked into tests.
- **Run manifests**: a simple JSON manifest written by production runtime so agents can see what “reproducible” means.

If you want, I can also generate the `docs/` pack (spec files, diagrams, output schemas, and a set of golden tests) so agents can start implementing immediately.
