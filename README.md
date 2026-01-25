# QuantETF

QuantETF is a systematic investment platform for ETF-based quantitative strategies designed to consistently outperform the S&P 500 (SPY).

## Primary Objective

**Build a systematic investment process that consistently outperforms SPY on a universe of ETFs.**

The platform uses historical data to generate, at any point in time, a portfolio of N ETFs designed to outperform SPY over the future time horizon. The investment process performs periodic rebalancing, generating a new optimized portfolio at each rebalance date.

### Regime-Aware Alpha Generation

The alpha generation process is designed to be **regime-aware**:

1. **Market State Analysis**: Analyze current market conditions based on historical price behavior and macroeconomic indicators (VIX, interest rates, credit spreads, etc.)
2. **Regime Detection**: Classify the market regime (e.g., risk-on, risk-off, high volatility, trending, mean-reverting)
3. **Model Selection**: Select the most appropriate alpha model or model weights for the detected regime
4. **Portfolio Generation**: Generate the optimal portfolio for the current market environment

This adaptive approach aims to improve consistency by using strategies suited to current conditions rather than a single static model.

### Success Criterion

**The investment process must demonstrate positive active returns (outperformance vs SPY) for at least 80% of rebalance cycles during the backtest period.**

This measures consistency rather than just aggregate performance - a strategy that beats SPY in 80%+ of rebalance periods is more robust and reliable than one with high total return but erratic period-by-period performance.

## Two Modes of Operation

- **Research**: explore datasets, define investable universes, prototype alpha signals, run backtests, and compare strategies to find winning configurations.
- **Production**: run a scheduled pipeline that outputs periodic **rebalancing recommendations** (target portfolio, trades, and audit trail). This project does not include automated order execution.

The codebase is structured so that you can swap components (data sources, alpha models, risk models, portfolio construction, rebalancing schedules) without rewriting the backtest engine or production runtime.


## Goals

- Build an investment process that generates portfolios to outperform SPY through periodic rebalancing.
- Make strategy research reproducible with versioned datasets and config-driven experiments.
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
4. **Detect market regime** using macro indicators and price behavior.
5. **Select alpha model(s)** appropriate for the detected regime.
6. **Score the universe** using the selected alpha model(s).
7. **Estimate risk** (covariance and exposures) as-of the same date.
8. **Construct a portfolio** (target weights) subject to constraints and costs.
9. **Simulate** holdings and returns (backtest) or **emit recommendations** (production).


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
- **`scripts/`** - Utility scripts (ingest.sh, backtest)
- **`notebooks/`** - Jupyter notebooks for research
- **`tests/`** - Unit tests, integration tests, golden tests
- **`artifacts/`** - Output from runs (metrics, plots, recommendations)

### Documentation
- **`handoffs/`** - Task handoffs and completions (organized by type)
- **`docs/`** - Reference documentation and handouts

See **Documentation Guide** section below for file purposes.


## Core abstractions

QuantETF is intentionally interface first. The important interfaces you will see across modules include:

- **Data**: ingestors/connectors, curated stores, snapshot selection by `DatasetVersion`.
- **UniverseProvider**: returns the eligible ETF list for a given as-of date.
- **FeatureComputer**: produces a feature matrix as-of a date for a universe.
- **RegimeDetector**: classifies market regime using macro indicators and price behavior as-of a date.
- **AlphaModel**: produces scores (or expected returns) for a universe as-of a date.
- **AlphaSelector**: selects or weights alpha models based on the detected regime.
- **RiskModel**: produces covariance and exposure summaries as-of a date.
- **PortfolioConstructor**: converts alpha + risk + constraints into target weights.
- **TransactionCostModel**: estimates costs from turnover and liquidity proxies.
- **BacktestEngine**: orchestrates a run on a schedule and returns a result bundle.
- **ProductionRuntime**: runs "today", writes a recommendation packet, and records a manifest.

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

This project uses 6 essential root-level documentation files:

| File | Purpose |
|------|---------|
| [README.md](README.md) | This file - public overview |
| [PROJECT_BRIEF.md](PROJECT_BRIEF.md) | Vision, goals, architecture |
| [STATUS.md](STATUS.md) | Current status - check first |
| [TASKS.md](TASKS.md) | Task queue |
| [CLAUDE_CONTEXT.md](CLAUDE_CONTEXT.md) | Coding standards |
| [AGENT_WORKFLOW.md](AGENT_WORKFLOW.md) | Agent roles and workflow |

### Handoffs Directory

Task handoffs and completions are organized in [handoffs/](handoffs/):

```
handoffs/
├── architecture/     # Long-lived architectural docs
├── research/         # Research findings
├── tasks/            # Active task handoffs
└── completions/      # Task completion records
```

### For New Agents

1. Read [STATUS.md](STATUS.md) for current project state
2. Check [TASKS.md](TASKS.md) for available tasks
3. Read [CLAUDE_CONTEXT.md](CLAUDE_CONTEXT.md) for coding standards
4. Read the relevant handoff file from [handoffs/tasks/](handoffs/tasks/)

-----

## Multi-Agent Development Workflow

This project uses a **distributed multi-agent workflow**:

- Multiple specialized agents work in parallel from shared documentation
- Tasks managed in [TASKS.md](TASKS.md) with clear dependencies
- Detailed handoffs in [handoffs/](handoffs/) for each task
- No context loss - each agent starts fresh with explicit documentation

See [AGENT_WORKFLOW.md](AGENT_WORKFLOW.md) for details on agent roles and workflow
