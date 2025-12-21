# QuantETF Strategy Platform

A research and production-ready Python codebase for ETF strategy development.

Core capabilities:
- Acquire and curate ETF market data and metadata
- Define dynamic investable universes (eligibility as-of each rebalance date)
- Compute features, generate alpha scores, and estimate risk
- Construct portfolios under constraints and transaction cost assumptions
- Backtest with reproducible data snapshots and config-driven experiments
- Run a scheduled production pipeline that outputs trading recommendations (no automated trading)

This repository is intentionally scaffolded as a modular platform:
- Research mode: notebooks + CLI for repeatable runs
- Production mode: scheduled pipeline that produces versioned artifacts

## Quick start (conceptual)
1. Ingest data into `data/raw/`, then curate into `data/curated/`
2. Create a dataset snapshot in `data/snapshots/` (a versioned, point-in-time dataset bundle)
3. Run backtests from config in `configs/`
4. Run the production pipeline to generate a dated recommendation packet

## Repository map
- `configs/` Config files for universes, strategies, schedules, and cost models
- `data/` Local data zones (raw, curated, snapshots)
- `src/quantetf/` The Python package with all platform components
- `notebooks/` Exploratory research notebooks (kept thin, call into library code)
- `artifacts/` Output from backtests and production runs (reports, plots, CSVs)
- `tests/` Unit tests and smoke tests
