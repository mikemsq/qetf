# Project Brief - QuantETF

**Last Updated:** January 7, 2026  
**Project Status:** Planning / Initial Development  
**Target Completion:** 8-12 weeks for MVP

-----

## Project Overview

### What are we building?

QuantETF is a modular quantitative investment platform for ETF-based strategies. It provides:

- **Research tools** for exploring datasets, prototyping signals, and running backtests
- **Production pipeline** that generates automated rebalancing recommendations
- **Modular architecture** allowing easy swapping of data sources, alpha models, risk models, and portfolio construction methods

The goal is to create and maintain a portfolio of ETFs that exceeds S&P 500 returns through systematic rebalancing based on quantitative signals.

### Problem Statement

Individual investors and small fund managers need:

1. A way to systematically evaluate ETF investment strategies
1. Reproducible backtests that avoid common pitfalls (lookahead bias, survivorship bias)
1. Automated trade recommendations without requiring full brokerage integration
1. Ability to experiment with different alpha signals and portfolio construction methods

Current solutions are either:

- Too expensive (institutional platforms like Bloomberg)
- Too simple (basic backtesting tools without proper point-in-time handling)
- Too complex (require significant infrastructure to set up)

### Success Criteria

**Primary goal:**

- [ ] Generate a portfolio that demonstrates superior risk-adjusted returns vs S&P 500 in backtest (Sharpe ratio > 0.8)

**Secondary goals:**

- [ ] Reproducible backtests with documented methodology
- [ ] Weekly rebalancing recommendations in production
- [ ] Modular codebase where components can be swapped without rewriting the engine
- [ ] Clear audit trail from data → signals → portfolio → recommendations

-----

## Project Scope

### In Scope (MVP)

**Core Infrastructure:**

- Data ingestion from at least one ETF pricing source
- Point-in-time data snapshots (versioned, reproducible)
- Universe definition (static list or rule-based)
- Backtest engine with realistic assumptions

**Alpha & Portfolio:**

- At least one alpha model (e.g., momentum-based)
- Basic portfolio construction (equal weight or optimization)
- Transaction cost modeling
- Rebalancing logic (weekly or monthly)

**Production:**

- Scheduled pipeline that generates trade recommendations
- Output format: CSV/JSON with target weights and trades
- Run manifest (code version, data version, timestamp)

**Research:**

- Jupyter notebooks for exploration
- Basic visualization of equity curves and metrics
- Comparison framework (strategy vs S&P 500)

### Out of Scope (v1)

**Not included:**

- Live order execution / brokerage integration
- Intraday trading or high-frequency strategies
- Options, futures, or non-ETF instruments
- Real-time streaming data
- Complex machine learning models (v1 focuses on simple signals)
- Web UI (terminal/notebook interface only for v1)

**Future considerations:**

- Multi-strategy portfolio allocation
- Risk parity and volatility targeting
- Tax-loss harvesting
- Integration with portfolio tracking tools

### Assumptions

1. **Data availability:** We can access historical ETF prices via API or CSV
1. **Universe size:** Working with 50-200 ETFs (not thousands)
1. **Rebalancing frequency:** Weekly or monthly (not daily)
1. **Computing resources:** Local machine or modest cloud VM is sufficient
1. **User technical skill:** Comfortable with Python, terminal, and Jupyter notebooks
1. **No real-time constraints:** Recommendations can be generated overnight

### Constraints

**Technical:**

- Must avoid lookahead bias (strict point-in-time data handling)
- Must be deterministic (same inputs → same outputs)
- Must handle missing data gracefully

**Time:**

- Aiming for MVP in 8-12 weeks
- Initial focus on getting one strategy working end-to-end

**Data:**

- Free/low-cost data sources preferred initially
- May need to budget for data if free sources insufficient

**Regulatory:**

- This is not investment advice
- No claims about future performance
- Clear disclaimers in all documentation

-----

## Target Audience

### Primary Users

**User Type 1: Quantitative Retail Investor**

- Has programming background (Python comfortable)
- Interested in systematic investing
- Wants to backtest ideas before committing capital
- Needs: Reproducible research environment, clear methodology

**User Type 2: Small Fund Manager / RIA**

- Managing small AUM ($1M-$50M)
- Wants to offer systematic strategies
- Needs audit trail for compliance
- Needs: Production recommendations, explainable signals

**User Type 3: Finance Student / Researcher**

- Learning about quantitative finance
- Wants to understand backtest mechanics
- Experimenting with different approaches
- Needs: Clear code, educational examples, documentation

### User Stories

1. As a **retail investor**, I want to **test a momentum strategy on a universe of sector ETFs** so that **I can decide if it’s worth implementing**
1. As a **fund manager**, I want to **receive weekly trade recommendations with justification** so that **I can review and execute them for my clients**
1. As a **researcher**, I want to **compare multiple strategies side-by-side** so that **I can understand which signals add value**
1. As a **developer**, I want to **swap in a new alpha model without rewriting the backtest engine** so that **I can iterate quickly on ideas**
1. As a **compliance officer**, I want to **see an audit trail from raw data to final recommendation** so that **I can verify the process is sound**

-----

## Technical Approach

### Technology Stack

- **Language:** Python 3.10+
- **Package Manager:** uv (for fast, reproducible dependency management)
- **Data Processing:** pandas, numpy
- **Backtesting:** Custom engine (src/quantetf/backtest)
- **Optimization:** scipy (for portfolio optimization if needed)
- **Visualization:** matplotlib, seaborn (for notebooks)
- **Testing:** pytest
- **Config:** YAML files for strategies and universes
- **Storage:** Local filesystem (CSV/Parquet for now)
- **Notebooks:** Jupyter for research
- **Version Control:** Git/GitHub
- **Deployment (future):** Docker for production pipeline

### Architecture Overview

```
Data Pipeline:
Raw Data → Ingestion → Curated Store → Versioned Snapshots
                                              ↓
Research Loop:                         Point-in-time Data
Universe Definition → Feature Engineering → Alpha Models
                            ↓                    ↓
                      Risk Models ← Portfolio Construction
                                              ↓
                                    Backtest Engine
                                              ↓
                                    Metrics & Visualization

Production Pipeline:
Scheduled Trigger → Load Latest Snapshot → Compute Signals
                                              ↓
                          Portfolio Optimization → Transaction Cost Model
                                              ↓
                                    Recommendation Packet
                                              ↓
                              CSV/JSON Output + Run Manifest
```

### Key Technical Decisions

1. **Decision:** Custom backtest engine instead of using existing libraries (zipline, backtrader)
- **Reasoning:** Full control over point-in-time handling, simpler codebase, easier to understand
- **Trade-offs:** More work upfront, but cleaner architecture for our specific needs
1. **Decision:** File-based storage (CSV/Parquet) instead of database
- **Reasoning:** Simpler for v1, version control friendly, sufficient for <200 ETFs
- **Trade-offs:** Less scalable, but adequate for MVP
1. **Decision:** Config-driven strategies (YAML) instead of code-based strategies
- **Reasoning:** Easier to version, compare, and document different strategy variants
- **Trade-offs:** Less flexible than pure Python, but forces cleaner abstractions
1. **Decision:** Python dataclasses for core objects instead of dictionaries
- **Reasoning:** Type safety, IDE support, self-documenting
- **Trade-offs:** Slightly more verbose, but much clearer
1. **Decision:** Explicit as_of dates for all operations
- **Reasoning:** Prevents lookahead bias, makes time-travel clear
- **Trade-offs:** More parameter passing, but essential for correctness

-----

## Project Phases

### Phase 1: Foundation (Weeks 1-2)

**Goal:** Set up project infrastructure and data pipeline

**Deliverables:**

- [ ] Project structure and documentation (CLAUDE_CONTEXT, PROGRESS_LOG)
- [ ] Data ingestion for at least one ETF price source
- [ ] Curated data store with basic quality checks
- [ ] Simple universe definition (static list of 20 ETFs)
- [ ] Basic data validation tests

**Success Metrics:** Can load historical prices for 20 ETFs and create a snapshot

### Phase 2: Backtest Engine (Weeks 3-4)

**Goal:** Build working backtest engine with one simple strategy

**Deliverables:**

- [ ] Backtest engine that handles rebalancing schedule
- [ ] Point-in-time data access (no lookahead)
- [ ] Simple alpha model (e.g., 12-month momentum)
- [ ] Portfolio constructor (equal weight top N)
- [ ] Transaction cost model (simple flat fee)
- [ ] Basic metrics (returns, Sharpe ratio, drawdown)

**Success Metrics:** Can run a complete backtest and generate equity curve

### Phase 3: Strategy Development (Weeks 5-6)

**Goal:** Implement multiple alpha signals and portfolio construction methods

**Deliverables:**

- [ ] 2-3 additional alpha models
- [ ] Mean-variance optimization for portfolio construction
- [ ] Risk model (simple covariance estimation)
- [ ] Enhanced metrics and visualization
- [ ] Strategy comparison framework

**Success Metrics:** Can compare multiple strategies and identify which performs best

### Phase 4: Production Pipeline (Weeks 7-8)

**Goal:** Automate recommendation generation

**Deliverables:**

- [ ] Production runtime that runs on schedule
- [ ] Recommendation packet generator (CSV/JSON output)
- [ ] Run manifest (reproducibility metadata)
- [ ] Data freshness checks
- [ ] Basic alerting for failures

**Success Metrics:** Can generate weekly recommendations automatically

### Phase 5: Polish & Documentation (Weeks 9-10)

**Goal:** Make the system production-ready and well-documented

**Deliverables:**

- [ ] Comprehensive README with examples
- [ ] Tutorial notebook walking through the system
- [ ] Documentation of all modules
- [ ] Golden test suite (known expected outputs)
- [ ] Performance optimization if needed

**Success Metrics:** New user can clone repo and run first backtest in < 30 minutes

-----

## Risks & Mitigation

|Risk                                                  |Impact|Probability|Mitigation Strategy                                                               |
|------------------------------------------------------|------|-----------|----------------------------------------------------------------------------------|
|Data quality issues (missing, incorrect prices)       |High  |Medium     |Implement robust validation, multiple data sources, clear handling of missing data|
|Lookahead bias in backtest                            |High  |Medium     |Strict point-in-time data access, code reviews, golden tests                      |
|Overfitting strategies to historical data             |High  |High       |Walk-forward validation, simple signals first, document all experiments           |
|Free data sources insufficient                        |Medium|Medium     |Budget for paid API if needed, start with free sources                            |
|Scope creep (adding features vs finishing MVP)        |Medium|High       |Stick to defined MVP, maintain “future features” backlog                          |
|Transaction costs higher than expected in live trading|Medium|Low        |Conservative cost estimates, paper trading before real money                      |

-----

## Resources & Dependencies

### Required Resources

- **Time Commitment:** 10-15 hours per week
- **Budget:**
  - $0-50/month for data API (if free sources insufficient)
  - $20/month Claude Pro (already have)
  - Minimal cloud costs (can run locally initially)
- **Tools/Services:**
  - GitHub account (have)
  - Python development environment
  - Jupyter notebooks
  - ETF data API access (TBD)

### External Dependencies

**Data providers:**

- Need to select and integrate ETF pricing source
- Options: yfinance (free), Alpha Vantage (free tier), others
- Impact: Core dependency, affects data quality

**Market data availability:**

- Dependent on data provider uptime
- Historical data completeness varies by provider

### Knowledge Gaps

**Need to learn:**

1. Best practices for ETF data (adjustments, corporate actions)
1. Realistic transaction cost modeling for retail investors
1. Portfolio optimization techniques (mean-variance, risk parity)
1. Walk-forward validation methodology
1. Production scheduling and monitoring

**Research needed:**

1. Survey of ETF data providers (free vs paid)
1. Literature review of common alpha signals for ETFs
1. Understanding of ETF-specific issues (tracking error, creation/redemption)

-----

## Communication & Collaboration

### Stakeholders

- **Primary user:** Myself (mikemsq)
- **Potential future users:** Open source community if project is made public

### Decision-Making Process

- Solo project, decisions documented in PROGRESS_LOG.md
- Major architecture decisions recorded in session notes
- Strategy choices documented in CLAUDE_CONTEXT.md

### Progress Updates

- **Frequency:** Daily (PROGRESS_LOG.md updates)
- **Format:** Session notes for substantial work
- **Weekly reviews:** End-of-week summary in PROGRESS_LOG.md

-----

## Success Metrics & KPIs

### Launch Criteria (MVP Complete)

- [ ] Can ingest and store ETF price data
- [ ] Can run reproducible backtest with documented methodology
- [ ] At least one strategy shows Sharpe ratio > 0.8 in backtest
- [ ] Production pipeline can generate weekly recommendations
- [ ] Comprehensive test suite passes
- [ ] Documentation sufficient for new user to get started

### Post-Launch Metrics

**Performance metrics:**

- **Backtest Sharpe Ratio:** Target > 0.8 (vs S&P 500 baseline)
- **Max Drawdown:** Target < 25%
- **Win Rate:** Track % of positive months
- **Turnover:** Monitor transaction costs impact

**Code quality metrics:**

- **Test Coverage:** Target > 80% for core modules
- **Documentation:** All public functions have docstrings
- **Type Coverage:** All functions have type hints

**Development velocity:**

- **Weekly commits:** Track consistency
- **Issues closed:** Track progress on backlog
- **Time to run backtest:** Target < 5 minutes for 5-year history

-----

## Appendix

### Related Documents

- [CLAUDE_CONTEXT.md](https://raw.githubusercontent.com/mikemsq/qetf/refs/heads/main/CLAUDE_CONTEXT.md) - Coding standards
- [PROGRESS_LOG.md](https://raw.githubusercontent.com/mikemsq/qetf/refs/heads/main/PROGRESS_LOG.md) - Daily progress
- [README.md](https://github.com/mikemsq/qetf) - Project overview
- [Boris Cherny’s Agentic Workflow](https://threadreaderapp.com/thread/2007179832300581177.html) - Development methodology

### Change Log

|Date      |Change              |Reason                                          |
|----------|--------------------|------------------------------------------------|
|2026-01-05|Initial draft       |Project kickoff                                 |
|2026-01-07|Comprehensive update|Filled in all sections, added phases and metrics|

### References

**ETF Data:**

- [yfinance documentation](https://github.com/ranaroussi/yfinance)
- [Alpha Vantage API](https://www.alphavantage.co/)

**Quantitative Finance:**

- “Quantitative Trading” by Ernie Chan
- “Algorithmic Trading” by Ernest Chan
- [QuantStart articles](https://www.quantstart.com/)

**Backtesting Best Practices:**

- [Common backtesting mistakes](https://www.quantstart.com/articles/Backtesting-Biases/)
- Walk-forward optimization methodology

-----

## Quick Reference

**Elevator Pitch:** A Python-based platform for researching and deploying systematic ETF strategies with reproducible backtests and automated rebalancing recommendations.

**One-Week Goal:** Set up data ingestion and create first snapshot of 20 ETF prices

**Current Blocker:** Need to select ETF data provider (evaluate yfinance, Alpha Vantage)

**Next Action:** Research ETF data providers and implement basic ingestion connector