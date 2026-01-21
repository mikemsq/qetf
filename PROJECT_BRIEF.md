# Project Brief - QuantETF

**Last Updated:** January 17, 2026
**Project Status:** Strategy Search Phase - Finding Winning Strategy

-----

## 🎯 PRIMARY GOAL

**Build a systematic investment process that consistently outperforms the S&P 500 (SPY ETF) on a universe of ETFs.**

The alpha generation component uses historical data to generate, at any point in time, a portfolio of N ETFs designed to outperform SPY over the future time horizon. The investment process performs periodic rebalancing, generating a new optimized portfolio at each rebalance date.

Everything else (code, infrastructure, analytics) is a means to achieve this goal.

### Core Investment Process
1. **Alpha Generation**: Use historical data to score ETFs and identify expected outperformers
2. **Portfolio Construction**: Select top N ETFs and determine optimal weights
3. **Periodic Rebalancing**: Generate a new portfolio at each rebalance period (weekly/monthly)
4. **Continuous Improvement**: Evaluate performance and refine the process

### Success Criteria
- ✅ **Active Return > 0** consistently over multiple evaluation periods
- ✅ **Information Ratio > 0** demonstrating skill-based outperformance
- Strategy must work on **Tier 4 universe (200 ETFs)**
- Use **10 years of historical data** for robust evaluation and validation

-----

## Project Overview

### What are we building?

QuantETF is a modular quantitative investment platform for ETF-based strategies. It provides:

- **Systematic investment process** that generates portfolios to outperform SPY
- **Research tools** for exploring datasets, prototyping signals, and running backtests
- **Production pipeline** that generates periodic rebalancing recommendations

**The infrastructure exists to serve one purpose: build and operate a winning systematic investment process.**

### Problem Statement

Individual investors and small fund managers need:

1. A systematic way to construct portfolios that outperform market benchmarks
1. Reproducible backtests that avoid common pitfalls (lookahead bias, survivorship bias)
1. Automated periodic rebalancing recommendations without requiring full brokerage integration
1. Ability to experiment with different alpha signals and portfolio construction methods

Current solutions are either:

- Too expensive (institutional platforms like Bloomberg)
- Too simple (basic backtesting tools without proper point-in-time handling)
- Too complex (require significant infrastructure to set up)

### Success Criteria

**Primary goal (MUST ACHIEVE):**

- [ ] **Build an investment process that consistently outperforms SPY**
- [ ] Active Return > 0 across multiple evaluation windows (1-year, 3-year, 5-year)
- [ ] Information Ratio > 0 demonstrating systematic skill

**Secondary goals:**

- [x] Reproducible backtests with documented methodology
- [x] Automated strategy optimizer to search parameter space
- [ ] Weekly/monthly rebalancing recommendations in production
- [ ] Clear audit trail from data → signals → portfolio → recommendations

**Evaluation Standard:**

| Period | Requirement |
|--------|-------------|
| 1-year | Active Return > 0, IR > 0 |
| 3-year | Active Return > 0, IR > 0 |
| 5-year | Active Return > 0, IR > 0 |

A strategy WINS if it consistently beats SPY across multiple time horizons.

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
1. **Universe size:** **Tier 4 (200 ETFs)** is the primary universe for strategy search
1. **Data period:** 10 years of historical data (2016-2026)
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
1. **Decision:** Cash as explicit portfolio holding with "$CASH$" ticker
- **Reasoning:** Ensures realistic NAV calculations, prevents NAV from dropping to 0 when strategy stays in cash, maintains proper financial modeling throughout backtest lifecycle
- **Trade-offs:** Slightly more complex portfolio construction logic, but essential for accurate backtesting and realistic results

-----

## Project Phases

### Phase 1: Foundation ✅ COMPLETE

**Goal:** Set up project infrastructure and data pipeline

**Deliverables:**

- [x] Project structure and documentation (CLAUDE_CONTEXT, PROGRESS_LOG)
- [x] Data ingestion for at least one ETF price source
- [x] Curated data store with basic quality checks
- [x] Simple universe definition (static list of 20 ETFs)
- [x] Basic data validation tests

**Success Metrics:** Can load historical prices for 20 ETFs and create a snapshot

### Phase 2: Backtest Engine ✅ COMPLETE

**Goal:** Build working backtest engine with one simple strategy

**Deliverables:**

- [x] Backtest engine architecture designed
- [x] Point-in-time data access (no lookahead)
- [x] Simple alpha model (12-month momentum)
- [x] Portfolio constructor (equal weight top N)
- [x] Transaction cost model (simple flat fee)
- [x] Basic metrics (returns, Sharpe ratio, drawdown)
- [x] End-to-end backtest working

**Success Metrics:** ✅ Can run a complete backtest and generate equity curve (101 tests passing)

### Phase 3: Analytics & Visualization (IN PROGRESS - 0%)

**Goal:** Build comprehensive analytical tools to evaluate and improve strategies

**Deliverables:**

- [ ] Enhanced metrics module (Sortino, Calmar, VaR, rolling Sharpe, etc.)
- [ ] Backtest visualization notebook (equity curves, heatmaps, drawdowns)
- [ ] Risk analytics module (correlations, beta, concentration metrics)
- [ ] Alpha diagnostics (IC analysis, signal decay, quintile performance)
- [ ] Strategy comparison framework
- [ ] Benchmark comparison (vs SPY, 60/40, oracle, random)
- [ ] Walk-forward validation (prevent overfitting)
- [ ] Parameter sensitivity analysis
- [ ] Transaction cost analysis with realistic models
- [ ] Stress testing (crisis period analysis)
- [ ] Auto-report generation (HTML reports)
- [ ] Data quality monitoring

**Success Metrics:** Can rigorously evaluate strategy robustness, compare variants, and generate professional reports

### Phase 4: Strategy Development

**Goal:** Implement multiple alpha signals and portfolio construction methods

**Deliverables:**

- [ ] 2-3 additional alpha models
- [ ] Mean-variance optimization for portfolio construction
- [ ] Risk model (simple covariance estimation)
- [ ] Multi-factor alpha combiner
- [ ] Risk parity constructor

**Success Metrics:** Can compare multiple strategies and identify which performs best

### Phase 5: Production Pipeline

**Goal:** Automate recommendation generation

**Deliverables:**

- [ ] Production runtime that runs on schedule
- [ ] Recommendation packet generator (CSV/JSON output)
- [ ] Run manifest (reproducibility metadata)
- [ ] Data freshness checks
- [ ] Basic alerting for failures

**Success Metrics:** Can generate recommendations on schedule

### Phase 6: Polish & Documentation

**Goal:** Make the system production-ready and well-documented

**Deliverables:**

- [ ] Comprehensive README with examples
- [ ] Tutorial notebook walking through the system
- [ ] Documentation of all modules
- [ ] Golden test suite (known expected outputs)
- [ ] Performance optimization if needed

**Success Metrics:** New user can clone repo and run first backtest quickly

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

- Solo project, decisions documented in STATUS.md
- Major architecture decisions recorded in session notes
- Strategy choices documented in CLAUDE_CONTEXT.md

### Progress Updates

- **Frequency:** Daily (STATUS.md updates)
- **Format:** Session notes for substantial work

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

- **Issues closed:** Track progress on backlog
- **Time to run backtest:** Target < 5 minutes for 5-year history

-----

## Appendix

### Related Documents

- [CLAUDE_CONTEXT.md](CLAUDE_CONTEXT.md) - Coding standards and patterns
- [AGENT_WORKFLOW.md](AGENT_WORKFLOW.md) - Multi-agent development process
- [STATUS.md](STATUS.md) - Current status and recent activity
- [TASKS.md](TASKS.md) - Task queue for parallel development
- [README.md](README.md) - Public-facing overview

### Change Log

|Date      |Change              |Reason                                          |
|----------|--------------------|------------------------------------------------|
|2026-01-05|Initial draft       |Project kickoff                                 |
|2026-01-07|Comprehensive update|Filled in all sections, added phases and metrics|
|2026-01-09|Removed timelines   |Align with multi-agent workflow, focus on phases not dates|

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

**Current Phase:** Phase 2 - Backtest Engine (30% complete)

**Current Focus:** Complete portfolio construction, transaction costs, and backtest engine

**See:** [STATUS.md](STATUS.md) for detailed current status and [TASKS.md](TASKS.md) for ready tasks