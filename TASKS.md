# Task Queue - QuantETF

**Last Updated:** January 11, 2026
**Active Phase:** Phase 3 - Analytics & Visualization

## Task Status Legend

- `ready` - Available for pickup by any agent
- `in_progress` - Currently being worked on
- `blocked` - Waiting on dependencies
- `completed` - Implementation done, needs review
- `merged` - Reviewed and merged to main

---

## Current Sprint: Phase 2 Backtest Engine

### IMPL-002: EqualWeightTopN Portfolio Constructor
**Status:** completed
**Priority:** high
**Completed:** 2026-01-09
**Dependencies:** []
**Assigned:** Session-IMPL-002

**Description:**
Implement EqualWeightTopN portfolio constructor that takes top N ETFs by alpha score and assigns equal weights.

**Files:**
- Created: `src/quantetf/portfolio/equal_weight.py`
- Created: `tests/test_equal_weight.py`

**Handoff:** `handoffs/handoff-IMPL-002.md`
**Completion:** `handoffs/completion-IMPL-002.md`

**Notes:**
- All 14 tests pass (covers edge cases, NaN handling, large universes)
- Handles fewer valid scores than top_n gracefully
- Comprehensive diagnostics for debugging
- Ready for integration in backtest engine

---

### IMPL-003: FlatTransactionCost Model
**Status:** completed
**Priority:** high
**Estimated:** 1 hour
**Completed:** 2026-01-09
**Dependencies:** []
**Assigned:** Session-IMPL-003

**Description:**
Implement simple flat transaction cost model (10 bps per trade).

**Files:**
- Update: `src/quantetf/portfolio/costs.py`
- Tests: `tests/test_transaction_costs.py`

**Handoff:** `handoffs/handoff-IMPL-003.md`

**Notes:**
- Implemented as dataclass following existing CostModel pattern
- 22 comprehensive tests, all passing
- Handles edge cases (empty/None weights, NaN values, misaligned tickers)
- Returns cost as fraction of NAV (e.g., 0.001 = 10 bps)

---

### IMPL-004: Simple Backtest Engine
**Status:** completed
**Priority:** high
**Completed:** 2026-01-10
**Dependencies:** [IMPL-002, IMPL-003] (both completed)
**Assigned:** Session-IMPL-004-Resume

**Description:**
Implement SimpleBacktestEngine that orchestrates the event-driven backtest loop.

**Files:**
- Created: `src/quantetf/backtest/simple_engine.py`
- Created: `tests/test_backtest_engine.py`

**Handoff:** `handoffs/handoff-IMPL-004.md`
**Completion:** `handoffs/completion-IMPL-004.md`

**Notes:**
- 353 lines of implementation + 475 lines of tests
- 17 comprehensive tests, all passing
- Event-driven architecture with T-1 data access
- Integrates MomentumAlpha, EqualWeightTopN, FlatTransactionCost
- Total test count: 68 → 85 (+17)
- Ready for IMPL-005

---

### TEST-001: No-Lookahead Tests
**Status:** completed
**Priority:** critical
**Estimated:** 2 hours
**Completed:** 2026-01-09
**Dependencies:** []
**Assigned:** Testing Agent

**Description:**
Create synthetic data tests to verify no lookahead bias in data access and alpha models.

**Files:**
- Created: `tests/test_no_lookahead.py`

**Handoff:** `handoffs/handoff-TEST-001.md`
**Completion:** `handoffs/completion-TEST-001.md`

**Notes:**
- 8 comprehensive tests all passing
- Synthetic data approach makes lookahead bugs visible
- Verifies SnapshotDataStore T-1 enforcement
- Verifies MomentumAlpha uses only historical data
- Strong confidence in no-lookahead enforcement

---

### IMPL-005: End-to-End Backtest Script
**Status:** completed
**Priority:** high
**Estimated:** 2 hours
**Completed:** 2026-01-10
**Dependencies:** [IMPL-004] (completed)
**Assigned:** Session-IMPL-005

**Description:**
Create script to run complete backtest on 5-year snapshot and generate results.

**Files:**
- Created: `scripts/run_backtest.py`
- Created: `tests/test_run_backtest.py`

**Handoff:** `handoffs/handoff-IMPL-005.md`
**Completion:** `handoffs/completion-IMPL-005.md`

**Notes:**
- 350+ lines of implementation with CLI interface
- 16 comprehensive tests, all passing
- Successfully runs on real 5yr snapshot data
- Generates equity curve, metrics, holdings, weights, config
- Sample results: 66.9% return, 1.50 Sharpe, -9.8% max drawdown (2023-2025)
- Phase 2 complete (100%)

---

## Completed Tasks

### IMPL-001: MomentumAlpha Model
**Status:** completed
**Priority:** high
**Completed:** 2026-01-09
**Agent:** Session-001

**Description:**
Implemented MomentumAlpha class with 252-day lookback and T-1 data access.

**Files:**
- Updated: `src/quantetf/alpha/momentum.py`
- Created: `src/quantetf/data/snapshot_store.py`

**Notes:**
- Includes comprehensive logging
- Handles missing data gracefully
- Ready for integration testing

---

### INFRA-001: Snapshot Data Store
**Status:** completed
**Priority:** high
**Completed:** 2026-01-09
**Agent:** Session-001

**Description:**
Implemented SnapshotDataStore for point-in-time data access from parquet files.

**Files:**
- Created: `src/quantetf/data/snapshot_store.py`

**Notes:**
- Enforces T-1 data access (no lookahead)
- Supports lookback windows
- Handles MultiIndex format correctly

---

## Current Sprint: Phase 3 - Analytics & Visualization

### ANALYSIS-001: Enhanced Metrics Module
**Status:** completed
**Priority:** high
**Estimated:** 2-3 hours
**Completed:** 2026-01-11
**Dependencies:** []
**Assigned:** Wave 1 Sequential Execution

**Description:**
Expand evaluation/metrics.py with advanced performance metrics for comprehensive strategy analysis.

**New Metrics Added:**
- Sortino ratio (downside-risk adjusted returns) ✅
- Calmar ratio (return / max drawdown) ✅
- Win rate (percentage of positive periods) ✅
- Value at Risk (VaR) and Conditional VaR (CVaR) at 95% confidence ✅
- Rolling Sharpe ratio (252-day window) ✅
- Information ratio (vs benchmark) ✅

**Files:**
- Updated: `src/quantetf/evaluation/metrics.py` (+350 lines, 7 new metrics)
- Created: `tests/test_advanced_metrics.py` (37 tests)

**Results:**
- All 7 metrics implemented with comprehensive docstrings
- 37 tests created (exceeds target of 21+)
- All tests passing (100%)
- Validated with real backtest data
- Total test count: 138 (was 101, +37)
- Metrics include edge case handling, proper error messages, and examples

**Handoff:** `handoffs/handoff-ANALYSIS-001.md`

---

### ANALYSIS-002: Risk Analytics Module
**Status:** completed
**Priority:** high
**Estimated:** 2-3 hours
**Completed:** 2026-01-11
**Dependencies:** [ANALYSIS-001] (completed)
**Assigned:** ANALYSIS-002 Session

**Description:**
Create new risk_analytics.py module for portfolio risk decomposition and analysis.

**Functionality:**
- Correlation matrix of holdings over time ✅
- Portfolio beta to benchmark (SPY, QQQ) ✅
- Portfolio alpha (Jensen's alpha) ✅
- Volatility clustering detection ✅
- Concentration metrics (HHI, effective N) ✅
- Exposure summary (sector, geography if available) ✅
- Drawdown analysis ✅
- Rolling correlation ✅
- Tail ratio analysis ✅

**Files:**
- Created: `src/quantetf/evaluation/risk_analytics.py` (11 functions, 400+ lines)
- Created: `tests/test_risk_analytics.py` (45 tests)
- Created: `scripts/validate_risk_analytics.py` (validation script)

**Results:**
- 11 risk analytics functions implemented (exceeds target of 5)
- 45 comprehensive tests (exceeds target of 15)
- All tests passing (100%)
- Validated with real 5-year backtest data
- Clear docstrings with financial context and examples
- Total test count: 183 (was 138, +45)

**Notes:**
- Comprehensive coverage: beta, alpha, correlation, concentration, volatility, drawdowns, tail risk
- Works seamlessly with backtest results format (weights_history, holdings_history, equity curve)
- Validation script demonstrates all functions on real data
- Ready for use in VIZ-001 (Backtest Analysis Notebook)

---

### VIZ-001: Backtest Analysis Notebook
**Status:** ready
**Priority:** high
**Estimated:** 3-4 hours
**Dependencies:** [ANALYSIS-001]

**Description:**
Create comprehensive Jupyter notebook for visualizing backtest results.

**Required Visualizations:**
1. Equity curve with dual-axis drawdown overlay
2. Monthly/yearly returns heatmap
3. Rolling Sharpe ratio (252-day window)
4. Drawdown waterfall chart
5. Returns distribution histogram
6. Underwater plot (time below high-water mark)
7. Holdings evolution over time
8. Turnover analysis

**Files:**
- Create: `notebooks/backtest_analysis.ipynb`
- Update: `notebooks/README.md`

**Acceptance Criteria:**
- Loads latest backtest from artifacts/backtests/
- All 8 visualizations implemented
- Clear markdown explanations
- Runs end-to-end without errors
- Generates professional-looking charts

**Notes:**
- Use matplotlib/seaborn for consistency
- Make plots interactive where beneficial (plotly optional)
- Include summary metrics table at top

---

### VIZ-002: Alpha Diagnostics Notebook
**Status:** ready
**Priority:** medium
**Estimated:** 2-3 hours
**Dependencies:** [ANALYSIS-001]

**Description:**
Create notebook for analyzing alpha signal quality and predictiveness.

**Analysis Sections:**
1. Information Coefficient (IC) time series
2. Signal decay analysis (correlation at different horizons)
3. Quintile performance analysis
4. Cross-sectional spread of alpha scores
5. Hit rate by signal strength
6. Turnover and stability analysis

**Files:**
- Create: `notebooks/alpha_diagnostics.ipynb`
- Create: `src/quantetf/evaluation/alpha_diagnostics.py` (helper functions)
- Create: `tests/test_alpha_diagnostics.py`

**Acceptance Criteria:**
- All 6 analysis sections implemented
- Helper functions tested (10+ tests)
- Works with momentum alpha model
- Clear interpretation guidelines

---

### ANALYSIS-003: Strategy Comparison Script
**Status:** ready
**Priority:** high
**Estimated:** 3-4 hours
**Dependencies:** [ANALYSIS-001, ANALYSIS-002]

**Description:**
Create script to run multiple strategy variants and generate comparative analysis.

**Features:**
- Run multiple configs in parallel or sequence
- Generate comparison table (all metrics side-by-side)
- Equity curve overlay chart
- Risk-return scatter plot
- Correlation matrix of strategy returns
- Statistical significance tests (Sharpe ratio t-test)

**Files:**
- Create: `scripts/compare_strategies.py`
- Create: `src/quantetf/evaluation/comparison.py`
- Create: `tests/test_strategy_comparison.py`

**Acceptance Criteria:**
- CLI accepts multiple config files
- Outputs HTML report with charts
- Saves comparison results to artifacts/
- 15+ tests for comparison logic
- Documentation in scripts/README.md

**Example Usage:**
```bash
python scripts/compare_strategies.py \
  --configs configs/strategies/*.yaml \
  --snapshot data/snapshots/snapshot_5yr_20etfs \
  --output artifacts/comparisons/
```

---

### ANALYSIS-004: Parameter Sensitivity Analysis
**Status:** ready
**Priority:** medium
**Estimated:** 2-3 hours
**Dependencies:** [ANALYSIS-003]

**Description:**
Create notebook for systematic parameter sweep and sensitivity testing.

**Parameters to Test:**
- Momentum lookback: [63, 126, 252, 504] days
- Top-N selection: [3, 5, 7, 10] ETFs
- Rebalance frequency: ['weekly', 'monthly', 'quarterly']
- Transaction costs: [5, 10, 20, 50] bps

**Visualizations:**
- Heatmap: Sharpe ratio vs (lookback, top_n)
- Surface plot: Returns vs (rebalance_freq, cost_bps)
- Robustness check: performance stability

**Files:**
- Create: `notebooks/parameter_sweep.ipynb`
- Create: `scripts/grid_search.py` (automation)

**Acceptance Criteria:**
- Automated parameter grid search
- Results saved to artifacts/parameter_sweeps/
- Clear heatmap visualizations
- Identifies robust parameter ranges

---

### ANALYSIS-005: Benchmark Comparison Framework
**Status:** ready
**Priority:** high
**Estimated:** 2-3 hours
**Dependencies:** [ANALYSIS-001, ANALYSIS-003]

**Description:**
Create framework for comparing strategies against standard benchmarks.

**Benchmarks to Implement:**
1. SPY buy-and-hold
2. 60/40 portfolio (SPY/AGG)
3. Equal-weight universe
4. Random selection (Monte Carlo)
5. Oracle (perfect foresight upper bound)

**Metrics:**
- Excess return (strategy - benchmark)
- Tracking error
- Information ratio
- Beta and alpha (regression)
- Drawdown comparison

**Files:**
- Create: `scripts/benchmark_comparison.py`
- Create: `src/quantetf/evaluation/benchmarks.py`
- Create: `tests/test_benchmarks.py`

**Acceptance Criteria:**
- All 5 benchmarks implemented
- Regression-based attribution
- HTML report generation
- 12+ tests

---

### ANALYSIS-006: Walk-Forward Validation Framework
**Status:** ready
**Priority:** critical
**Estimated:** 4-5 hours
**Dependencies:** [ANALYSIS-003]

**Description:**
Implement rolling window validation to test strategy robustness and prevent overfitting.

**Methodology:**
1. Define rolling windows (e.g., 2-year train, 1-year test)
2. For each window: run backtest on train period
3. Test on out-of-sample period
4. Roll window forward
5. Aggregate results

**Analysis:**
- Out-of-sample Sharpe distribution
- In-sample vs out-of-sample degradation
- Parameter stability over time
- Regime-specific performance

**Files:**
- Create: `scripts/walk_forward_test.py`
- Create: `src/quantetf/evaluation/walk_forward.py`
- Create: `tests/test_walk_forward.py`

**Acceptance Criteria:**
- Configurable window sizes
- Prevents data leakage
- Comprehensive report generation
- 15+ tests
- Documentation on interpreting results

---

### ANALYSIS-007: Transaction Cost Analysis
**Status:** ready
**Priority:** medium
**Estimated:** 2-3 hours
**Dependencies:** [ANALYSIS-001]

**Description:**
Enhance cost modeling with realistic cost structures and sensitivity analysis.

**New Cost Models:**
- SlippageCostModel (volume-based)
- SpreadCostModel (bid-ask spreads)
- ImpactCostModel (market impact for large trades)

**Analysis:**
- Cost sensitivity notebook
- Rebalance frequency vs cost drag
- High-turnover vs low-turnover comparison
- Identify expensive trades (illiquid ETFs)

**Files:**
- Update: `src/quantetf/portfolio/costs.py`
- Create: `notebooks/cost_sensitivity.ipynb`
- Update: `tests/test_transaction_costs.py`

**Acceptance Criteria:**
- 3 new cost models implemented
- Each model has 5+ tests
- Sensitivity notebook functional
- Cost comparison framework

---

### VIZ-003: Stress Test Notebook
**Status:** ready
**Priority:** medium
**Estimated:** 2-3 hours
**Dependencies:** [VIZ-001, ANALYSIS-001]

**Description:**
Analyze strategy performance during specific market regimes and crisis periods.

**Periods to Test:**
- 2020 COVID crash (Feb-Mar 2020)
- 2022 Bond/Tech selloff
- High volatility periods (VIX > 30)
- Low volatility periods (VIX < 15)
- Bull markets vs bear markets

**Metrics:**
- Max drawdown in crisis
- Recovery time
- Volatility spike behavior
- Correlation breakdown

**Files:**
- Create: `notebooks/stress_test.ipynb`

**Acceptance Criteria:**
- Identifies crisis periods automatically
- Regime classification logic
- Clear before/during/after analysis
- Comparative visualizations

---

### VIZ-004: Auto-Report Generation
**Status:** ready
**Priority:** medium
**Estimated:** 3-4 hours
**Dependencies:** [VIZ-001, ANALYSIS-001, ANALYSIS-002]

**Description:**
Create automated HTML report generator from backtest results.

**Report Sections:**
1. Executive summary (key metrics table)
2. Performance charts (equity, drawdown, returns)
3. Risk analytics (correlations, beta, VaR)
4. Holdings analysis (turnover, concentration)
5. Trade log with attribution
6. Benchmark comparison

**Files:**
- Create: `scripts/generate_report.py`
- Create: `src/quantetf/evaluation/report_builder.py`
- Create: `templates/report_template.html` (Jinja2)

**Acceptance Criteria:**
- Professional HTML output
- Embeds charts (base64 or inline SVG)
- Self-contained single file
- CLI interface
- Example usage in scripts/README.md

---

### INFRA-002: Data Quality Monitoring
**Status:** ready
**Priority:** medium
**Estimated:** 2-3 hours
**Dependencies:** []

**Description:**
Create script to audit data quality and detect anomalies.

**Checks:**
- Missing data summary (% NaN by ticker)
- Price spikes (>10% single-day moves)
- Stale data detection (gaps > 5 days)
- Volume anomalies
- Correlation matrix (detect duplicates)
- Delisting detection

**Files:**
- Create: `scripts/data_health_check.py`
- Create: `src/quantetf/data/quality.py`
- Create: `tests/test_data_quality.py`

**Acceptance Criteria:**
- Automated quality scoring
- Generates quality report
- Flags suspicious data
- 12+ tests

---

## Backlog (Future Phases)

### Phase 4: Strategy Development
- IMPL-006: Mean reversion alpha model
- IMPL-007: Multi-factor alpha combiner
- IMPL-008: Mean-variance portfolio optimizer
- IMPL-009: Risk parity constructor
- IMPL-010: Covariance estimation

### Phase 5: Production Pipeline
- IMPL-011: Production recommendation generator
- IMPL-012: Run manifest creation
- IMPL-013: Alerting system
- IMPL-014: Data freshness checks

### Phase 6: Documentation & Polish
- DOC-001: Tutorial notebook
- DOC-002: API documentation
- TEST-002: Golden test suite

---

## Task Creation Guidelines

When creating new tasks:

1. **Keep tasks focused** - 1-3 hours of work
2. **Make independent** - Minimize dependencies
3. **Provide context** - Create handoff file
4. **Clear acceptance criteria** - How to verify done
5. **Include examples** - Show expected usage

## Task Pickup Process

1. Find a `ready` task that interests you
2. Update status to `in_progress`
3. Add your agent ID to `Assigned`
4. Read the handoff file completely
5. Read CLAUDE_CONTEXT.md for standards
6. Implement and test
7. Update status to `completed`
8. Create completion note

## Need Help?

- **Blocked?** Update task status and note blocker
- **Unclear requirements?** Check handoff file and CLAUDE_CONTEXT.md
- **Architecture questions?** See PROJECT_BRIEF.md
- **Can't find pattern?** Look at similar completed tasks
