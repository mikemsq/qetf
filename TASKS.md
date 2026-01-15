# Task Queue - QuantETF

**Last Updated:** January 15, 2026
**Active Phase:** Regime-Based Strategy Research

## 🎯 PRIMARY GOAL

**Find a strategy that beats SPY by adapting to market regimes.**

| Requirement | Value |
|-------------|-------|
| Primary Universe | **Tier 4 (200 ETFs)** |
| Data Period | 10 years (2016-2026) |
| Evaluation Periods | **1yr AND 3yr** (changed from 3yr/5yr/10yr) |
| Win Criteria | Active Return > 0, IR > 0 in BOTH periods |

---

## Task Status Legend

- `ready` - Available for pickup by any agent
- `in_progress` - Currently being worked on
- `blocked` - Waiting on dependencies
- `completed` - Implementation done, needs review
- `merged` - Reviewed and merged to main

---

## 🚀 IMMEDIATE PRIORITY: Regime-Based Strategy Research

**Research Question:** Can we beat SPY over any 1-year period by adapting strategy to market regime?

**Status:** Research complete. Implementation tasks ready.
**Findings Document:** `handoffs/RESEARCH-001-REGIME-HYPOTHESIS.md`
**Quick Reference:** `handoffs/RESEARCH_AGENDA_QUICKREF.md`

### Key Findings
- Perfect foresight: 10/10 years beatable (avg +55.8%/year spread)
- Simple momentum: 3/9 years win rate (33%)
- Regime patterns exist: Bull (7yr), Bear (1yr), High Vol (1yr)
- No single strategy dominates all regimes

---

### IMPL-006: New Alpha Models for Regime Research
**Status:** completed
**Priority:** CRITICAL
**Estimated:** 3-4 hours
**Completed:** 2026-01-15
**Dependencies:** []

**Description:**
Implement three new alpha models identified in RESEARCH-001:
1. **TrendFilteredMomentum** - Momentum with MA200 trend filter
2. **DualMomentum** - Absolute + relative momentum (Gary Antonacci style)
3. **ValueMomentum** - Blend momentum and mean-reversion signals

**Files Created:**
- `src/quantetf/alpha/trend_filtered_momentum.py` (240 lines)
- `src/quantetf/alpha/dual_momentum.py` (185 lines)
- `src/quantetf/alpha/value_momentum.py` (220 lines)
- `tests/test_trend_filtered_momentum.py` (9 tests)
- `tests/test_dual_momentum.py` (9 tests)
- `tests/test_value_momentum.py` (10 tests)

**Full Specification:** `handoffs/IMPL-006-NEW-ALPHA-MODELS.md`

**Acceptance Criteria:**
- [x] All 3 alpha models implemented
- [x] 28 unit tests (exceeds 15+ requirement)
- [x] Exports updated in `__init__.py`

**Notes:**
- All models follow existing AlphaModel interface (compatible with backtest engine)
- TrendFilteredMomentum gracefully handles missing trend ticker
- DualMomentum implements absolute momentum threshold filtering
- ValueMomentum z-scores signals before blending for proper weighting

---

### IMPL-007: FRED Macro Data Ingestion
**Status:** completed
**Priority:** HIGH
**Estimated:** 2-3 hours
**Completed:** 2026-01-15
**Dependencies:** []

**Description:**
Create data ingestion for FRED economic indicators to enable macro regime detection.

**Key Indicators:**
- VIX (VIXCLS) - Volatility regime
- 10Y Treasury (DGS10) - Risk-free rate
- 2Y-10Y Spread (T10Y2Y) - Recession signal
- High Yield Spread (BAMLH0A0HYM2) - Credit stress

**Files Created:**
- `scripts/ingest_fred_data.py` - CLI for downloading FRED data (288 lines)
- `src/quantetf/data/macro_loader.py` - Loader and regime detector (160 lines)
- `tests/data/test_macro_loader.py` - Unit tests (18 tests)
- `fredapi>=0.5.1` added to pyproject.toml

**Full Specification:** `handoffs/IMPL-007-DATA-INGESTION.md`

**Acceptance Criteria:**
- [x] FRED ingestion script working
- [x] MacroDataLoader class implemented
- [x] RegimeDetector class implemented
- [x] Manifest and combined dataset creation
- [x] 18 unit tests passing

**Notes:**
- Ingestion script supports 12 default FRED indicators
- MacroDataLoader provides helpers for VIX, yield curve spread, regime detection
- RegimeDetector classifies: RISK_ON, ELEVATED_VOL, HIGH_VOL, RECESSION_WARNING
- Exports registered in `src/quantetf/data/__init__.py`
- Requires FRED API key (free at https://fred.stlouisfed.org/docs/api/api_key.html)

---

### EXP-001: Monthly vs Annual Rebalancing
**Status:** blocked
**Priority:** HIGH
**Estimated:** 2-3 hours
**Dependencies:** [IMPL-006]

**Description:**
Test hypothesis that monthly rebalancing captures regime shifts faster than annual.

**Analysis:**
- Run 12M momentum with monthly vs annual rebalance
- Compare: win rate, max drawdown, Sharpe ratio
- Document results

---

### EXP-002: Trend Filter Backtest
**Status:** blocked
**Priority:** HIGH
**Estimated:** 2-3 hours
**Dependencies:** [IMPL-006]

**Description:**
Test TrendFilteredMomentum strategy over 10-year period.

**Hypothesis:** SPY > MA200 filter reduces catastrophic losses (2018, 2022).

---

### EXP-003: Ensemble vs Switching
**Status:** blocked
**Priority:** MEDIUM
**Estimated:** 2-3 hours
**Dependencies:** [IMPL-006, IMPL-007]

**Description:**
Test whether blending strategies beats regime-switching.

---

### EXP-004: Walk-Forward Validation
**Status:** blocked
**Priority:** HIGH
**Estimated:** 3-4 hours
**Dependencies:** [EXP-001, EXP-002]

**Description:**
Train regime rules on 2016-2020, test on 2021-2025.

---

## 📊 SECONDARY PRIORITY: Strategy Optimizer Search

### DATA-001: Ingest Tier 4 10-Year Data
**Status:** completed
**Priority:** HIGH
**Completed:** 2026-01-15
**Dependencies:** []

**Description:**
Ingest 10 years of historical data for the Tier 4 universe (200 ETFs) and create a snapshot for strategy optimization.

**Steps:**
1. Run ingest script for Tier 4 (200 ETFs, 2016-01-15 to 2026-01-15)
2. Create snapshot from ingested data
3. Validate data quality

**Command:**
```bash
# Ingest data
python scripts/ingest_etf_data.py \
    --universe tier4_broad_200 \
    --start-date 2016-01-15 \
    --end-date 2026-01-15

# Create snapshot
python scripts/create_snapshot.py --universe tier4_broad_200
```

**Results:**
- Snapshot: `data/snapshots/snapshot_20260115_170559/`
- 200 ETFs with 10 years of data (2016-01-15 to 2026-01-14)
- 2514 trading days
- 103/200 tickers passed strict OHLC validation (97 had minor data quality flags)
- 1.49% NaN in Close prices (acceptable for backtesting)
- SPY benchmark present and validated

**Acceptance Criteria:**
- [x] 200 ETFs with 10 years of data
- [x] Snapshot created and validated
- [x] Ready for optimizer

---

### SEARCH-001: Run Strategy Optimizer on Tier 4
**Status:** ready
**Priority:** CRITICAL
**Estimated:** 1-2 hours (compute time)
**Dependencies:** [DATA-001] (completed)

**Description:**
Run the strategy optimizer on Tier 4 data with 1yr/3yr evaluation periods to find winning strategies.

**Command:**
```bash
python scripts/find_best_strategy.py \
    --snapshot data/snapshots/<tier4_snapshot> \
    --periods 1,3 \
    --parallel 4
```

**Expected Output:**
- `artifacts/optimization/<timestamp>/winners.csv` - Strategies that beat SPY in both periods
- `artifacts/optimization/<timestamp>/best_strategy.yaml` - Best configuration

**Acceptance Criteria:**
- [ ] At least one strategy beats SPY in both 1yr AND 3yr periods
- [ ] Winners documented with full metrics
- [ ] Best strategy config saved for production use

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

### REFACTOR-001: Active Returns Focus Refactor
**Status:** completed
**Priority:** CRITICAL
**Estimated:** 2-3 hours
**Completed:** 2026-01-15
**Dependencies:** []
**Assigned:** Session-REFACTOR-001

**Description:**
Refactor existing performance analysis to emphasize active returns vs SPY benchmark. Current implementation shows portfolio performance in isolation, but the project goal is to beat SPY.

**User Requirement (Jan 12, 2026):**
> "I always want to see the portfolio performance, benchmark performance and active performance. The pure portfolio performance is less interesting than comparison to buy and hold spy strategy. The goal of this project is to create a strategy that can beat spy."

**Changes Implemented:**
1. ✅ Added `calculate_active_metrics()` helper to metrics.py (lines 408-536)
2. ✅ Updated backtest_analysis.ipynb to show strategy vs SPY overlaid
3. ✅ All visualizations show relative performance with outperformance shading
4. ✅ Notebook leads with "🎯 ACTIVE PERFORMANCE SUMMARY" section
5. ✅ SPY comparison is default in compare_strategies.py via `add_spy_benchmark()`

**Files Updated:**
- `src/quantetf/evaluation/metrics.py` - Added calculate_active_metrics() (128 lines)
- `notebooks/backtest_analysis.ipynb` - SPY overlay in all charts, warmup alignment
- `scripts/compare_strategies.py` - Auto-adds SPY benchmark (add_spy_benchmark function)
- `tests/test_advanced_metrics.py` - 16 tests for calculate_active_metrics (TestCalculateActiveMetrics class)

**Handoff:** `handoffs/PERFORMANCE_ANALYSIS_REFACTOR.md`

**Acceptance Criteria (All Met):**
- ✅ calculate_active_metrics() function implemented and tested (16 tests)
- ✅ Notebook shows strategy vs SPY overlaid in all charts (equity, drawdown, rolling Sharpe)
- ✅ Notebook leads with "Beat SPY by X%" summary (🎯 ACTIVE PERFORMANCE SUMMARY)
- ✅ All metrics tables show strategy, benchmark, and active side-by-side
- ✅ Documentation updated (CLAUDE_CONTEXT.md ✅, PROJECT_BRIEF.md ✅)

**Notes:**
- Implements warmup period alignment for fair comparison (strategy starts at first active trade)
- Shows dollar value comparison AND normalized return comparison
- Includes rolling active returns analysis (1-year rolling)
- compare_strategies.py auto-adds SPY if backtests are loaded

---

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

### ANALYSIS-003: Strategy Comparison Script
**Status:** completed
**Priority:** high
**Estimated:** 3-4 hours
**Completed:** 2026-01-11
**Dependencies:** [ANALYSIS-001, ANALYSIS-002] (both completed)
**Assigned:** ANALYSIS-003 Session

**Description:**
Create script to run multiple strategy variants and generate comparative analysis.

**Features Implemented:**
- Load and compare multiple backtest results
- Generate comparison table (all metrics side-by-side)
- Equity curve overlay chart
- Risk-return scatter plot
- Correlation matrix of strategy returns
- Statistical significance tests (Jobson-Korkie Sharpe ratio t-test)
- HTML report generation with styling
- Console summary output

**Files:**
- Created: `scripts/compare_strategies.py` (370+ lines, full CLI)
- Created: `src/quantetf/evaluation/comparison.py` (560+ lines, 11 functions)
- Created: `tests/test_strategy_comparison.py` (28 tests, all passing)
- Updated: `scripts/README.md` (comprehensive documentation)
- Updated: `pyproject.toml` (added scipy and seaborn dependencies)

**Results:**
- 28 comprehensive tests (exceeds target of 15+)
- All tests passing (100%)
- Full comparison functionality working
- Professional visualizations (equity overlay, risk-return scatter)
- Statistical significance testing with Jobson-Korkie method
- HTML table generation with CSS styling
- Total test count: 211 → 239 (+28)

**Notes:**
- Implements 11 comparison functions: load, metrics, correlation, t-test, charts, reports
- Jobson-Korkie test for statistically rigorous Sharpe ratio comparison
- Handles edge cases: single strategy, empty results, missing data
- Console output includes formatted tables and significance tests
- Ready for use in parameter sensitivity analysis (ANALYSIS-004)

---

### VIZ-001: Backtest Analysis Notebook
**Status:** completed
**Priority:** high
**Estimated:** 3-4 hours
**Completed:** 2026-01-11
**Dependencies:** [ANALYSIS-001] (completed)
**Assigned:** VIZ-001 Session

**Description:**
Create comprehensive Jupyter notebook for visualizing backtest results.

**Required Visualizations:**
1. Equity curve with dual-axis drawdown overlay ✅
2. Monthly/yearly returns heatmap ✅
3. Rolling Sharpe ratio (252-day window) ✅
4. Drawdown waterfall chart ✅
5. Returns distribution histogram ✅
6. Underwater plot (time below high-water mark) ✅
7. Holdings evolution over time ✅
8. Turnover analysis ✅

**Files:**
- Created: `notebooks/backtest_analysis.ipynb` (808KB with execution outputs)
- Updated: `notebooks/README.md` (comprehensive documentation)

**Results:**
- All 8 visualizations implemented and tested (exceeds requirements)
- Automatically loads latest valid backtest from artifacts/backtests/
- Clear markdown explanations for each section
- Runs end-to-end without errors (tested with nbconvert)
- Professional matplotlib/seaborn visualizations
- Robust error handling (KDE optional, scipy optional)
- Includes comprehensive statistics for each visualization
- Summary section ties all analyses together

**Acceptance Criteria:**
- ✅ Loads latest backtest from artifacts/backtests/
- ✅ All 8 visualizations implemented
- ✅ Clear markdown explanations
- ✅ Runs end-to-end without errors
- ✅ Generates professional-looking charts

**Notes:**
- Uses matplotlib/seaborn for consistency (as requested)
- Handles edge cases gracefully (empty data, missing scipy, etc.)
- Includes detailed statistics for each visualization
- Ready for use in strategy analysis workflow

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
**Status:** completed
**Priority:** high
**Estimated:** 2-3 hours
**Completed:** 2026-01-11
**Dependencies:** [ANALYSIS-001, ANALYSIS-003] (both completed)
**Assigned:** ANALYSIS-005 Session

**Description:**
Create framework for comparing strategies against standard benchmarks.

**Benchmarks Implemented:**
1. SPY buy-and-hold ✅
2. 60/40 portfolio (SPY/AGG) ✅
3. Equal-weight universe ✅
4. Random selection (Monte Carlo) ✅
5. Oracle (perfect foresight upper bound) ✅

**Metrics:**
- Excess return (strategy - benchmark) ✅
- Tracking error ✅
- Information ratio ✅
- Beta and alpha (regression) ✅
- Drawdown comparison ✅

**Files:**
- Created: `scripts/benchmark_comparison.py` (370+ lines, full CLI)
- Created: `src/quantetf/evaluation/benchmarks.py` (700+ lines, 5 benchmarks + utilities)
- Created: `tests/test_benchmarks.py` (20 tests, all passing)

**Results:**
- All 5 benchmarks implemented with proper point-in-time data access
- 20 comprehensive tests (exceeds target of 12+)
- All tests passing (100%)
- Regression-based attribution (beta, alpha, tracking error, information ratio)
- HTML report generation with visualizations
- Console output with formatted tables
- Total test count: 239 → 259 (+20)

**Acceptance Criteria:**
- ✅ All 5 benchmarks implemented
- ✅ Regression-based attribution
- ✅ HTML report generation
- ✅ 12+ tests (20 delivered)

**Notes:**
- SPY buy-and-hold: 100% passive market exposure
- 60/40 portfolio: Quarterly rebalanced balanced portfolio
- Equal-weight universe: Monthly rebalanced equal allocation
- Random selection: Monte Carlo average with configurable trials
- Oracle: Perfect foresight upper bound for strategy assessment
- All benchmarks handle missing data gracefully
- CLI supports custom date ranges and benchmark selection
- Ready for use in parameter sensitivity analysis

---

### ANALYSIS-006: Walk-Forward Validation Framework
**Status:** completed
**Priority:** critical
**Estimated:** 4-5 hours
**Completed:** 2026-01-11
**Dependencies:** [ANALYSIS-003] (completed)
**Assigned:** ANALYSIS-006 Session

**Description:**
Implement rolling window validation to test strategy robustness and prevent overfitting.

**Methodology Implemented:**
1. Define rolling windows (e.g., 2-year train, 1-year test) ✅
2. For each window: run backtest on train period ✅
3. Test on out-of-sample period ✅
4. Roll window forward ✅
5. Aggregate results ✅

**Analysis Features:**
- Out-of-sample Sharpe distribution ✅
- In-sample vs out-of-sample degradation ✅
- Parameter stability over time ✅
- Regime-specific performance tracking ✅

**Files:**
- Created: `scripts/walk_forward_test.py` (400+ lines, full CLI)
- Created: `src/quantetf/evaluation/walk_forward.py` (700+ lines, 11 functions)
- Created: `tests/test_walk_forward.py` (20 tests, all passing)
- Updated: `scripts/README.md` (comprehensive documentation)

**Results:**
- All 20 comprehensive tests passing (exceeds target of 15+)
- Configurable train/test window sizes
- Prevents data leakage (point-in-time enforcement)
- JSON summary output + CSV tables + visualizations
- Validated with real 5-year backtest data
- Total test count: 255 → 275 (+20)

**Acceptance Criteria:**
- ✅ Configurable window sizes (train_years, test_years, step_months)
- ✅ Prevents data leakage (uses SnapshotDataStore with T-1 access)
- ✅ Comprehensive report generation (summary.json, window_results.csv, plots)
- ✅ 20 tests (exceeds 15+ requirement)
- ✅ Documentation on interpreting results (README.md + docstrings)

**Notes:**
- Walk-forward framework provides critical robustness testing
- Generates 4 visualization plots: Sharpe evolution, returns evolution, OOS distribution, degradation
- Summary includes degradation metrics, stability metrics, and window-by-window analysis
- CLI supports custom date ranges, strategy parameters, and output locations
- Example run on real data shows OOS Sharpe of 0.89 (4 windows)

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

## Current Sprint: Strategy Optimizer System ✅ COMPLETE

This sprint implements an automated system to find ETF strategies that beat SPY across multiple time periods (3yr, 5yr, 10yr). The system systematically searches across alpha models, parameters, schedules, and portfolio construction options.

**Status:** All 5 tasks completed (OPT-001 through OPT-005)
**Handout Reference**: See `/workspaces/qetf/docs/handouts/HANDOUT_strategy_optimizer.md` for full architecture.

---

### OPT-001: Parameter Grid Generator
**Status:** completed
**Priority:** HIGH
**Estimated:** 2-3 hours
**Completed:** 2026-01-14
**Dependencies:** []
**Assigned:** Session-OPT-001

**Description:**
Create the parameter grid generator module that defines schedule-specific parameter search spaces for all 4 alpha models (momentum, momentum_acceleration, vol_adjusted_momentum, residual_momentum).

**Key Design Insight:**
- Weekly rebalancing uses shorter lookback periods (faster signals match frequent trading)
- Monthly rebalancing uses longer lookback periods (stable signals match infrequent trading)

**Files Created:**
- `src/quantetf/optimization/grid.py` (450+ lines)
- `tests/optimization/test_grid.py` (47 tests)

**Implementation Details:**
- Define `PARAMETER_GRIDS_WEEKLY` and `PARAMETER_GRIDS_MONTHLY` dictionaries
- Create `StrategyConfig` dataclass with `to_dict()` and `generate_name()` methods
- Implement `is_valid_config()` for validation (e.g., short_lookback < long_lookback)
- Implement `generate_configs()` to generate all valid combinations
- Implement `count_configs()` for reporting

**Results:**
- 324 total configurations generated
- 47 comprehensive tests, all passing
- Commit: `b9e4932`

**Handoff:** `docs/handouts/HANDOUT_grid_generator.md`

**Acceptance Criteria:**
- [x] All 4 alpha models have schedule-specific parameter grids
- [x] `is_valid_config()` rejects invalid combinations (short >= long for momentum_acceleration)
- [x] `generate_configs()` returns list of StrategyConfig objects
- [x] `count_configs()` returns breakdown by schedule and alpha type
- [x] 10+ tests covering edge cases and validation (47 delivered)
- [x] Type hints and docstrings complete

---

### OPT-002: Multi-Period Evaluator
**Status:** completed
**Priority:** HIGH
**Estimated:** 3-4 hours
**Completed:** 2026-01-15
**Dependencies:** [OPT-001] (completed)
**Assigned:** Session-OPT-002

**Description:**
Create the multi-period evaluator that runs a single strategy configuration across configurable time windows and determines if it beats SPY.

**UPDATED REQUIREMENT (Jan 15, 2026):**
- **Evaluation periods changed to: 1yr AND 3yr** (from 3yr/5yr/10yr)
- Strategy wins if it beats SPY in BOTH 1-year and 3-year periods

**Key Features:**
- Evaluate strategy across configurable time periods
- Calculate active metrics vs SPY benchmark
- Determine "beats SPY" status (positive active return AND positive IR in ALL periods)
- Calculate composite score for ranking

**Files Created:**
- `src/quantetf/optimization/evaluator.py` (400+ lines)
- `tests/optimization/test_evaluator.py` (22 tests)

**Implementation Details:**
- Created `PeriodMetrics` dataclass (strategy return, SPY return, active return, IR, etc.)
- Created `MultiPeriodResult` dataclass with `beats_spy_all_periods` flag and `composite_score`
- Implemented `MultiPeriodEvaluator` class with `evaluate()` method
- Composite score = avg(IR) - consistency_penalty + winner_bonus
- **Periods are configurable** - pass `--periods 1,3` to CLI for new requirement

**Dependencies (existing modules):**
- `src/quantetf/backtest/simple_engine.py` - `SimpleBacktestEngine`
- `src/quantetf/evaluation/metrics.py` - `calculate_active_metrics`
- `src/quantetf/evaluation/benchmarks.py` - `get_spy_returns`
- `src/quantetf/data/snapshot_store.py` - `SnapshotDataStore`

**Handoff:** `docs/handouts/HANDOUT_multi_period_evaluator.md`

**Results:**
- 22 comprehensive tests (exceeds target of 12+)
- All tests passing (100%)
- Integrates with existing backtest engine and alpha factory
- Supports configurable evaluation periods
- Total test count: 356 → 378 (+22 optimization tests)

**Acceptance Criteria:**
- [x] Evaluates strategy across configurable periods (default: 1yr, 3yr)
- [x] Calculates all metrics: strategy return, SPY return, active return, IR, Sharpe, max DD
- [x] `beats_spy_all_periods` correctly identifies winning strategies
- [x] Composite score rewards consistency and penalizes volatility
- [x] Graceful error handling for failed evaluations
- [x] 12+ tests covering normal operation and edge cases (22 delivered)

---

### OPT-003: Strategy Optimizer
**Status:** completed
**Priority:** HIGH
**Estimated:** 3-4 hours
**Completed:** 2026-01-15
**Dependencies:** [OPT-001, OPT-002] (both completed)
**Assigned:** Session-OPT-003

**Description:**
Create the main optimizer orchestrator that generates all configurations, runs evaluations, ranks results, and produces reports.

**Key Features:**
- Sequential or parallel execution (configurable)
- Progress tracking with tqdm
- Graceful error handling (log and skip failed configs)
- Comprehensive output files

**Files Created:**
- `src/quantetf/optimization/optimizer.py` (450+ lines)
- `tests/optimization/test_optimizer.py` (26 tests)

**Implementation Details:**
- Created `OptimizationResult` dataclass with all_results, winners, best_config
- Implemented `StrategyOptimizer` class with `run()` method
- Support `max_workers` parameter for parallel execution
- Generates output files:
  - `all_results.csv` - every config with metrics
  - `winners.csv` - only configs that beat SPY
  - `best_strategy.yaml` - ready-to-use config file
  - `optimization_report.md` - human-readable summary

**Output Directory Structure:**
```
artifacts/optimization/TIMESTAMP/
├── all_results.csv
├── winners.csv
├── best_strategy.yaml
└── optimization_report.md
```

**Handoff:** `docs/handouts/HANDOUT_optimizer.md`

**Results:**
- 26 comprehensive tests, all passing
- Supports filtering by schedule_names and alpha_types
- Progress callback for custom progress tracking
- Integration tests with real snapshot data

**Acceptance Criteria:**
- [x] Generates all configs via grid.py
- [x] Runs evaluations via evaluator.py
- [x] Sorts results by composite score
- [x] Identifies and exports winning strategies
- [x] Creates all 4 output files
- [x] Progress bar shows evaluation progress
- [x] Handles failed configs gracefully
- [x] 10+ tests (26 delivered)

---

### OPT-004: CLI Script
**Status:** completed
**Priority:** HIGH
**Estimated:** 1-2 hours
**Completed:** 2026-01-15
**Dependencies:** [OPT-003] (completed)
**Assigned:** Session-OPT-004

**Description:**
Create the command-line interface for running the strategy optimizer.

**Files Created:**
- `scripts/find_best_strategy.py` (200+ lines)

**CLI Arguments:**
- `--snapshot` (required): Path to data snapshot directory
- `--output` (default: artifacts/optimization): Output directory
- `--periods` (default: 3,5,10): Comma-separated evaluation periods in years
- `--max-configs` (optional): Limit configs for debugging
- `--parallel` (default: 1): Number of parallel workers
- `--cost-bps` (default: 10.0): Transaction cost in basis points
- `--schedules` (optional): Filter by schedule type (weekly, monthly)
- `--alpha-types` (optional): Filter by alpha type
- `--verbose/-v`: Enable debug logging
- `--dry-run`: Just count configs without running

**Usage Examples:**
```bash
# Basic run
python scripts/find_best_strategy.py \
    --snapshot data/snapshots/snapshot_20260113_232157

# Quick test with 20 configs
python scripts/find_best_strategy.py \
    --snapshot data/snapshots/snapshot_20260113_232157 \
    --max-configs 20 --verbose

# Parallel execution
python scripts/find_best_strategy.py \
    --snapshot data/snapshots/snapshot_20260113_232157 \
    --parallel 4

# Dry run to count configs
python scripts/find_best_strategy.py \
    --snapshot data/snapshots/snapshot_20260113_232157 \
    --dry-run
```

**Handoff:** `docs/handouts/HANDOUT_cli_script.md`

**Acceptance Criteria:**
- [x] All CLI arguments work correctly
- [x] `--dry-run` counts configs without running
- [x] `--verbose` enables debug logging
- [x] Prints summary with winners count and best strategy
- [x] Reports output directory location
- [x] Executable (chmod +x)

**Notes:**
- Added additional filtering options (`--schedules`, `--alpha-types`, `--cost-bps`)
- Comprehensive help text with examples
- Validates snapshot path and periods format
- Updated scripts/README.md with full documentation

---

### OPT-005: Update __init__.py Exports
**Status:** completed
**Priority:** MEDIUM
**Estimated:** 30 minutes
**Completed:** 2026-01-15
**Dependencies:** [OPT-001, OPT-002, OPT-003] (all completed)
**Assigned:** Session-OPT-003

**Description:**
Update `src/quantetf/optimization/__init__.py` to export all new classes and functions.

**Files Updated:**
- `src/quantetf/optimization/__init__.py`

**Exports Added:**
- From evaluator.py: `PeriodMetrics`, `MultiPeriodResult`, `MultiPeriodEvaluator`
- From optimizer.py: `OptimizationResult`, `StrategyOptimizer`

**Acceptance Criteria:**
- [x] All public classes importable from `quantetf.optimization`
- [x] `__all__` list updated

---

## Implementation Order

**Recommended sequence for serial implementation:**

1. **OPT-001: Grid Generator** (no dependencies, pure Python)
2. **OPT-002: Multi-Period Evaluator** (depends on OPT-001)
3. **OPT-003: Strategy Optimizer** (depends on OPT-001, OPT-002)
4. **OPT-004: CLI Script** (depends on OPT-003)
5. **OPT-005: Update Exports** (depends on all above)

**For parallel implementation:**
- Agent A: OPT-001 (Grid Generator) → OPT-003 (Optimizer)
- Agent B: OPT-002 (Evaluator) - can start after OPT-001 type definitions available
- Agent C: OPT-004 (CLI) - can start after OPT-003 interface defined

**Total estimated time:** 10-14 hours serial, 5-7 hours parallel

---

## Risk Mitigation Notes

1. **Overfitting Prevention**: Using 3 different time periods as pseudo-out-of-sample validation
2. **Data Snooping**: Report ALL results, not just winners
3. **Computational Cost**: ~354 configs is manageable (~15-25 min)
4. **Error Handling**: Log and skip failed configs, don't abort entire run

---

## Backlog (Future Phases)

### Phase 4: Strategy Development
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
