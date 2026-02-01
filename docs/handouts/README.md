# Coding Agent Handouts - Phase 1 Strategies

This directory contains detailed, standalone handout documents for implementing Phase 1 Enhanced Momentum strategies.

## Purpose

These handouts are designed to be **complete, self-contained specifications** that coding agents can use to implement strategies with ZERO additional context needed. Each handout includes:

- Executive summary and research rationale
- Complete mathematical definitions
- Implementation specifications with pseudocode
- Full code templates with TODOs
- Comprehensive testing requirements (8+ test cases)
- Integration guides with config examples
- Edge case handling strategies
- Acceptance checklists (25+ items)
- Reference files with specific line numbers

## Available Handouts

### ✅ HANDOUT_momentum_acceleration.md
**Status**: Complete (615 lines)
**Strategy**: Trend strength (3M returns - 12M returns)
**Complexity**: LOW (Simplest!)
**Priority**: HIGH (best to start here!)

Key features:
- Captures momentum inflection points
- Just two return calculations and a subtraction
- Expected: Earlier entry/exit signals, higher CAGR
- **Recommended first implementation**

### ✅ HANDOUT_vol_adjusted_momentum.md
**Status**: Complete (685 lines)
**Strategy**: Sharpe-style momentum (returns / volatility)
**Complexity**: LOW
**Priority**: HIGH (implement second)

Key features:
- Ranks by risk-adjusted returns
- Simple division operation (no regression)
- Expected: Better drawdown control, smoother equity curve
- **Recommended second implementation**

### ✅ HANDOUT_residual_momentum.md
**Status**: Complete (868 lines)
**Strategy**: Beta-neutral momentum via OLS regression
**Complexity**: MEDIUM
**Priority**: HIGH (implement third)

Key features:
- Regresses ticker returns on SPY to extract residuals
- Ranks by cumulative residual returns
- Expected: Lower correlation to SPY, better risk-adjusted returns
- **Most complex, implement last**

Key features:
- Captures momentum inflection points
- Earlier entry/exit signals
- Expected: Higher CAGR, better timing

---

## Walk-Forward Optimizer Handouts (NEW)

### HANDOUT_wf_evaluator_dataclasses.md
**Status**: Complete
**Task**: IMPL-036-A
**Complexity**: LOW
**Creates**: `WalkForwardEvaluatorConfig`, `WindowResult`, `WalkForwardEvaluationResult` dataclasses

### HANDOUT_wf_evaluator_core.md
**Status**: Complete
**Task**: IMPL-036-B/C/D
**Complexity**: MEDIUM
**Creates**: `WalkForwardEvaluator` class with `evaluate()`, `_evaluate_window()`, `_calculate_composite_score()`

### HANDOUT_wf_optimizer_integration.md
**Status**: Complete
**Task**: IMPL-036-E/F/G
**Complexity**: MEDIUM
**Modifies**: `StrategyOptimizer` to use walk-forward evaluation

### HANDOUT_wf_cli_updates.md
**Status**: Complete
**Task**: IMPL-036-H/I/J
**Complexity**: LOW
**Modifies**: `scripts/run_backtests.py` with walk-forward CLI arguments

---

## Implementation Order

### Recommended (by computational simplicity):

1. **First**: Momentum Acceleration (simplest, fastest to implement)
2. **Second**: Volatility-Adjusted Momentum (defensive characteristics)
3. **Third**: Residual Momentum (most complex, requires regression)

### Alternative (by alpha potential):

1. **First**: Residual Momentum (highest alpha potential, but complex)
2. **Second**: Volatility-Adjusted Momentum (defensive characteristics)
3. **Third**: Momentum Acceleration (tactical timing)

All three can be implemented in parallel by different agents if desired.

## Using These Handouts

### For Coding Agents

1. Read the entire handout (don't skip sections)
2. Study the reference files listed in Section 9
3. Follow the implementation checklist in Section 10
4. Use the code template in Section 4 as starting point
5. Implement all test cases in Section 5
6. Verify against acceptance checklist in Section 8

### For Review

Use the acceptance checklist (Section 8) to verify:
- Code quality (type hints, docstrings, style)
- Functionality (correct outputs, error handling)
- Point-in-time compliance (no lookahead bias)
- Testing (>90% coverage, all edge cases)
- Integration (works with backtest engine)
- Performance (meets success criteria)

## Estimated Implementation Times

- **Residual Momentum**: 3-4 hours (includes regression, edge cases)
- **Volatility-Adjusted Momentum**: 2-3 hours (simpler calculation)
- **Momentum Acceleration**: 2-3 hours (simplest of the three)

**Total for Phase 1**: 7-10 hours for serial implementation

## Success Metrics

Each strategy must meet:

- **Sharpe Ratio**: > 0.3 (residual), > 0.6 (vol-adj), > 0.5 (accel)
- **Information Ratio vs SPY**: > 0.3
- **Max Drawdown**: < 40%
- **Code Coverage**: > 90%
- **All Tests Passing**: 8+ test cases per strategy

---

## Walk-Forward Optimizer Handouts (IMPL-036)

These handouts implement walk-forward validation for the strategy optimizer, replacing in-sample scoring with out-of-sample metrics to prevent overfitting.

### Implementation Order

| Task | Handout | Purpose | Complexity |
|------|---------|---------|------------|
| IMPL-036-A | `HANDOUT_wf_evaluator_dataclasses.md` | Config and result dataclasses | LOW |
| IMPL-036-B/C/D | `HANDOUT_wf_evaluator_core.md` | Core WalkForwardEvaluator class | MEDIUM |
| IMPL-036-E/F/G | `HANDOUT_wf_optimizer_integration.md` | Modify StrategyOptimizer | MEDIUM |
| IMPL-036-H/I/J | `HANDOUT_wf_cli_updates.md` | CLI arguments and output format | LOW |

### Master Reference

See `HANDOUT_walk_forward_optimizer.md` for the complete specification including:
- Problem statement and rationale
- Architecture overview
- Detailed requirements
- Success criteria
- Testing strategy
- Composite score formulas

### Quick Start

1. Start with `HANDOUT_wf_evaluator_dataclasses.md` - create the data structures
2. Then `HANDOUT_wf_evaluator_core.md` - implement the evaluator
3. Then `HANDOUT_wf_optimizer_integration.md` - integrate with optimizer
4. Finally `HANDOUT_wf_cli_updates.md` - add CLI support

---

## Related Documents

- `/workspaces/qetf/STRATEGY_RESEARCH_PLAN.md` - Overall research plan (10+ strategies)
- `/workspaces/qetf/handoffs/PHASE3_IMPLEMENTATION_PLAN.md` - Previous phase context
- `/workspaces/qetf/src/quantetf/alpha/momentum.py` - Primary reference implementation

## Questions?

If any part of a handout is unclear:
1. Check the reference files (Section 9) for concrete examples
2. Look at existing alpha models in `/workspaces/qetf/src/quantetf/alpha/`
3. Review test patterns in `/workspaces/qetf/tests/`

---

**Created**: 2026-01-13
**Author**: Quant Research Architect
**Status**: All 3 handouts complete and ready for implementation! ✅
