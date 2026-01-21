#!/usr/bin/env python3
"""Simple test to validate cash handling in backtest engine."""

import pandas as pd
from quantetf.portfolio.equal_weight import EqualWeightTopN
from quantetf.types import AlphaScores, Universe, CASH_TICKER

def test_cash_allocation():
    """Test that EqualWeightTopN allocates to cash when insufficient valid scores."""

    # Create constructor
    constructor = EqualWeightTopN(top_n=5)

    # Create universe with 3 tickers
    universe = Universe(as_of=pd.Timestamp("2023-01-01"), tickers=("SPY", "QQQ", "IWM"))

    # Test 1: All tickers have valid scores - should allocate to top 5 but only 3 available
    alpha_scores = pd.Series([0.8, 0.6, 0.4], index=["SPY", "QQQ", "IWM"])
    alpha = AlphaScores(as_of=pd.Timestamp("2023-01-01"), scores=alpha_scores)

    weights = constructor.construct(
        as_of=pd.Timestamp("2023-01-01"),
        universe=universe,
        alpha=alpha,
        risk=None,
        store=None
    )

    print("Test 1 - All valid scores:")
    print(f"Total weight: {weights.weights.sum():.6f}")
    print(f"Cash weight: {weights.weights.get(CASH_TICKER, 0):.6f}")
    print(f"Selected tickers: {weights.diagnostics['selected']}")
    print()

    # Test 2: Only 2 valid scores - should allocate remaining weight to cash
    alpha_scores_partial = pd.Series([0.8, 0.6, float('nan')], index=["SPY", "QQQ", "IWM"])
    alpha_partial = AlphaScores(as_of=pd.Timestamp("2023-01-01"), scores=alpha_scores_partial)

    weights_partial = constructor.construct(
        as_of=pd.Timestamp("2023-01-01"),
        universe=universe,
        alpha=alpha_partial,
        risk=None,
        store=None
    )

    print("Test 2 - Only 2 valid scores:")
    print(f"Total weight: {weights_partial.weights.sum():.6f}")
    print(f"Cash weight: {weights_partial.weights.get(CASH_TICKER, 0):.6f}")
    print(f"Selected tickers: {weights_partial.diagnostics['selected']}")
    print()

    # Test 3: No valid scores - should allocate 100% to cash
    alpha_scores_none = pd.Series([float('nan'), float('nan'), float('nan')], index=["SPY", "QQQ", "IWM"])
    alpha_none = AlphaScores(as_of=pd.Timestamp("2023-01-01"), scores=alpha_scores_none)

    weights_none = constructor.construct(
        as_of=pd.Timestamp("2023-01-01"),
        universe=universe,
        alpha=alpha_none,
        risk=None,
        store=None
    )

    print("Test 3 - No valid scores:")
    print(f"Total weight: {weights_none.weights.sum():.6f}")
    print(f"Cash weight: {weights_none.weights.get(CASH_TICKER, 0):.6f}")
    print(f"Selected tickers: {weights_none.diagnostics['selected']}")
    print()

if __name__ == "__main__":
    test_cash_allocation()