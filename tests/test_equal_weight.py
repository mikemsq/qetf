"""Tests for equal-weight portfolio constructor."""

import pandas as pd
import pytest
import numpy as np

from quantetf.portfolio.equal_weight import EqualWeightTopN
from quantetf.types import AlphaScores, Universe, TargetWeights


class TestEqualWeightTopN:
    """Test suite for EqualWeightTopN portfolio constructor."""

    def test_init_valid_top_n(self):
        """Test initialization with valid top_n parameter."""
        constructor = EqualWeightTopN(top_n=5)
        assert constructor.top_n == 5

    def test_init_default_top_n(self):
        """Test initialization with default top_n."""
        constructor = EqualWeightTopN()
        assert constructor.top_n == 5

    def test_init_invalid_top_n(self):
        """Test initialization with invalid top_n raises ValueError."""
        with pytest.raises(ValueError, match="top_n must be >= 1"):
            EqualWeightTopN(top_n=0)

        with pytest.raises(ValueError, match="top_n must be >= 1"):
            EqualWeightTopN(top_n=-5)

    def test_basic_equal_weight_selection(self):
        """Test basic equal-weight selection with top_n < num_tickers."""
        # Create test data with 6 tickers
        tickers = ['A', 'B', 'C', 'D', 'E', 'F']
        scores = pd.Series([0.5, 0.3, 0.8, 0.1, 0.6, 0.2], index=tickers)

        alpha = AlphaScores(
            as_of=pd.Timestamp("2023-12-31"),
            scores=scores
        )

        universe = Universe(
            as_of=pd.Timestamp("2023-12-31"),
            tickers=tuple(tickers)
        )

        # Construct portfolio with top 3
        constructor = EqualWeightTopN(top_n=3)
        weights = constructor.construct(
            as_of=pd.Timestamp("2023-12-31"),
            universe=universe,
            alpha=alpha,
            risk=None,
            store=None
        )

        # Verify type
        assert isinstance(weights, TargetWeights)

        # Verify top 3: C (0.8), E (0.6), A (0.5) each get 1/3
        assert weights.weights['C'] == pytest.approx(1/3)
        assert weights.weights['E'] == pytest.approx(1/3)
        assert weights.weights['A'] == pytest.approx(1/3)

        # Verify non-selected get 0.0
        assert weights.weights['B'] == 0.0
        assert weights.weights['D'] == 0.0
        assert weights.weights['F'] == 0.0

        # Verify sum
        assert weights.weights.sum() == pytest.approx(1.0)

        # Verify diagnostics
        assert weights.diagnostics['top_n'] == 3
        assert weights.diagnostics['num_selected'] == 3
        assert set(weights.diagnostics['selected']) == {'C', 'E', 'A'}
        assert weights.diagnostics['num_valid_scores'] == 6

    def test_with_nan_scores(self):
        """Test that NaN scores are skipped correctly."""
        tickers = ['A', 'B', 'C', 'D', 'E']
        scores = pd.Series([0.5, np.nan, 0.8, np.nan, 0.6], index=tickers)

        alpha = AlphaScores(
            as_of=pd.Timestamp("2023-12-31"),
            scores=scores
        )

        universe = Universe(
            as_of=pd.Timestamp("2023-12-31"),
            tickers=tuple(tickers)
        )

        # Top 2 from valid scores (A, C, E)
        constructor = EqualWeightTopN(top_n=2)
        weights = constructor.construct(
            as_of=pd.Timestamp("2023-12-31"),
            universe=universe,
            alpha=alpha,
            risk=None,
            store=None
        )

        # Top 2 valid: C (0.8), E (0.6)
        assert weights.weights['C'] == pytest.approx(0.5)
        assert weights.weights['E'] == pytest.approx(0.5)
        assert weights.weights['A'] == 0.0  # Valid but not in top 2
        assert weights.weights['B'] == 0.0  # NaN
        assert weights.weights['D'] == 0.0  # NaN

        assert weights.weights.sum() == pytest.approx(1.0)
        assert weights.diagnostics['num_valid_scores'] == 3
        assert weights.diagnostics['num_selected'] == 2

    def test_top_n_exceeds_available_scores(self):
        """Test when top_n is greater than available valid scores."""
        tickers = ['A', 'B', 'C']
        scores = pd.Series([0.5, 0.3, 0.8], index=tickers)

        alpha = AlphaScores(
            as_of=pd.Timestamp("2023-12-31"),
            scores=scores
        )

        universe = Universe(
            as_of=pd.Timestamp("2023-12-31"),
            tickers=tuple(tickers)
        )

        # Request top 10 but only 3 available
        constructor = EqualWeightTopN(top_n=10)
        weights = constructor.construct(
            as_of=pd.Timestamp("2023-12-31"),
            universe=universe,
            alpha=alpha,
            risk=None,
            store=None
        )

        # All 3 should be selected with equal weight
        assert weights.weights['C'] == pytest.approx(1/3)
        assert weights.weights['A'] == pytest.approx(1/3)
        assert weights.weights['B'] == pytest.approx(1/3)

        assert weights.weights.sum() == pytest.approx(1.0)
        assert weights.diagnostics['num_selected'] == 3

    def test_all_nan_scores(self):
        """Test when all scores are NaN (no valid data)."""
        tickers = ['A', 'B', 'C']
        scores = pd.Series([np.nan, np.nan, np.nan], index=tickers)

        alpha = AlphaScores(
            as_of=pd.Timestamp("2023-12-31"),
            scores=scores
        )

        universe = Universe(
            as_of=pd.Timestamp("2023-12-31"),
            tickers=tuple(tickers)
        )

        constructor = EqualWeightTopN(top_n=2)
        weights = constructor.construct(
            as_of=pd.Timestamp("2023-12-31"),
            universe=universe,
            alpha=alpha,
            risk=None,
            store=None
        )

        # All weights should be 0.0
        assert (weights.weights == 0.0).all()
        assert weights.weights.sum() == 0.0
        assert weights.diagnostics['num_valid_scores'] == 0
        assert weights.diagnostics['selected'] == []

    def test_single_ticker_universe(self):
        """Test with single ticker universe."""
        tickers = ['SPY']
        scores = pd.Series([0.5], index=tickers)

        alpha = AlphaScores(
            as_of=pd.Timestamp("2023-12-31"),
            scores=scores
        )

        universe = Universe(
            as_of=pd.Timestamp("2023-12-31"),
            tickers=tuple(tickers)
        )

        constructor = EqualWeightTopN(top_n=5)
        weights = constructor.construct(
            as_of=pd.Timestamp("2023-12-31"),
            universe=universe,
            alpha=alpha,
            risk=None,
            store=None
        )

        # Single ticker gets 100% weight
        assert weights.weights['SPY'] == pytest.approx(1.0)
        assert weights.weights.sum() == pytest.approx(1.0)
        assert weights.diagnostics['num_selected'] == 1

    def test_weights_index_matches_universe(self):
        """Test that returned weights index exactly matches universe tickers."""
        tickers = ['A', 'B', 'C', 'D']
        scores = pd.Series([0.5, 0.3, 0.8, 0.1], index=tickers)

        alpha = AlphaScores(
            as_of=pd.Timestamp("2023-12-31"),
            scores=scores
        )

        universe = Universe(
            as_of=pd.Timestamp("2023-12-31"),
            tickers=tuple(tickers)
        )

        constructor = EqualWeightTopN(top_n=2)
        weights = constructor.construct(
            as_of=pd.Timestamp("2023-12-31"),
            universe=universe,
            alpha=alpha,
            risk=None,
            store=None
        )

        # Weights index should match universe tickers exactly
        assert list(weights.weights.index) == list(tickers)
        assert len(weights.weights) == len(tickers)

    def test_negative_scores(self):
        """Test with negative alpha scores (momentum can be negative)."""
        tickers = ['A', 'B', 'C', 'D']
        scores = pd.Series([-0.2, 0.1, -0.5, 0.3], index=tickers)

        alpha = AlphaScores(
            as_of=pd.Timestamp("2023-12-31"),
            scores=scores
        )

        universe = Universe(
            as_of=pd.Timestamp("2023-12-31"),
            tickers=tuple(tickers)
        )

        constructor = EqualWeightTopN(top_n=2)
        weights = constructor.construct(
            as_of=pd.Timestamp("2023-12-31"),
            universe=universe,
            alpha=alpha,
            risk=None,
            store=None
        )

        # Top 2: D (0.3), B (0.1) - highest values even if some are negative
        assert weights.weights['D'] == pytest.approx(0.5)
        assert weights.weights['B'] == pytest.approx(0.5)
        assert weights.weights['A'] == 0.0
        assert weights.weights['C'] == 0.0

        assert weights.weights.sum() == pytest.approx(1.0)

    def test_tied_scores(self):
        """Test behavior with tied alpha scores."""
        tickers = ['A', 'B', 'C', 'D']
        scores = pd.Series([0.5, 0.5, 0.8, 0.3], index=tickers)

        alpha = AlphaScores(
            as_of=pd.Timestamp("2023-12-31"),
            scores=scores
        )

        universe = Universe(
            as_of=pd.Timestamp("2023-12-31"),
            tickers=tuple(tickers)
        )

        constructor = EqualWeightTopN(top_n=2)
        weights = constructor.construct(
            as_of=pd.Timestamp("2023-12-31"),
            universe=universe,
            alpha=alpha,
            risk=None,
            store=None
        )

        # C (0.8) is definitely selected, then one of A or B (0.5 each)
        assert weights.weights['C'] == pytest.approx(0.5)

        # One of A or B should be selected (pandas sort is stable, so first occurrence wins)
        selected_tied = [t for t in ['A', 'B'] if weights.weights[t] > 0]
        assert len(selected_tied) == 1
        assert weights.weights[selected_tied[0]] == pytest.approx(0.5)

        assert weights.weights.sum() == pytest.approx(1.0)
        assert weights.diagnostics['num_selected'] == 2

    def test_as_of_date_preserved(self):
        """Test that as_of date is correctly preserved in output."""
        tickers = ['A', 'B', 'C']
        scores = pd.Series([0.5, 0.3, 0.8], index=tickers)

        as_of_date = pd.Timestamp("2024-06-15")

        alpha = AlphaScores(
            as_of=as_of_date,
            scores=scores
        )

        universe = Universe(
            as_of=as_of_date,
            tickers=tuple(tickers)
        )

        constructor = EqualWeightTopN(top_n=2)
        weights = constructor.construct(
            as_of=as_of_date,
            universe=universe,
            alpha=alpha,
            risk=None,
            store=None
        )

        assert weights.as_of == as_of_date

    def test_diagnostics_complete(self):
        """Test that diagnostics contain all expected fields."""
        tickers = ['A', 'B', 'C', 'D']
        scores = pd.Series([0.5, 0.3, 0.8, 0.1], index=tickers)

        alpha = AlphaScores(
            as_of=pd.Timestamp("2023-12-31"),
            scores=scores
        )

        universe = Universe(
            as_of=pd.Timestamp("2023-12-31"),
            tickers=tuple(tickers)
        )

        constructor = EqualWeightTopN(top_n=2)
        weights = constructor.construct(
            as_of=pd.Timestamp("2023-12-31"),
            universe=universe,
            alpha=alpha,
            risk=None,
            store=None
        )

        # Check all expected diagnostic fields are present
        assert 'top_n' in weights.diagnostics
        assert 'selected' in weights.diagnostics
        assert 'num_valid_scores' in weights.diagnostics
        assert 'num_universe_tickers' in weights.diagnostics
        assert 'num_selected' in weights.diagnostics
        assert 'weight_per_ticker' in weights.diagnostics
        assert 'total_weight' in weights.diagnostics

        # Verify values
        assert weights.diagnostics['top_n'] == 2
        assert len(weights.diagnostics['selected']) == 2
        assert weights.diagnostics['num_valid_scores'] == 4
        assert weights.diagnostics['num_universe_tickers'] == 4
        assert weights.diagnostics['num_selected'] == 2
        assert weights.diagnostics['weight_per_ticker'] == pytest.approx(0.5)
        assert weights.diagnostics['total_weight'] == pytest.approx(1.0)

    def test_large_universe(self):
        """Test with larger universe (20 ETFs like production case)."""
        # Create 20 ETF tickers
        tickers = [f'ETF{i:02d}' for i in range(20)]
        # Random-ish scores
        scores = pd.Series(
            [0.15, -0.05, 0.25, 0.10, -0.10, 0.30, 0.05, -0.15, 0.20, 0.08,
             0.12, -0.08, 0.18, 0.22, -0.12, 0.28, 0.03, -0.02, 0.16, 0.14],
            index=tickers
        )

        alpha = AlphaScores(
            as_of=pd.Timestamp("2023-12-31"),
            scores=scores
        )

        universe = Universe(
            as_of=pd.Timestamp("2023-12-31"),
            tickers=tuple(tickers)
        )

        # Production case: top 5 out of 20
        constructor = EqualWeightTopN(top_n=5)
        weights = constructor.construct(
            as_of=pd.Timestamp("2023-12-31"),
            universe=universe,
            alpha=alpha,
            risk=None,
            store=None
        )

        # Verify exactly 5 non-zero weights
        non_zero = (weights.weights > 0).sum()
        assert non_zero == 5

        # Each should be 0.20
        selected_weights = weights.weights[weights.weights > 0]
        assert all(w == pytest.approx(0.20) for w in selected_weights)

        # Sum should be 1.0
        assert weights.weights.sum() == pytest.approx(1.0)

        # Verify top 5 scores are selected
        top_5_tickers = scores.sort_values(ascending=False).head(5).index.tolist()
        assert set(weights.diagnostics['selected']) == set(top_5_tickers)
