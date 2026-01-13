# CLAUDE_CONTEXT.md

**Last Updated:** January 13, 2026
**Project:** QuantETF

-----

## How to Use This File

This file contains instructions and context for Claude across all sessions. Read this file at the start of every session by fetching:

```
https://raw.githubusercontent.com/mikemsq/qetf/refs/heads/main/CLAUDE_CONTEXT.md
```

Always check PROGRESS_LOG.md for current status.

-----

## Project Overview

**What we’re building:**  
QuantETF - A modular quantitative investment platform for ETF-based strategies with backtesting, research tools, and automated rebalancing recommendations to exceed S&P 500 returns.

**Current phase:**  
Planning / Initial Development

**Tech stack:**

- **Language:** Python 3.x
- **Package Manager:** uv
- **Data Processing:** pandas, numpy
- **Backtesting:** Custom engine in src/quantetf/backtest
- **Data Sources:** TBD (ETF pricing and fundamentals APIs)
- **Notebooks:** Jupyter for research and exploration
- **Testing:** pytest
- **Version Control:** Git/GitHub

-----

## Project Files Structure

```
qetf/
├── PROJECT_BRIEF.md          # Strategic overview and goals
├── CLAUDE_CONTEXT.md         # This file - read first
├── PROGRESS_LOG.md           # Daily updates - check current status
├── README.md                 # Public-facing documentation
├── pyproject.toml            # Python dependencies and config
├── uv.lock                   # Locked dependency versions
├── /configs                  # Strategy and universe configs (YAML)
├── /data                     # Data storage
│   ├── /raw                  # Immutable ingested data
│   ├── /curated              # Cleaned, normalized data
│   └── /snapshots            # Versioned point-in-time datasets
├── /artifacts                # Output bundles (metrics, plots, recommendations)
├── /notebooks                # Jupyter notebooks for research
├── /scripts                  # Utility scripts (ingest, backtest, etc.)
├── /session-notes            # Notes from each Claude session
├── /src/quantetf             # Main library code
│   ├── /data                 # Data ingestion and connectors
│   ├── /universe             # Universe builders and filters
│   ├── /features             # Feature engineering
│   ├── /alpha                # Alpha models
│   ├── /risk                 # Risk models and covariance
│   ├── /portfolio            # Portfolio construction
│   ├── /backtest             # Backtest engine
│   ├── /evaluation           # Metrics and reporting
│   ├── /production           # Production runtime
│   ├── /cli                  # Command line interface
│   └── /utils                # Shared utilities
└── /tests                    # Unit and integration tests
```

-----

## Core Principles

### 1. Always verify before implementing

- Read the full requirements before starting
- Ask clarifying questions if anything is ambiguous
- Propose a plan before writing code
- **Critical for quant:** Prevent lookahead bias and data leakage

### 2. Keep it simple

- Use the simplest solution that works
- Avoid over-engineering
- Write clear, readable code over clever code
- Financial calculations must be transparent and verifiable

### 3. Document as you go

- Update PROGRESS_LOG.md after completing tasks
- Add docstrings for all functions and classes
- Document decisions in /docs/decisions/ (if needed)
- Explain calculation methods and data sources
- Document “as-of” dates for all point-in-time operations

### 4. Test your work

- Write tests for financial calculations with known expected values
- Test edge cases (missing data, market holidays, boundary conditions)
- Validate against synthetic datasets with known outcomes
- Run: `pytest tests/` before committing
- Check for lookahead bias in backtests

-----

## Coding Standards

### Style Guide

- **Language:** Python 3.x
- **Formatting:** Follow PEP 8, use Black formatter (4 spaces)
- **Type hints:** Use type annotations for function signatures
- **Naming conventions:**
  - Variables: snake_case (e.g., `portfolio_value`, `etf_data`)
  - Functions: snake_case (e.g., `calculate_returns`, `fetch_prices`)
  - Classes: PascalCase (e.g., `UniverseProvider`, `AlphaModel`)
  - Constants: UPPER_SNAKE_CASE (e.g., `API_KEY`, `MAX_RETRIES`)
  - Files: snake_case.py (e.g., `portfolio_constructor.py`)
  - Private methods: _leading_underscore

### Code Organization

- One class per file when practical
- Keep modules under 300 lines when possible
- Group related functionality together
- Separate data fetching from calculation logic
- Keep business logic separate from I/O operations
- Use dataclasses for data containers

### File Path Handling

- **Always use `pathlib.Path` objects for file paths, not strings**
- Accept `Union[str, Path]` at API boundaries, convert to `Path` immediately
- Use Path objects for all internal path manipulation and passing between methods
- Benefits: type safety, cross-platform compatibility, rich API, immutability
- Example:
  ```python
  from pathlib import Path
  from typing import Union

  def load_config(config_path: Union[str, Path]) -> dict:
      """Load configuration from file."""
      config_path = Path(config_path)  # Convert immediately
      if not config_path.exists():
          raise FileNotFoundError(f"{config_path} not found")
      return yaml.safe_load(config_path.read_text())
  ```

### Documentation

- Write clear docstrings for all public functions/classes
- Use Google-style docstrings:
  
  ```python
  def calculate_returns(prices: pd.DataFrame) -> pd.Series:
      """Calculate returns from price series.
      
      Args:
          prices: DataFrame with datetime index and price columns
          
      Returns:
          Series of percentage returns
          
      Raises:
          ValueError: If prices contain NaN or negative values
      """
  ```
- Add comments for “why”, not “what”
- **Always document financial formulas and assumptions**
- Document data sources and expected formats

-----

## Things to ALWAYS Do

✅ Read PROGRESS_LOG.md before starting work  
✅ Check if similar code exists before creating new patterns  
✅ Run tests before marking work complete (`pytest tests/`)  
✅ Update PROGRESS_LOG.md when finishing a task  
✅ Create small, focused commits with clear messages  
✅ Add error handling for data fetching and API calls  
✅ Validate input data before calculations  
✅ Use explicit `as_of` dates for all point-in-time operations  
✅ Consider edge cases (missing data, market holidays, delisted ETFs)  
✅ Use appropriate numeric types (avoid float precision issues for money)  
✅ Document data sources and assumptions  
✅ Check for lookahead bias in backtests

-----

## Things to NEVER Do

❌ Don’t modify files outside the current task scope  
❌ Don’t delete or comment out code without discussion  
❌ Don’t introduce new dependencies without noting in PROGRESS_LOG.md  
❌ Don’t commit broken/non-functional code  
❌ Don’t ignore errors or warnings  
❌ Don’t hardcode API keys or credentials (use environment variables)  
❌ Don’t skip input validation  
❌ Don’t use future data in backtests (lookahead bias)  
❌ Don’t make financial calculations without documenting the formula  
❌ Don’t use mutable default arguments in Python  
❌ Don’t ignore timezone issues with financial data  
❌ Don’t cache stale data without timestamps and freshness checks

-----

## Common Patterns

### Pattern 1: Data Fetching with Error Handling

```python
import logging
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)

def fetch_etf_prices(
    ticker: str,
    start_date: str,
    end_date: str
) -> Optional[pd.DataFrame]:
    """Fetch ETF price data with robust error handling.
    
    Args:
        ticker: ETF ticker symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        DataFrame with datetime index and OHLCV columns, or None if failed
    """
    try:
        # API call here
        data = external_api.get_prices(ticker, start_date, end_date)
        
        # Validate required fields
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Missing required columns for {ticker}")
        
        # Check for data quality issues
        if data['close'].isna().any():
            logger.warning(f"Missing close prices for {ticker}")
        
        return data
        
    except Exception as e:
        logger.error(f"Error fetching prices for {ticker}: {e}")
        return None
```

### Pattern 2: Point-in-Time Operations (No Lookahead)

```python
from datetime import datetime
import pandas as pd

def compute_features_as_of(
    universe: list[str],
    as_of_date: datetime,
    lookback_days: int = 252
) -> pd.DataFrame:
    """Compute features using only data available as of the given date.
    
    Critical: This function must not use any data after as_of_date
    to prevent lookahead bias in backtests.
    
    Args:
        universe: List of ETF tickers
        as_of_date: The date as of which features are computed
        lookback_days: Days of history to use for calculations
        
    Returns:
        DataFrame with features for each ticker, indexed by ticker
    """
    # Only fetch data up to (but not including) as_of_date
    end_date = as_of_date
    start_date = as_of_date - pd.Timedelta(days=lookback_days)
    
    features = {}
    for ticker in universe:
        prices = fetch_etf_prices(ticker, start_date, end_date)
        if prices is None:
            continue
            
        # Compute momentum using only historical data
        returns = prices['close'].pct_change()
        momentum_252d = returns.iloc[-252:].sum()
        
        features[ticker] = {
            'momentum_252d': momentum_252d,
            'as_of_date': as_of_date
        }
    
    return pd.DataFrame.from_dict(features, orient='index')
```

### Pattern 3: Financial Calculations with Validation

```python
import numpy as np
from typing import Union

def calculate_portfolio_returns(
    weights: pd.Series,
    returns: pd.DataFrame
) -> pd.Series:
    """Calculate portfolio returns from weights and asset returns.
    
    Formula: r_p(t) = Σ(w_i * r_i(t))
    
    Args:
        weights: Series of portfolio weights (should sum to ~1.0)
        returns: DataFrame of asset returns, columns are tickers
        
    Returns:
        Series of portfolio returns over time
        
    Raises:
        ValueError: If weights don't sum to approximately 1.0
        ValueError: If tickers don't match between weights and returns
    """
    # Validate inputs
    weight_sum = weights.sum()
    if not np.isclose(weight_sum, 1.0, atol=0.01):
        raise ValueError(f"Weights sum to {weight_sum:.4f}, expected 1.0")
    
    # Check ticker alignment
    missing_tickers = set(weights.index) - set(returns.columns)
    if missing_tickers:
        raise ValueError(f"Missing return data for: {missing_tickers}")
    
    # Align and calculate
    aligned_returns = returns[weights.index]
    portfolio_returns = (aligned_returns * weights).sum(axis=1)
    
    return portfolio_returns
```

### Pattern 4: Configuration-Driven Code

```python
from pathlib import Path
import yaml
from dataclasses import dataclass

@dataclass
class StrategyConfig:
    """Configuration for a quantitative strategy."""
    name: str
    universe_type: str
    rebalance_frequency: str
    lookback_days: int
    top_n: int
    max_position_size: float
    
    @classmethod
    def from_yaml(cls, path: Path) -> 'StrategyConfig':
        """Load strategy config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

# Usage:
# config = StrategyConfig.from_yaml(Path('configs/momentum_strategy.yaml'))
```

-----

## Common Mistakes (Learn from these!)

### Issue: [Issues will be added as they occur]

**What happened:** [Brief description]  
**Why it happened:** [Root cause]  
**Solution:** [How to avoid it]  
**Date added:** [Date]

-----

## Financial Data Guidelines

### Data Accuracy

- Always validate data from external sources
- Cross-reference critical calculations
- Handle missing data explicitly (don't forward-fill blindly)
- Display "as-of" timestamps with all financial data
- Indicate when data is delayed, estimated, or backfilled
- Check for survivorship bias in historical datasets

### Performance Analysis Standards

**CRITICAL: Always show relative performance, not just absolute performance**

The goal of QuantETF is to **beat SPY (S&P 500)**. All performance analysis must emphasize comparison to benchmark:

- **Always show three metrics together:**
  1. Portfolio performance (strategy returns)
  2. Benchmark performance (SPY buy-and-hold)
  3. Active performance (strategy excess returns vs benchmark)

- **Active performance is the primary metric** - it shows whether the strategy is adding value
- Pure portfolio performance is less interesting than relative performance
- Default visualizations should overlay strategy vs SPY equity curves
- Metrics tables should show strategy metrics, SPY metrics, and differences side-by-side
- Reports should lead with "Strategy beat SPY by X%" or "Strategy underperformed SPY by X%"

**Examples:**
- ❌ Wrong: "Strategy returned 45% with Sharpe 1.2"
- ✅ Right: "Strategy returned 45% vs SPY 35% (+10% active return), Sharpe 1.2 vs 0.9"

- ❌ Wrong: Chart showing only strategy equity curve
- ✅ Right: Chart showing strategy vs SPY equity curves overlaid with shaded active return area

**Implementation:**
- Notebooks should load SPY benchmark data automatically
- All visualization functions should accept optional benchmark parameter (default to SPY)
- Metrics calculations should include `calculate_active_metrics(strategy, benchmark)` helper
- HTML reports should have "Active Performance" section prominently displayed

**Warmup Period Alignment (CRITICAL):**
- Strategies with lookback periods (momentum, technical indicators) require warmup time before signals are available
- **ALWAYS align benchmark start date to when strategy starts trading** (after warmup)
- Example: 252-day momentum needs ~1 year of warmup before first signal
- Detect first active trading date: find first rebalance where portfolio has non-cash positions
- Start SPY benchmark from this same date to ensure fair comparison
- Document warmup period in analysis (e.g., "Warmup: 365 days, ~1.4 years")
- Without alignment, benchmark gets unfair head start while strategy holds cash

### Number Precision

- Use Decimal for money if precision is critical
- Be aware of floating point limitations
- Round appropriately for display (2 decimals for dollars, 4 for %)
- Don’t compare floats with == (use np.isclose)

### Date/Time Handling

- Always use timezone-aware datetime objects (UTC for storage)
- Handle market hours and non-trading days
- Be explicit about business day vs calendar day calculations
- Use pandas’ business day utilities for market calendars
- Store all timestamps in UTC, convert to local only for display

### Survivorship Bias

- Be cautious of using only currently-listed ETFs
- Consider delisted/merged ETFs in historical analysis
- Document if analysis includes survivorship bias

-----

## Questions to Ask

If you're unsure about anything, ask these questions:

- What is the user trying to accomplish?
- What's the simplest way to achieve this?
- Does this fit with our existing patterns?
- How can we verify this works correctly?
- Is there risk of lookahead bias?
- What could go wrong?
- Is there existing code that does something similar?
- Does this need to be point-in-time aware?

-----

## Important Reminders

- 📖 Always read PROGRESS_LOG.md first for current status
- 🎯 Understand the goal before coding
- ⏰ Use explicit as_of dates for all point-in-time operations
- 🚫 Never use future data in backtests (lookahead bias)
- ✅ Verify your work before finishing
- 📝 Update session notes incrementally as you work (don't wait until end)
- 🤝 Save progress frequently to avoid quota issues
- 🧪 Test with synthetic data when possible

-----

## Development Workflow

See [AGENT_WORKFLOW.md](AGENT_WORKFLOW.md) for the multi-agent development process.

For quick reference:
1. Check [TASKS.md](TASKS.md) for available work
2. Read relevant handoff file from [handoffs/](handoffs/)
3. Implement following this file's coding standards
4. Update session notes as you go
5. Commit and push changes

-----

## Version History

|Date      |Change                    |Reason                                            |
|----------|--------------------------|--------------------------------------------------|
|2026-01-06|Initial creation          |Project setup                                     |
|2026-01-07|Updated for Python project|Fixed JS examples, added quant-specific guidelines|
|2026-01-09|Streamlined duplication   |Removed session workflow (now in AGENT_WORKFLOW.md)|
|2026-01-13|Added Path object standard|Mandate pathlib.Path for all file path handling  |