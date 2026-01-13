"""Configuration loader for strategy YAML files.

This module loads and parses strategy configuration files, instantiating
all necessary components for backtesting.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml
import logging
import pandas as pd

from quantetf.alpha.factory import create_alpha_model
from quantetf.alpha.base import AlphaModel
from quantetf.portfolio.base import PortfolioConstructor
from quantetf.portfolio.equal_weight import EqualWeightTopN
from quantetf.portfolio.costs import FlatTransactionCost
from quantetf.types import Universe

logger = logging.getLogger(__name__)


class StrategyConfig:
    """Represents a complete strategy configuration.

    This class holds all components needed to run a backtest, loaded
    from a YAML configuration file.
    """

    def __init__(
        self,
        name: str,
        alpha_model: AlphaModel,
        portfolio_construction: PortfolioConstructor,
        cost_model: FlatTransactionCost,
        universe_tickers: tuple[str, ...],
        rebalance_frequency: str = 'monthly',
        description: Optional[str] = None,
        raw_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize strategy config.

        Args:
            name: Strategy name
            alpha_model: Instantiated alpha model
            portfolio_construction: Instantiated portfolio construction
            cost_model: Instantiated cost model
            universe_tickers: Tuple of ticker symbols
            rebalance_frequency: Rebalance frequency ('monthly', 'weekly', etc.)
            description: Optional strategy description
            raw_config: Raw config dict for reference
        """
        self.name = name
        self.alpha_model = alpha_model
        self.portfolio_construction = portfolio_construction
        self.cost_model = cost_model
        self.universe_tickers = universe_tickers
        self.rebalance_frequency = rebalance_frequency
        self.description = description
        self.raw_config = raw_config or {}

    def create_universe(self, as_of: pd.Timestamp) -> Universe:
        """Create a Universe object for a given date.

        Args:
            as_of: Point-in-time date for universe

        Returns:
            Universe object with configured tickers
        """
        return Universe(as_of=as_of, tickers=self.universe_tickers)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"StrategyConfig(name='{self.name}', "
            f"alpha={type(self.alpha_model).__name__}, "
            f"portfolio={type(self.portfolio_construction).__name__}, "
            f"universe_size={len(self.universe_tickers)})"
        )


def load_universe_config(universe_path: Path) -> tuple[str, ...]:
    """Load universe tickers from YAML file.

    Args:
        universe_path: Path to universe config file

    Returns:
        Tuple of ticker symbols

    Raises:
        FileNotFoundError: If universe file doesn't exist
        ValueError: If universe config is invalid
    """
    if not universe_path.exists():
        raise FileNotFoundError(f"Universe config not found: {universe_path}")

    with open(universe_path) as f:
        universe_config = yaml.safe_load(f)

    # Try to find tickers in different locations
    if 'tickers' in universe_config:
        tickers = universe_config['tickers']
    elif 'source' in universe_config and 'tickers' in universe_config['source']:
        tickers = universe_config['source']['tickers']
    else:
        raise ValueError(f"Universe config must have 'tickers' or 'source.tickers' field: {universe_path}")

    if not isinstance(tickers, list):
        raise ValueError(f"Universe 'tickers' must be a list: {universe_path}")

    return tuple(tickers)


def load_schedule_config(schedule_path: Path) -> str:
    """Load rebalance schedule from YAML file.

    Args:
        schedule_path: Path to schedule config file

    Returns:
        Rebalance frequency string ('monthly', 'weekly', etc.)

    Raises:
        FileNotFoundError: If schedule file doesn't exist
    """
    if not schedule_path.exists():
        raise FileNotFoundError(f"Schedule config not found: {schedule_path}")

    with open(schedule_path) as f:
        schedule_config = yaml.safe_load(f)

    # Extract frequency (default to 'monthly' if not specified)
    frequency = schedule_config.get('frequency', 'monthly')

    return frequency


def load_cost_config(cost_path: Path) -> FlatTransactionCost:
    """Load cost model from YAML file.

    Args:
        cost_path: Path to cost config file

    Returns:
        Instantiated cost model

    Raises:
        FileNotFoundError: If cost file doesn't exist
    """
    if not cost_path.exists():
        raise FileNotFoundError(f"Cost config not found: {cost_path}")

    with open(cost_path) as f:
        cost_config = yaml.safe_load(f)

    cost_type = cost_config.get('type', 'flat')

    if cost_type == 'flat':
        cost_bps = cost_config.get('cost_bps', 10.0)
        return FlatTransactionCost(cost_bps=cost_bps)
    else:
        raise ValueError(f"Unsupported cost type: {cost_type}")


def create_portfolio_construction(config: Dict[str, Any]) -> PortfolioConstructor:
    """Create portfolio construction from config dict.

    Args:
        config: Portfolio construction config with 'type' and parameters

    Returns:
        Instantiated portfolio construction object

    Raises:
        ValueError: If portfolio type is unsupported
    """
    portfolio_type = config.get('type', 'equal_weight_top_n')

    if portfolio_type == 'equal_weight_top_n':
        top_n = config.get('top_n', 5)
        # Note: constraints are ignored for now, can be added later
        return EqualWeightTopN(top_n=top_n)
    else:
        raise ValueError(f"Unsupported portfolio construction type: {portfolio_type}")


def load_strategy_config(config_path: Path | str, base_dir: Optional[Path] = None) -> StrategyConfig:
    """Load complete strategy configuration from YAML file.

    Args:
        config_path: Path to strategy config YAML file
        base_dir: Base directory for resolving relative paths (defaults to config file parent)

    Returns:
        StrategyConfig object with all components instantiated

    Raises:
        FileNotFoundError: If config file or referenced files don't exist
        ValueError: If config is invalid

    Example:
        >>> config = load_strategy_config('configs/strategies/momentum_acceleration_top5.yaml')
        >>> print(config.name)
        momentum_acceleration_top5_ew
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Strategy config not found: {config_path}")

    # Set base directory for resolving relative paths
    if base_dir is None:
        base_dir = config_path.parent.parent.parent  # Go up to project root

    logger.info(f"Loading strategy config: {config_path}")

    # Load main config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Extract strategy name
    name = config.get('name', config_path.stem)

    # Load alpha model
    if 'alpha_model' not in config:
        raise ValueError(f"Config must have 'alpha_model' section: {config_path}")

    alpha_model = create_alpha_model(config['alpha_model'])
    logger.info(f"Created alpha model: {type(alpha_model).__name__}")

    # Load universe
    if 'universe' in config:
        universe_path = base_dir / config['universe']
        universe_tickers = load_universe_config(universe_path)
        logger.info(f"Loaded universe: {len(universe_tickers)} tickers")
    else:
        raise ValueError(f"Config must have 'universe' field: {config_path}")

    # Load schedule (optional, defaults to monthly)
    if 'schedule' in config:
        schedule_path = base_dir / config['schedule']
        rebalance_frequency = load_schedule_config(schedule_path)
    else:
        rebalance_frequency = 'monthly'
        logger.info("No schedule specified, defaulting to monthly rebalance")

    # Load cost model (optional, defaults to 10bps)
    if 'cost_model' in config:
        cost_path = base_dir / config['cost_model']
        cost_model = load_cost_config(cost_path)
    else:
        cost_model = FlatTransactionCost(cost_bps=10.0)
        logger.info("No cost model specified, defaulting to 10bps flat cost")

    # Load portfolio construction
    if 'portfolio_construction' in config:
        portfolio = create_portfolio_construction(config['portfolio_construction'])
    else:
        # Default to top 5 equal weight
        portfolio = EqualWeightTopN(top_n=5)
        logger.info("No portfolio construction specified, defaulting to equal-weight top 5")

    # Extract description
    description = config.get('description')

    strategy_config = StrategyConfig(
        name=name,
        alpha_model=alpha_model,
        portfolio_construction=portfolio,
        cost_model=cost_model,
        universe_tickers=universe_tickers,
        rebalance_frequency=rebalance_frequency,
        description=description,
        raw_config=config
    )

    logger.info(f"Successfully loaded strategy: {strategy_config}")

    return strategy_config


def load_multiple_configs(config_paths: list[Path | str], base_dir: Optional[Path] = None) -> list[StrategyConfig]:
    """Load multiple strategy configs.

    Args:
        config_paths: List of paths to strategy config files
        base_dir: Base directory for resolving relative paths

    Returns:
        List of StrategyConfig objects

    Raises:
        ValueError: If any config fails to load
    """
    configs = []
    failed = []

    for path in config_paths:
        try:
            config = load_strategy_config(path, base_dir)
            configs.append(config)
        except Exception as e:
            logger.error(f"Failed to load config {path}: {e}")
            failed.append((path, str(e)))

    if failed:
        error_msg = "\n".join([f"  {path}: {err}" for path, err in failed])
        raise ValueError(f"Failed to load {len(failed)} config(s):\n{error_msg}")

    return configs
