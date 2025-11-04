"""pycam: A minimal Python implementation of LaCAM* for Multi-Agent Path Finding.

This package provides an implementation of the LaCAM* algorithm along with
utilities for loading MAPF problem instances, validating solutions, and
computing solution costs.

Example:
    >>> from pycam import LaCAM, get_grid, get_scenario
    >>> grid = get_grid("assets/tunnel.map")
    >>> starts, goals = get_scenario("assets/tunnel.scen", N=4)
    >>> planner = LaCAM()
    >>> solution = planner.solve(
    ...     grid=grid,
    ...     starts=starts,
    ...     goals=goals,
    ...     time_limit_ms=5000,
    ...     verbose=1
    ... )
"""

import sys

from loguru import logger

from .lacam import LaCAM
from .mapf_utils import (
    Config,
    Configs,
    Coord,
    Deadline,
    Grid,
    get_grid,
    get_scenario,
    get_sum_of_loss,
    is_valid_mapf_solution,
    save_configs_for_visualizer,
    validate_mapf_solution,
)

__all__ = [
    # Main algorithm
    "LaCAM",
    # Type aliases for type hints
    "Grid",
    "Coord",
    "Config",
    "Configs",
    "Deadline",
    # Utility functions
    "get_grid",
    "get_scenario",
    "is_valid_mapf_solution",
    "save_configs_for_visualizer",
    "validate_mapf_solution",
    "get_sum_of_loss",
    # Logger configuration
    "configure_logger",
]


def configure_logger(
    level: str = "DEBUG",
    colorize: bool = True,
    format_string: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>",
) -> None:
    """Configure the pycam package logger.

    By default, the logger is configured automatically when the package is imported.
    Call this function to reconfigure the logger with different settings.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        colorize: Whether to colorize log output.
        format_string: Log message format string (Loguru format).

    Example:
        >>> from pycam import configure_logger
        >>> configure_logger(level="INFO", colorize=False)
    """
    logger.remove()
    logger.add(
        sys.stdout,
        colorize=colorize,
        format=format_string,
        level=level,
    )


# Configure logger with default settings on import
# Users can call configure_logger() to customize
configure_logger()
