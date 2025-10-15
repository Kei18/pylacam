import sys

from loguru import logger

from .lacam import LaCAM
from .mapf_utils import (
    get_grid,
    get_scenario,
    get_sum_of_loss,
    is_valid_mapf_solution,
    save_configs_for_visualizer,
    validate_mapf_solution,
)

# set logger
logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>",
)

__all__ = [
    "get_grid",
    "get_scenario",
    "is_valid_mapf_solution",
    "save_configs_for_visualizer",
    "validate_mapf_solution",
    "get_sum_of_loss",
    "LaCAM",
]
