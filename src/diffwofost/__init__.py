"""Documentation about diffwofost."""

import logging
from diffwofost.physical_models.crop import leaf_dynamics
from diffwofost.physical_models.crop import root_dynamics
from diffwofost.physical_models import utils

logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = ""
__email__ = ""
__version__ = "0.1.1"

__all__ = [
    "leaf_dynamics",
    "root_dynamics",
    "utils",
]