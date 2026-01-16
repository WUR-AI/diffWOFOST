"""Documentation about diffwofost."""

import logging
from diffwofost.physical_models import utils
from diffwofost.physical_models.crop import assimilation
from diffwofost.physical_models.crop import leaf_dynamics
from diffwofost.physical_models.crop import partitioning
from diffwofost.physical_models.crop import phenology
from diffwofost.physical_models.crop import root_dynamics

logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = ""
__email__ = ""
__version__ = "0.3.0"

__all__ = [
    "leaf_dynamics",
    "root_dynamics",
    "phenology",
    "assimilation",
    "partitioning",
    "utils",
]
