"""Wrappers that run a water balance and a nitrogen balance as one soil module.

WOFOST 8.1 configurations point the engine at one of these wrappers. Potential
production pairs the field-capacity water balance with a non-depleting nitrogen
pool. The SNOMIN configuration pairs the layered water balance with SNOMIN.
"""

from pcse.base import SimulationObject
from pcse.traitlets import Instance
from diffwofost.physical_models.soil.classic_waterbalance import WaterbalancePP
from diffwofost.physical_models.soil.multilayer_waterbalance import WaterBalanceLayered
from diffwofost.physical_models.soil.n_soil_dynamics import N_PotentialProduction
from diffwofost.physical_models.soil.snomin import SNOMIN


class BaseSoilWrapper(SimulationObject):
    """Run a water-balance class and a nutrient-balance class on the same day.

    These wrappers add no parameters. Gradients are those of the water-balance
    and nitrogen-balance classes they construct.
    """

    waterbalance_class = None
    nutrientbalance_class = None
    waterbalance = Instance(SimulationObject)
    nutrientbalance = Instance(SimulationObject)

    def initialize(self, day, kiosk, parvalues, shape=None):
        """Initialise water first so SNOMIN can read the soil profile it publishes."""
        if self.waterbalance_class is not None:
            self.waterbalance = self.waterbalance_class(day, kiosk, parvalues, shape=shape)
        if self.nutrientbalance_class is not None:
            self.nutrientbalance = self.nutrientbalance_class(day, kiosk, parvalues, shape=shape)

    def calc_rates(self, day, drv):
        """Calculate water rates before nutrient rates, matching PCSE."""
        if self.waterbalance_class is not None:
            self.waterbalance.calc_rates(day, drv)
        if self.nutrientbalance_class is not None:
            self.nutrientbalance.calc_rates(day, drv)

    def integrate(self, day, delt=1.0):
        """Integrate water, then nutrients."""
        if self.waterbalance_class is not None:
            self.waterbalance.integrate(day, delt)
        if self.nutrientbalance_class is not None:
            self.nutrientbalance.integrate(day, delt)

    def finalize(self, day):
        """Finalize both balances."""
        if self.waterbalance_class is not None:
            self.waterbalance.finalize(day)
        if self.nutrientbalance_class is not None and hasattr(self.nutrientbalance, "finalize"):
            self.nutrientbalance.finalize(day)
        SimulationObject.finalize(self, day)


class SoilModuleWrapper_PP(BaseSoilWrapper):
    """Potential production: field-capacity water and a non-depleting nitrogen pool."""

    waterbalance_class = WaterbalancePP
    nutrientbalance_class = N_PotentialProduction


class SoilModuleWrapper_NWLP_MLWB_SNOMIN(BaseSoilWrapper):
    """Water- and nitrogen-limited production with the layered water balance and SNOMIN."""

    waterbalance_class = WaterBalanceLayered
    nutrientbalance_class = SNOMIN
