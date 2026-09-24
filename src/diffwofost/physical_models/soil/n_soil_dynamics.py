"""Soil nitrogen supply for WOFOST 8.1 potential production.

Torch port of ``pcse.soil.n_soil_dynamics.N_PotentialProduction``. Available
nitrogen stays at 100 kg/ha, so crop uptake does not deplete the soil pool.
"""

from pcse.base import SimulationObject
from diffwofost.physical_models.base import TensorStatesTemplate
from diffwofost.physical_models.traitlets import Tensor

# PCSE's potential-production nitrogen supply keeps NAVAIL constant.
# Crop N uptake is therefore not subtracted from this pool.
_NAVAIL_PP = 100.0


class N_PotentialProduction(SimulationObject):
    """Unlimited soil nitrogen for potential-production simulations.

    **Gradient mapping (which parameters have a gradient):**

    | Output | Parameters influencing it |
    |--------|---------------------------|
    | NAVAIL | none                      |

    [!NOTE]
    ``NAVAIL`` stays at 100 kg/ha. Crop uptake reads it and does not change it,
    so the gradient of this pool with respect to uptake is zero.
    """

    class StateVariables(TensorStatesTemplate):
        NAVAIL = Tensor(-99.0)

    def initialize(self, day, kiosk, parvalues, shape=None):
        """Publish a constant available-nitrogen pool of 100 kg/ha."""
        self.states = self.StateVariables(kiosk, publish=["NAVAIL"], NAVAIL=_NAVAIL_PP, shape=shape)

    def calc_rates(self, day, drv):
        """Potential production has no soil-nitrogen rates."""

    def integrate(self, day, delt=1.0):
        """Put the stored pool back into the kiosk after the daily flush.

        ``touch`` reassigns the current tensor. It does not recompute it.
        Crop uptake only reads ``NAVAIL``, so the stored value stays 100 kg/ha.
        """
        self.states.touch()
