"""Water balance modules.

This module currently implements only `WaterbalancePP` (potential production).

Important: The implementation is tensor-compatible (supports batched crop
parameters) by using diffWOFOST's Tensor templates.
"""

import torch
from pcse.base import SimulationObject
from diffwofost.physical_models.base import TensorParamTemplate
from diffwofost.physical_models.base import TensorRatesTemplate
from diffwofost.physical_models.base import TensorStatesTemplate
from diffwofost.physical_models.traitlets import Tensor


class WaterbalancePP(SimulationObject):
    """Fake waterbalance for simulation under potential production.

    Keeps the soil moisture content at field capacity and only accumulates crop transpiration
    and soil evaporation rates through the course of the simulation
    """

    # Counter for Days-Dince-Last-Rain
    DSLR = Tensor(1.0)
    # rainfall rate of previous day
    RAINold = Tensor(0.0)

    class Parameters(TensorParamTemplate):
        SMFCF = Tensor(-99.0)

    class StateVariables(TensorStatesTemplate):
        SM = Tensor(-99.0)
        WTRAT = Tensor(-99.0)
        EVST = Tensor(-99.0)

    class RateVariables(TensorRatesTemplate):
        EVS = Tensor(0.0)
        WTRA = Tensor(0.0)

    def initialize(self, day, kiosk, parvalues, shape: tuple | torch.Size | None = None):
        """Initialize the potential-production waterbalance.

        :param day: start date of the simulation
        :param kiosk: variable kiosk of this PCSE  instance
        :param parvalues: ParameterProvider object containing all parameters

        This waterbalance keeps the soil moisture always at field capacity. Therefore
        `WaterbalancePP` has only one parameter (`SMFCF`: the field capacity of the
        soil) and one state variable (`SM`: the volumetric soil moisture).
        """
        self.params = self.Parameters(parvalues, shape=shape)
        self.rates = self.RateVariables(kiosk, publish="EVS", shape=shape)
        self.states = self.StateVariables(
            kiosk,
            SM=self.params.SMFCF,
            publish="SM",
            EVST=0.0,
            WTRAT=0.0,
            shape=shape,
        )

    def calc_rates(self, day, drv):
        """Calculate soil evaporation and transpiration rates."""
        r = self.rates
        # Transpiration and maximum soil and surface water evaporation rates
        # are calculated by the crop Evapotranspiration module.
        # However, if the crop is not yet emerged then set TRA=0 and use
        # the potential soil/water evaporation rates directly because there is
        # no shading by the canopy.
        if "TRA" not in self.kiosk:
            r.WTRA = torch.zeros_like(torch.as_tensor(drv.ES0))
            EVSMX = torch.as_tensor(drv.ES0)
        else:
            r.WTRA = self.kiosk["TRA"]
            EVSMX = torch.as_tensor(self.kiosk["EVSMX"])

        # Actual evaporation rates

        rain_ge_1 = torch.as_tensor(self.RAINold) >= 1.0

        # If rainfall amount >= 1cm on previous day assume maximum soil evaporation.
        # Else soil evaporation is a function days-since-last-rain (DSLR).
        dslr_inc = torch.as_tensor(self.DSLR) + 1.0
        evsmxt = EVSMX * (torch.sqrt(dslr_inc) - torch.sqrt(dslr_inc - 1.0))
        evs_else = torch.minimum(EVSMX, evsmxt + torch.as_tensor(self.RAINold))

        r.EVS = torch.where(rain_ge_1, EVSMX, evs_else)
        self.DSLR = torch.where(rain_ge_1, torch.ones_like(dslr_inc), dslr_inc)

        # Hold rainfall amount to keep track of soil surface wetness and reset self.DSLR if needed
        self.RAINold = torch.as_tensor(drv.RAIN)

    def integrate(self, day, delt=1.0):
        """Integrate state variables over one time step."""
        # Keep soil moisture on field capacity
        self.states.SM = self.params.SMFCF

        # Accumulated transpiration and soil evaporation amounts
        # Avoid in-place updates because TensorStatesTemplate may broadcast
        # state tensors using `expand()`, which forbids in-place writes.
        self.states.EVST = self.states.EVST + (self.rates.EVS * delt)
        self.states.WTRAT = self.states.WTRAT + (self.rates.WTRA * delt)
