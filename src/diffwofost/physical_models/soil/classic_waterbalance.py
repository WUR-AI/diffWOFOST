"""Water balance modules.

This module implements two water balance variants:

- ``WaterbalancePP``: fake water balance for potential production (soil moisture
  fixed at field capacity).
- ``WaterbalanceFD``: freely-draining water balance for water-limited production.

Both implementations are tensor-compatible (support batched crop parameters) by
using diffWOFOST's Tensor templates, and are therefore fully differentiable with
respect to their soil/site parameters.
"""

import torch
from pcse import exceptions as exc
from pcse import signals
from pcse.base import SimulationObject
from diffwofost.physical_models.base import TensorParamTemplate
from diffwofost.physical_models.base import TensorRatesTemplate
from diffwofost.physical_models.base import TensorStatesTemplate
from diffwofost.physical_models.config import ComputeConfig
from diffwofost.physical_models.kernels import kernel
from diffwofost.physical_models.traitlets import Tensor
from diffwofost.physical_models.utils import Afgen


class WaterbalancePP(SimulationObject):
    """Fake waterbalance for simulation under potential production.

    Keeps the soil moisture content at field capacity and only accumulates
    crop transpiration and soil evaporation rates through the course of the
    simulation.
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

        Args:
            day: start date of the simulation
            kiosk: variable kiosk of this PCSE instance
            parvalues: ParameterProvider object containing all parameters
            shape: optional shape for state and rate tensors (default None for scalar)

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
            r.WTRA = torch.zeros_like(drv["ES0"])
            EVSMX = drv["ES0"]
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
        self.RAINold = drv["RAIN"]

    def integrate(self, day, delt=1.0):
        """Integrate state variables over one time step."""
        # Keep soil moisture on field capacity
        self.states.SM = self.params.SMFCF

        # Accumulated transpiration and soil evaporation amounts
        # Avoid in-place updates because TensorStatesTemplate may broadcast
        # state tensors using `expand()`, which forbids in-place writes.
        self.states.EVST = self.states.EVST + (self.rates.EVS * delt)
        self.states.WTRAT = self.states.WTRAT + (self.rates.WTRA * delt)


class WaterbalanceFD(SimulationObject):
    """Waterbalance for freely draining soils under water-limited production.

    The purpose of the soil water balance calculations is to estimate the
    daily value of the soil moisture content. The soil moisture content
    influences soil moisture uptake and crop transpiration.

    The dynamic calculations are carried out in two sections, one for the
    calculation of rates of change per timestep (= 1 day) and one for the
    calculation of summation variables and state variables. The water balance
    is driven by rainfall, possibly buffered as surface storage, and
    evapotranspiration. The processes considered are infiltration, soil water
    retention, percolation (here conceived as downward water flow from rooted
    zone to second layer), and the loss of water beyond the maximum root zone.

    The textural profile of the soil is conceived as homogeneous. Initially the
    soil profile consists of two layers, the actually rooted soil and the soil
    immediately below the rooted zone until the maximum rooting depth is reached
    by roots(soil and crop dependent). The extension of the root zone from the
    initial rooting depth to maximum rooting depth is described in Root_Dynamics
    class. From the moment that the maximum rooting depth is reached the soil
    profile may be described as a one layer system depending if the roots are
    able to penetrate the entire profile. If not a non-rooted part remains
    at the bottom of the profile.

    The class WaterbalanceFD is derived from WATFD.FOR in WOFOST7.1 with the
    exception that the depth of the soil is now completely determined by the
    maximum soil depth (RDMSOL) and not by the minimum of soil depth and crop
    maximum rooting depth (RDMCR).

    **Simulation parameters:**

    | Name   | Description                                                                     | Type | Unit      |
    |--------|---------------------------------------------------------------------------------|------|-----------|
    | SMFCF  | Field capacity of the soil                                                      | SSo  | -         |
    | SM0    | Porosity of the soil                                                            | SSo  | -         |
    | SMW    | Wilting point of the soil                                                       | SSo  | -         |
    | CRAIRC | Soil critical air content (waterlogging)                                        | SSo  | -         |
    | SOPE   | Maximum percolation rate root zone                                              | SSo  | cm day⁻¹  |
    | KSUB   | Maximum percolation rate subsoil                                                | SSo  | cm day⁻¹  |
    | RDMSOL | Soil rootable depth                                                             | SSo  | cm        |
    | IFUNRN | Indicates whether non-infiltrating fraction of rain is a function of storm size (1) or not (0) | SSi  | - |
    | SSMAX  | Maximum surface storage                                                         | SSi  | cm        |
    | SSI    | Initial surface storage                                                         | SSi  | cm        |
    | WAV    | Initial amount of water in total soil profile                                   | SSi  | cm        |
    | NOTINF | Maximum fraction of rain not infiltrating into the soil                         | SSi  | -         |
    | SMLIM  | Initial maximum moisture content in initial rooting depth zone                  | SSi  | -         |

    **State variables:**

    | Name   | Description                                                                     | Pbl  | Unit      |
    |--------|---------------------------------------------------------------------------------|------|-----------|
    | SM     | Volumetric moisture content in root zone                                        | Y    | -         |
    | SS     | Surface storage (layer of water on surface)                                     | N    | cm        |
    | SSI    | Initial surface storage                                                         | N    | cm        |
    | W      | Amount of water in root zone                                                    | N    | cm        |
    | WI     | Initial amount of water in the root zone                                        | N    | cm        |
    | WLOW   | Amount of water in the subsoil between current rooting depth and maximum rootable depth | N | cm |
    | WLOWI  | Initial amount of water in the subsoil                                          | N    | cm        |
    | WWLOW  | Total amount of water in the soil profile; WWLOW = WLOW + W                     | N    | cm        |
    | WTRAT  | Total water lost as transpiration from the water balance; can differ from CTRAT which only counts transpiration for a crop cycle | N | cm |
    | EVST   | Total evaporation from the soil surface                                         | N    | cm        |
    | EVWT   | Total evaporation from a water surface                                          | N    | cm        |
    | TSR    | Total surface runoff                                                            | N    | cm        |
    | RAINT  | Total amount of rainfall (effective + non-effective)                            | N    | cm        |
    | WDRT   | Amount of water added to root zone by increase of root growth                   | N    | cm        |
    | TOTINF | Total amount of infiltration                                                    | N    | cm        |
    | TOTIRR | Total amount of effective irrigation                                            | N    | cm        |
    | PERCT  | Total amount of water percolating from rooted zone to subsoil                   | N    | cm        |
    | LOSST  | Total amount of water lost to deeper soil                                       | N    | cm        |
    | DSOS   | Days since oxygen stress, accumulating consecutive days of oxygen stress         | Y    | -         |
    | WBALRT | Checksum for root zone water balance; computed in `finalize()`, abs(WBALRT) > 0.0001 raises a WaterBalanceError | N | cm |
    | WBALTT | Checksum for total water balance; computed in `finalize()`, abs(WBALTT) > 0.0001 raises a WaterBalanceError | N | cm |

    **Rate variables:**

    | Name       | Description                                                                    | Pbl  | Unit      |
    |------------|--------------------------------------------------------------------------------|------|-----------|
    | EVS        | Actual evaporation rate from soil                                              | N    | cm day⁻¹  |
    | EVW        | Actual evaporation rate from water surface                                     | N    | cm day⁻¹  |
    | WTRA       | Actual transpiration rate from plant canopy, directly derived from `TRA` in the evapotranspiration module | N | cm day⁻¹ |
    | RAIN_INF   | Infiltrating rainfall rate for current day                                     | N    | cm day⁻¹  |
    | RAIN_NOTINF| Non-infiltrating rainfall rate for current day                                 | N    | cm day⁻¹  |
    | RIN        | Infiltration rate for current day                                              | N    | cm day⁻¹  |
    | RIRR       | Effective irrigation rate for current day, computed as irrigation amount times efficiency | N | cm day⁻¹ |
    | PERC       | Percolation rate to non-rooted zone                                            | N    | cm day⁻¹  |
    | LOSS       | Rate of water loss to deeper soil                                              | N    | cm day⁻¹  |
    | DW         | Change in amount of water in rooted zone due to infiltration, transpiration, and evaporation | N | cm day⁻¹ |
    | DWLOW      | Change in amount of water in subsoil                                           | N    | cm day⁻¹  |
    | DTSR       | Change in surface runoff                                                       | N    | cm day⁻¹  |
    | DSS        | Change in surface storage                                                      | N    | cm day⁻¹  |

    **External dependencies:**

    | Name   | Description                                             | Provided by         | Unit      |
    |--------|---------------------------------------------------------|---------------------|-----------|
    | TRA    | Crop transpiration rate                                 | Evapotranspiration  | cm day⁻¹  |
    | EVSMX  | Maximum evaporation rate from a soil surface below the crop canopy | Evapotranspiration  | cm day⁻¹  |
    | EVWMX  | Maximum evaporation rate from a water surface below the crop canopy | Evapotranspiration  | cm day⁻¹  |
    | RD     | Rooting depth                                           | Root_dynamics       | cm        |

    **Outputs**

    | Name   | Description                              | Pbl  | Unit      |
    |--------|------------------------------------------|------|-----------|
    | SM     | Volumetric soil moisture in root zone    | Y    | -         |
    | EVS    | Actual evaporation rate from soil        | Y    | cm day⁻¹  |
    | DSOS   | Days since oxygen stress                 | Y    | -         |

    **Gradient mapping (which parameters have a gradient):**

    | Output | Parameters influencing it                |
    |--------|------------------------------------------|
    | SM     | SMFCF, SMW, SM0                          |
    | EVS    | SMFCF, SMW, SM0                          |
    | DSOS   | None; thresholded diagnostic output      |

    **Exceptions raised:**

    A WaterbalanceError is raised when the waterbalance is not closing at the
    end of the simulation cycle (e.g water has "leaked" away).
    """  # noqa: E501

    # Previous and maximum rooting depth (cm)
    RDold = Tensor(-99.0)
    RDM = Tensor(-99.0)
    # Counter for Days-Since-Last-Rain
    DSLR = Tensor(-99.0)
    # Infiltration rate of previous day (cm/day)
    RINold = Tensor(0.0)
    # Fraction of non-infiltrating rainfall as function of storm size (Afgen)
    NINFTB = None
    # Flag indicating crop present or not (plain Python bool, not differentiable)
    in_crop_cycle = False
    # Flag indicating that a crop was started or finished and therefore the depth
    # of the root zone may have changed, required a redistribution of water
    # between the root zone and the lower zone
    rooted_layer_needs_reset = False
    # Placeholder for irrigation rate (cm/day)
    _RIRR = Tensor(0.0)
    # Default depth of the upper rooted layer when no crop RD is available (cm)
    DEFAULT_RD = Tensor(10.0)
    # Sum of the increments on W due to forced state updates
    _increments_w = None

    class Parameters(TensorParamTemplate):
        # Soil parameters
        SMFCF = Tensor(-99.0)
        SM0 = Tensor(-99.0)
        SMW = Tensor(-99.0)
        CRAIRC = Tensor(-99.0)
        SOPE = Tensor(-99.0)
        KSUB = Tensor(-99.0)
        RDMSOL = Tensor(-99.0)
        # Site parameters
        IFUNRN = Tensor(-99.0)
        SSMAX = Tensor(-99.0)
        SSI = Tensor(-99.0)
        WAV = Tensor(-99.0)
        NOTINF = Tensor(-99.0)

    class StateVariables(TensorStatesTemplate):
        SM = Tensor(-99.0)
        SS = Tensor(-99.0)
        SSI = Tensor(-99.0)
        W = Tensor(-99.0)
        WI = Tensor(-99.0)
        WLOW = Tensor(-99.0)
        WLOWI = Tensor(-99.0)
        WWLOW = Tensor(-99.0)
        # Summation variables
        WTRAT = Tensor(-99.0)
        EVST = Tensor(-99.0)
        EVWT = Tensor(-99.0)
        TSR = Tensor(-99.0)
        RAINT = Tensor(-99.0)
        WDRT = Tensor(-99.0)
        TOTINF = Tensor(-99.0)
        TOTIRR = Tensor(-99.0)
        PERCT = Tensor(-99.0)
        LOSST = Tensor(-99.0)
        # Checksums for rootzone (RT) and total system (TT)
        WBALRT = Tensor(-999.0)
        WBALTT = Tensor(-999.0)
        DSOS = Tensor(0.0)

    class RateVariables(TensorRatesTemplate):
        EVS = Tensor(0.0)
        EVW = Tensor(0.0)
        WTRA = Tensor(0.0)
        RIN = Tensor(0.0)
        RIRR = Tensor(0.0)
        PERC = Tensor(0.0)
        LOSS = Tensor(0.0)
        DW = Tensor(0.0)
        DWLOW = Tensor(0.0)
        DTSR = Tensor(0.0)
        DSS = Tensor(0.0)
        DRAINT = Tensor(0.0)

    def initialize(self, day, kiosk, parvalues, shape=None):
        """Initialize the freely-draining water balance.

        Args:
            day: Start date of the simulation.
            kiosk: Variable kiosk used to read and publish crop state.
            parvalues: Parameter provider containing the physical-model
                parameters for the soil.
            shape: Target tensor shape for state and rate variables.
        """
        dtype = ComputeConfig.get_dtype()
        device = ComputeConfig.get_device()

        self.params = self.Parameters(parvalues, shape=shape)

        # Check validity of SMLIM (site parameter, not stored in self.params)
        SMLIM_raw = torch.as_tensor(parvalues["SMLIM"], dtype=dtype, device=device)
        SMLIM = torch.maximum(self.params.SMW, torch.minimum(self.params.SM0, SMLIM_raw))
        if torch.any(SMLIM != SMLIM_raw):
            self.logger.warning("SMLIM not in valid range, adjusted.")

        # Default rooting depth: 10 cm; derive maximum rootable depth
        self.RDold = torch.as_tensor(self.DEFAULT_RD, dtype=dtype, device=device)
        self.RDM = torch.maximum(self.RDold, self.params.RDMSOL)

        # Initial soil moisture content in rooted zone, clamped to [SMW, SMLIM]
        SM = torch.clamp(
            self.params.SMW + self.params.WAV / self.RDold,
            min=self.params.SMW,
            max=SMLIM,
        )
        W = SM * self.RDold

        # Initial water in subsoil (below rooted zone to maximum rootable depth)
        WLOW = torch.clamp(
            self.params.WAV + self.RDM * self.params.SMW - W,
            min=torch.zeros_like(W),
            max=self.params.SM0 * (self.RDM - self.RDold),
        )

        # Days since last rain: 1 if SM is above halfway between SMW and SMFCF, else 5
        halfway = self.params.SMW + 0.5 * (self.params.SMFCF - self.params.SMW)
        self.DSLR = torch.where(SM >= halfway, torch.ones_like(SM), torch.full_like(SM, 5.0))

        # Initialize helper variables
        self.RINold = torch.zeros_like(SM)
        self.in_crop_cycle = False
        self.rooted_layer_needs_reset = False
        self.NINFTB = Afgen([0.0, 0.0, 0.5, 0.0, 1.5, 1.0])
        self._increments_w = 0.0

        # Initialize state and rate variables
        self.states = self.StateVariables(
            kiosk,
            publish=["SM", "DSOS"],
            SM=SM,
            SS=self.params.SSI,
            SSI=self.params.SSI,
            W=W,
            WI=W,
            WLOW=WLOW,
            WLOWI=WLOW,
            WWLOW=(W + WLOW),  # Total water in soil column
            WTRAT=0.0,
            EVST=0.0,
            EVWT=0.0,
            TSR=0.0,
            RAINT=0.0,
            WDRT=0.0,
            TOTINF=0.0,
            TOTIRR=0.0,
            DSOS=0.0,
            PERCT=0.0,
            LOSST=0.0,
            WBALRT=-999.0,
            WBALTT=-999.0,
            shape=shape,
        )
        self.rates = self.RateVariables(kiosk, publish="EVS", shape=shape)

        # Connect signals
        self._connect_signal(self._on_CROP_START, signals.crop_start)
        self._connect_signal(self._on_CROP_FINISH, signals.crop_finish)
        self._connect_signal(self._on_IRRIGATE, signals.irrigate)

    def calc_rates(self, day, drv):
        """Calculate the rates of change for all water balance components.

        Args:
            day (datetime.date): The current date of the simulation.
            drv (dict): A container holding weather data elements as key/value. The values are
                arrays or scalars.

        """
        s = self.states
        p = self.params
        r = self.rates
        k = self.kiosk
        dtype = ComputeConfig.get_dtype()
        device = ComputeConfig.get_device()

        # Rate of irrigation (RIRR)
        r.RIRR = self._RIRR
        self._RIRR = torch.zeros_like(self._RIRR)

        # Transpiration and maximum evaporation rates from crop module.
        # Before emergence there is no canopy shading yet, so the water balance
        # must fall back to the weather-driven soil and surface evaporation.
        crop = None
        if "TRA" in k:
            crop = tuple(
                torch.as_tensor(k[name], dtype=dtype, device=device)
                for name in ("TRA", "EVWMX", "EVSMX")
            )
        dvs = torch.as_tensor(k["DVS"], dtype=dtype, device=device) if "DVS" in k else None

        (
            r.WTRA,
            r.EVW,
            r.EVS,
            self.DSLR,
            r.RIN,
            r.PERC,
            r.LOSS,
            r.DW,
            r.DWLOW,
            r.DSS,
            r.DTSR,
        ) = _waterbalance_rates(
            crop,
            dvs,
            drv["E0"],
            drv["ES0"],
            drv["RAIN"],
            r.RIRR,
            s.SS,
            s.SM,
            s.W,
            s.WLOW,
            self.DSLR,
            self.RINold,
            self._determine_rooting_depth(),
            self.RDM,
            p.NOTINF,
            p.IFUNRN,
            p.SOPE,
            p.SMFCF,
            p.SM0,
            p.KSUB,
            p.SSMAX,
            self.NINFTB,
        )
        self.RINold = r.RIN
        # incoming rainfall rate
        r.DRAINT = drv["RAIN"]

    def integrate(self, day, delt=1.0):
        """Integrate state variables over one time step.

        Args:
            day (datetime.date): The current date of the simulation.
            delt (float, optional): The time step for integration. Defaults to 1.0.
        """
        s = self.states
        p = self.params
        r = self.rates

        RD = self._determine_rooting_depth()
        (
            s.WTRAT,
            s.EVWT,
            s.EVST,
            s.RAINT,
            s.TOTINF,
            s.TOTIRR,
            s.SS,
            s.TSR,
            s.W,
            s.PERCT,
            s.LOSST,
            s.WLOW,
            s.WWLOW,
            s.WDRT,
            s.SM,
            s.DSOS,
        ) = _waterbalance_integrate(
            s.WTRAT, s.EVWT, s.EVST, s.RAINT, s.TOTINF, s.TOTIRR, s.SS, s.TSR, s.W, s.PERCT,
            s.LOSST, s.WLOW, s.WDRT, s.DSOS,
            r.WTRA, r.EVW, r.EVS, r.DRAINT, r.RIN, r.RIRR, r.DSS, r.DTSR, r.DW, r.PERC, r.LOSS,
            r.DWLOW,
            RD, self.RDold, p.RDMSOL, p.SM0, p.CRAIRC, delt,
        )  # fmt: skip

        # save rooting depth
        self.RDold = RD

    def finalize(self, day):
        """Calculate and check water balance checksums at end of simulation."""
        s = self.states

        wbal_increments = self._increments_w

        # Checksums for rootzone (RT) and whole system (TT)
        s.WBALRT = s.TOTINF + s.WI + s.WDRT - s.EVST - s.WTRAT - s.PERCT - s.W + wbal_increments
        s.WBALTT = (
            s.SSI
            + s.RAINT
            + s.TOTIRR
            + s.WI
            - s.W
            + wbal_increments
            + s.WLOWI
            - s.WLOW
            - s.WTRAT
            - s.EVWT
            - s.EVST
            - s.TSR
            - s.LOSST
            - s.SS
        )

        if torch.any(torch.abs(s.WBALRT) > 0.0001):
            msg = "Water balance for root zone does not close."
            raise exc.WaterBalanceError(msg)

        if torch.any(torch.abs(s.WBALTT) > 0.0001):
            total_in = (s.WI + s.WLOWI + s.SSI + s.TOTIRR + s.RAINT).tolist()
            total_out = (s.W + s.WLOW + s.SS + s.EVWT + s.EVST + s.WTRAT + s.TSR + s.LOSST).tolist()
            msg = "Water balance for complete soil profile does not close.\n"
            msg += f"Total INIT + IN:   {total_in}\n"
            msg += f"Total FINAL + OUT: {total_out}"
            raise exc.WaterBalanceError(msg)

        SimulationObject.finalize(self, day)

    def _determine_rooting_depth(self):
        """Return current rooting depth as a tensor.

        Returns the crop rooting depth from the kiosk when a crop is present,
        otherwise holds the upper layer at the default 10 cm depth.
        """
        if "RD" in self.kiosk:
            return self.kiosk["RD"]
        dtype = ComputeConfig.get_dtype()
        device = ComputeConfig.get_device()
        return torch.as_tensor(self.DEFAULT_RD, dtype=dtype, device=device)

    def _on_CROP_START(self):
        self.in_crop_cycle = True
        self.rooted_layer_needs_reset = True

    def _on_CROP_FINISH(self):
        self.in_crop_cycle = False
        self.rooted_layer_needs_reset = True

    def _on_IRRIGATE(self, amount, efficiency):
        self._RIRR = torch.as_tensor(
            amount * efficiency,
            dtype=ComputeConfig.get_dtype(),
            device=ComputeConfig.get_device(),
        )

    def _set_variable_SM(self, nSM):
        """Force soil moisture to ``nSM`` and record the water increment.

        Updates W and WWLOW consistently, and adds the increment to
        ``_increments_w`` so the water balance checksum in ``finalize()``
        still closes.
        """
        s = self.states
        oSM = s.SM
        oW = s.W
        nW = nSM / oSM * s.W
        s.W = nW
        s.SM = nSM
        s.WWLOW = s.WLOW + s.W
        self._increments_w = self._increments_w + (nW - oW)
        return {"W": nW - oW, "SM": nSM - oSM}


@kernel
def _waterbalance_rates(
    crop, dvs, E0, ES0, RAIN, RIRR, SS, SM, W, WLOW, DSLR, RINold, RD, RDM,
    NOTINF, IFUNRN, SOPE, SMFCF, SM0, KSUB, SSMAX, NINFTB,
):  # fmt: skip
    """Rates of the freely-draining water balance, see `WaterbalanceFD.calc_rates`.

    `crop` holds the crop's (TRA, EVWMX, EVSMX), or is None without a crop.
    """
    # Transpiration and maximum evaporation rates from crop module.
    # Before emergence there is no canopy shading yet, so the water balance
    # must fall back to the weather-driven soil and surface evaporation.
    weather_wtra = torch.zeros_like(ES0)
    if crop is None:
        WTRA, EVWMX, EVSMX = weather_wtra, E0, ES0
    elif dvs is not None:
        emerged_mask = dvs >= 0.0
        WTRA = torch.where(emerged_mask, crop[0], weather_wtra)
        EVWMX = torch.where(emerged_mask, crop[1], E0)
        EVSMX = torch.where(emerged_mask, crop[2], ES0)
    else:
        WTRA, EVWMX, EVSMX = crop

    # Actual evaporation rates
    # If SS > 1 cm: evaporate from surface water; else from soil
    ss_gt_1 = SS > 1.0
    rin_ge_1 = RINold >= 1.0
    dslr_inc = DSLR + 1.0
    # Soil evaporation as function of days-since-last-rain (DSLR)
    EVSMXT = EVSMX * (torch.sqrt(dslr_inc) - torch.sqrt(DSLR))
    evs_dslr = torch.minimum(EVSMX, EVSMXT + RINold)
    EVS_no_ss = torch.where(rin_ge_1, EVSMX, evs_dslr)

    EVW = torch.where(ss_gt_1, EVWMX, torch.zeros_like(EVWMX))
    EVS = torch.where(ss_gt_1, torch.zeros_like(EVS_no_ss), EVS_no_ss)

    # Update DSLR:
    #   - SS > 1: unchanged (no soil evaporation)
    #   - SS <= 1 and RINold >= 1: reset to 1
    #   - SS <= 1 and RINold < 1: increment by 1
    DSLR = torch.where(
        ss_gt_1,
        DSLR,
        torch.where(rin_ge_1, torch.ones_like(DSLR), dslr_inc),
    )

    # Potentially infiltrating rainfall
    RINPRE_fixed = (1.0 - NOTINF) * RAIN
    RINPRE_storm = (1.0 - NOTINF * NINFTB(RAIN)) * RAIN
    # IFUNRN: 0 = fixed non-infiltrating fraction, 1 = function of storm size
    RINPRE = torch.where(IFUNRN == 0, RINPRE_fixed, RINPRE_storm)

    # Second stage: add surface storage and irrigation
    RINPRE = RINPRE + RIRR + SS
    # With surface storage, infiltration is limited by SOPE
    AVAIL = RINPRE - EVW
    RINPRE = torch.where(SS > 0.1, torch.minimum(SOPE, AVAIL), RINPRE)

    # Equilibrium water in rooted zone
    WE = SMFCF * RD
    # Percolation: excess moisture above field capacity, capped at SOPE
    PERC1 = torch.maximum(
        torch.zeros_like(W),
        torch.minimum(SOPE, (W - WE) - WTRA - EVS),
    )

    # Loss from lower zone
    WELOW = SMFCF * (RDM - RD)
    LOSS = torch.maximum(
        torch.zeros_like(WLOW),
        torch.minimum(KSUB, WLOW - WELOW + PERC1),
    )

    # Percolation capped by subsoil uptake capacity
    PERC2 = (RDM - RD) * SM0 - WLOW + LOSS
    PERC = torch.minimum(PERC1, PERC2)

    # Adjusted infiltration rate
    RIN = torch.minimum(RINPRE, (SM0 - SM) * RD + WTRA + EVS + PERC)

    # Rates of change in water amounts
    DW = RIN - WTRA - EVS - PERC
    DWLOW = PERC - LOSS

    # Prevent W from going negative by reducing EVS
    Wtmp = W + DW
    under_zero = Wtmp < 0.0
    EVS = torch.where(under_zero, torch.maximum(torch.zeros_like(EVS), EVS + Wtmp), EVS)
    DW = torch.where(under_zero, -W, DW)

    # Surface storage and runoff
    SStmp = RAIN + RIRR - EVW - RIN
    DSS = torch.minimum(SStmp, SSMAX - SS)
    DTSR = SStmp - DSS
    return WTRA, EVW, EVS, DSLR, RIN, PERC, LOSS, DW, DWLOW, DSS, DTSR


@kernel
def _waterbalance_integrate(
    WTRAT, EVWT, EVST, RAINT, TOTINF, TOTIRR, SS, TSR, W, PERCT, LOSST, WLOW, WDRT, DSOS,
    WTRA, EVW, EVS, DRAINT, RIN, RIRR, DSS, DTSR, DW, PERC, LOSS, DWLOW,
    RD, RDold, RDMSOL, SM0, CRAIRC, delt,
):  # fmt: skip
    """State integration of the freely-draining water balance, see `WaterbalanceFD.integrate`."""
    # INTEGRALS OF THE WATERBALANCE: SUMMATIONS AND STATE VARIABLES

    # total transpiration
    WTRAT = WTRAT + WTRA * delt

    # total evaporation from surface water layer and/or soil
    EVWT = EVWT + EVW * delt
    EVST = EVST + EVS * delt

    # totals for rainfall, irrigation and infiltration
    RAINT = RAINT + DRAINT * delt
    TOTINF = TOTINF + RIN * delt
    TOTIRR = TOTIRR + RIRR * delt

    # Update surface storage and total surface runoff (TSR)
    SS = SS + DSS * delt
    TSR = TSR + DTSR * delt

    # amount of water in rooted zone
    W = W + DW * delt

    # total percolation and loss of water by deep leaching
    PERCT = PERCT + PERC * delt
    LOSST = LOSST + LOSS * delt

    # amount of water in unrooted, lower part of rootable zone
    WLOW = WLOW + DWLOW * delt
    # total amount of water in the whole rootable zone
    WWLOW = W + WLOW

    # CHANGE OF ROOTZONE SUBSYSTEM BOUNDARY
    # Redistribution of water is needed when roots grow during the growing season
    # and when the crop is finished and the root zone shifts back from the crop rooted
    # depth to the default depth of the upper (rooted) layer of the water balance.
    # Or when the initial rooting depth of a crop is different from the default one used
    # by the water balance module (10 cm)
    RDchange = RD - RDold

    # Safe denominator for downward-growth case
    rdm_minus_rdold = RDMSOL - RDold
    safe_rdm_delta = torch.where(
        rdm_minus_rdold.abs() > 1e-10,
        rdm_minus_rdold,
        torch.ones_like(rdm_minus_rdold),
    )
    # Roots grow down: pull water up from subsoil
    WDR_down = torch.minimum(WLOW, WLOW * RDchange / safe_rdm_delta)

    # Roots move up (or negligible change): push water down to subsoil
    safe_rdold = torch.where(
        RDold.abs() > 1e-10,
        RDold,
        torch.ones_like(RDold),
    )
    WDR_other = W * RDchange / safe_rdold

    WDR = torch.where(RDchange > 0.001, WDR_down, WDR_other)

    apply = WDR.abs() > 0.0
    WLOW = torch.where(apply, WLOW - WDR, WLOW)
    W = torch.where(apply, W + WDR, W)
    WDRT = torch.where(apply, WDRT + WDR, WDRT)

    # mean soil moisture content in rooted zone
    SM = W / RD

    # Accumulate days since oxygen stress
    stress_mask = SM >= (SM0 - CRAIRC)
    DSOS = torch.where(stress_mask, DSOS + 1.0, torch.zeros_like(DSOS))
    return (
        WTRAT, EVWT, EVST, RAINT, TOTINF, TOTIRR, SS, TSR, W, PERCT, LOSST, WLOW, WWLOW, WDRT,
        SM, DSOS,
    )  # fmt: skip
