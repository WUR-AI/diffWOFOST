import datetime
import torch
from pcse.base import SimulationObject
from pcse.base.parameter_providers import ParameterProvider
from pcse.base.variablekiosk import VariableKiosk
from pcse.base.weather import WeatherDataContainer
from pcse.decorators import prepare_rates
from pcse.decorators import prepare_states
from pcse.traitlets import Any
from pcse.traitlets import Bool
from pcse.traitlets import Instance
from diffwofost.physical_models.base import TensorParamTemplate
from diffwofost.physical_models.base import TensorRatesTemplate
from diffwofost.physical_models.base import TensorStatesTemplate
from diffwofost.physical_models.config import ComputeConfig
from diffwofost.physical_models.traitlets import Tensor
from diffwofost.physical_models.utils import AfgenTrait
from diffwofost.physical_models.utils import _broadcast_to
from diffwofost.physical_models.utils import _get_drv


def SWEAF(ET0: torch.Tensor, DEPNR: torch.Tensor) -> torch.Tensor:
    """Soil Water Easily Available Fraction (SWEAF).

    The fraction of easily available soil water between field capacity and
    wilting point is a function of the potential evapotranspiration rate (for a
    closed canopy) in cm/day, ET0, and the crop group number, DEPNR (from 1
    (=drought-sensitive) to 5 (=drought-resistent)). The function SWEAF
    describes this relationship given in tabular form by Doorenbos & Kassam
    (1979) and by Van Keulen & Wolf (1986; p.108, table 20)
    http://edepot.wur.nl/168025.

    Args:
        ET0: The evapotranpiration from a reference crop.
        DEPNR: The crop dependency number.
    Returns:
        SWEAF value between 0.10 and 0.95.
    """
    A = 0.76
    B = 1.5
    sweaf = 1.0 / (A + B * ET0) - (5.0 - DEPNR) * 0.10
    correction = (ET0 - 0.6) / (DEPNR * (DEPNR + 3.0))
    # NOTE: PCSE applies `correction` only when `DEPNR < 3` (hard switch), which
    # is non-differentiable at `DEPNR==3` and causes numerical vs autograd
    # gradient mismatches when treating DEPNR as a continuous tensor.
    #
    # To keep regression behaviour intact we preserve exact values at the
    # discrete DEPNR values used in the YAML fixtures (2.0/3.0/3.5/4.5):
    # - DEPNR <= 2: full correction
    # - DEPNR >= 3: no correction
    # and smoothly transition (C1) between 2 and 3 using a cubic smoothstep.
    t = DEPNR - 2.0
    s = 3.0 * t**2 - 2.0 * t**3  # smoothstep on [0,1]
    taper_mid = 1.0 - s
    taper = torch.where(
        DEPNR <= 2.0,
        torch.ones_like(DEPNR),
        torch.where(DEPNR >= 3.0, torch.zeros_like(DEPNR), taper_mid),
    )
    sweaf = sweaf + correction * taper
    return torch.clamp(sweaf, min=0.10, max=0.95)


class EvapotranspirationWrapper(SimulationObject):
    """Selects the evapotranspiration implementation.

    Selection logic:
    - If `soil_profile` is present in parameters: use the layered CO2-aware module.
    - Else if `CO2TRATB` is present: use the non-layered CO2 module.
    - Else: use the non-layered (no CO2) module.
    """

    etmodule = Instance(SimulationObject)

    def initialize(
        self,
        day: datetime.date,
        kiosk: VariableKiosk,
        parvalues: ParameterProvider,
        shape: tuple | None = None,
    ) -> None:
        """Select and initialize the evapotranspiration implementation.

        Chooses between layered CO2-aware, non-layered CO2, or standard evapotranspiration
        based on available parameters.

        Args:
            day (datetime.date): The starting date of the simulation.
            kiosk (VariableKiosk): A container for registering and publishing
                (internal and external) state variables. See PCSE documentation for
                details.
            parvalues (ParameterProvider): A dictionary-like container holding
                all parameter sets (crop, soil, site) as key/value. The values are
                arrays or scalars. See PCSE documentation for details.
            shape (tuple | torch.Size | None): Target shape for the state and rate variables.
        """
        if "soil_profile" in parvalues:
            self.etmodule = EvapotranspirationCO2Layered(day, kiosk, parvalues, shape=shape)
        elif "CO2TRATB" in parvalues:
            self.etmodule = EvapotranspirationCO2(day, kiosk, parvalues, shape=shape)
        else:
            self.etmodule = Evapotranspiration(day, kiosk, parvalues, shape=shape)

    @prepare_rates
    def calc_rates(self, day: datetime.date = None, drv: WeatherDataContainer = None):
        """Delegate rate calculation to the selected evapotranspiration module.

        Args:
            day (datetime.date, optional): The current date of the simulation.
            drv (WeatherDataContainer, optional): A dictionary-like container holding
                weather data elements as key/value. The values are
                arrays or scalars. See PCSE documentation for details.
        """
        return self.etmodule.calc_rates(day, drv)

    def __call__(self, day: datetime.date = None, drv: WeatherDataContainer = None):
        """Callable interface for rate calculation."""
        return self.calc_rates(day, drv)

    @prepare_states
    def integrate(self, day: datetime.date = None, delt=1.0) -> None:
        """Delegate state integration to the selected evapotranspiration module.

        Args:
            day (datetime.date, optional): The current date of the simulation.
            delt (float, optional): The time step for integration. Defaults to 1.0.
        """
        return self.etmodule.integrate(day, delt)


class _BaseEvapotranspiration(SimulationObject):
    """Shared base class for evapotranspiration implementations."""

    params_shape = None

    @property
    def device(self):
        """Get device from ComputeConfig."""
        return getattr(self, "_device", ComputeConfig.get_device())

    @property
    def dtype(self):
        """Get dtype from ComputeConfig."""
        return getattr(self, "_dtype", ComputeConfig.get_dtype())

    class RateVariables(TensorRatesTemplate):
        EVWMX = Tensor(0.0)
        EVSMX = Tensor(0.0)
        TRAMX = Tensor(0.0)
        TRA = Tensor(0.0)
        TRALY = Tensor(0.0)
        IDOS = Bool(False)
        IDWS = Bool(False)
        RFWS = Tensor(0.0)
        RFOS = Tensor(0.0)
        RFTRA = Tensor(0.0)

    class StateVariables(TensorStatesTemplate):
        IDOST = Tensor(-99.0)
        IDWST = Tensor(-99.0)

    def _initialize_base(
        self,
        day: datetime.date,
        kiosk: VariableKiosk,
        parvalues: ParameterProvider,
        *,
        publish_rates: list[str],
        shape: tuple | None = None,
    ) -> None:
        """Shared initialization for evapotranspiration modules.

        Sets up parameters, rate and state variables, and numerical epsilon for all
        evapotranspiration implementations.
        """
        self.kiosk = kiosk
        self.params = self.Parameters(parvalues)
        if shape is None:
            shape = self.params.shape
        self.params_shape = shape
        self._device = ComputeConfig.get_device()
        self._dtype = ComputeConfig.get_dtype()
        self.rates = self.RateVariables(kiosk, publish=publish_rates, shape=shape)
        self.states = self.StateVariables(kiosk, shape=shape, IDOST=-999, IDWST=-999)
        self._epsilon = torch.tensor(1e-12, dtype=self.dtype, device=self.device)

    def __call__(self, day: datetime.date = None, drv: WeatherDataContainer = None):
        """Callable interface for rate calculation."""
        return self.calc_rates(day, drv)

    @prepare_states
    def integrate(self, day: datetime.date = None, delt=1.0) -> None:
        """Accumulate stress-day counters for water and oxygen stress."""
        rfws_stress = (self.rates.RFWS < 1.0).to(dtype=self.dtype)
        rfos_stress = (self.rates.RFOS < 1.0).to(dtype=self.dtype)
        self.states.IDWST = self.states.IDWST + rfws_stress
        self.states.IDOST = self.states.IDOST + rfos_stress


class _BaseEvapotranspirationNonLayered(_BaseEvapotranspiration):
    """Shared implementation for non-layered evapotranspiration."""

    def _rf_tramx_co2(self, drv: WeatherDataContainer, et0: torch.Tensor) -> torch.Tensor:
        """Return CO2 reduction factor for TRAMX (no CO2 effect in base implementation)."""
        return torch.ones_like(et0)

    @prepare_rates
    def calc_rates(self, day: datetime.date = None, drv: WeatherDataContainer = None):
        p = self.params
        r = self.rates
        k = self.kiosk

        lai = k["LAI"]
        sm = k["SM"]
        # [!] DVS needs to be broadcasted explicetly because it is used
        # in torch.where and the kiosk does not format it correctly
        # TODO see #22
        dvs = _broadcast_to(k["DVS"], self.params_shape, dtype=self.dtype, device=self.device)

        et0 = _get_drv(drv.ET0, self.params_shape, dtype=self.dtype, device=self.device)
        e0 = _get_drv(drv.E0, self.params_shape, dtype=self.dtype, device=self.device)
        es0 = _get_drv(drv.ES0, self.params_shape, dtype=self.dtype, device=self.device)
        rf_tramx_co2 = self._rf_tramx_co2(drv, et0)

        pre_emergence = dvs < 0.0
        if bool(torch.all(pre_emergence)):
            _z = torch.zeros_like(et0)
            _o = torch.ones_like(et0)
            r.EVWMX = _z
            r.EVSMX = _z
            r.TRAMX = _z
            r.TRA = _z
            r.TRALY = _z
            r.RFWS = _o
            r.RFOS = _o
            r.RFTRA = _o
            r.IDWS = False
            r.IDOS = False
            return r.TRA, r.TRAMX

        kglob = 0.75 * p.KDIFTB(dvs)
        # crop specific correction on potential transpiration rate
        et0_crop = torch.clamp(p.CFET * et0, min=0.0)
        # maximum evaporation and transpiration rates
        ekl = torch.exp(-kglob * lai)

        r.EVWMX = e0 * ekl
        r.EVSMX = torch.clamp(es0 * ekl, min=0.0)
        r.TRAMX = et0_crop * (1.0 - ekl) * rf_tramx_co2

       # Critical soil moisture
        swdep = SWEAF(et0_crop, p.DEPNR)
        smcr = (1.0 - swdep) * (p.SMFCF - p.SMW) + p.SMW

        # Reduction factor for transpiration in case of water shortage (RFWS)
        denom = torch.where((smcr - p.SMW).abs() > self._epsilon, (smcr - p.SMW), self._epsilon)
        r.RFWS = torch.clamp((sm - p.SMW) / denom, min=0.0, max=1.0)

        # reduction in transpiration in case of oxygen shortage (RFOS)
        # for non-rice crops, and possibly deficient land drainage
        r.RFOS = torch.ones_like(et0)
        iairdu = p.IAIRDU
        iox = p.IOX
        mask_ox = (iairdu == 0) & (iox == 1)

        if "DSOS" in k:
            dsos = k["DSOS"]
        else:
            dsos = torch.zeros_like(dvs)

        crairc = p.CRAIRC
        sm0 = p.SM0
        denom_ox = torch.where(crairc.abs() > self._epsilon, crairc, self._epsilon)
        rfosmx = torch.clamp((sm0 - sm) / denom_ox, min=0.0, max=1.0)
        rfos = rfosmx + (1.0 - torch.clamp(dsos, max=4.0) / 4.0) * (1.0 - rfosmx)
        r.RFOS = torch.where(mask_ox, rfos, r.RFOS)

        r.RFTRA = r.RFOS * r.RFWS
        r.TRA = r.TRAMX * r.RFTRA
        r.TRALY = r.TRA

        if bool(torch.any(pre_emergence)):
            r.EVWMX = torch.where(pre_emergence, 0.0, r.EVWMX)
            r.EVSMX = torch.where(pre_emergence, 0.0, r.EVSMX)
            r.TRAMX = torch.where(pre_emergence, 0.0, r.TRAMX)
            r.TRA = torch.where(pre_emergence, 0.0, r.TRA)
            r.TRALY = torch.where(pre_emergence, 0.0, r.TRALY)
            r.RFWS = torch.where(pre_emergence, 1.0, r.RFWS)
            r.RFOS = torch.where(pre_emergence, 1.0, r.RFOS)
            r.RFTRA = torch.where(pre_emergence, 1.0, r.RFTRA)

        r.IDWS = bool(torch.any(r.RFWS < 1.0))
        r.IDOS = bool(torch.any(r.RFOS < 1.0))
        return r.TRA, r.TRAMX


class Evapotranspiration(_BaseEvapotranspirationNonLayered):
    """Calculation of potential evaporation (water and soil) rates and actual
    crop transpiration rate.

    **Simulation parameters**

    | Name   | Description                                             | Type | Unit |
    |--------|---------------------------------------------------------|------|------|
    | CFET   | Correction factor for potential transpiration rate       | SCr  | -    |
    | DEPNR  | Dependency number for crop sensitivity to soil moisture stress.       | SCr  | -    |
    | KDIFTB | Extinction coefficient for diffuse visible light vs DVS  | TCr  | -    |
    | IAIRDU | Switch airducts on (1) or off (0)                        | SCr  | -    |
    | IOX    | Switch oxygen stress on (1) or off (0)                   | SCr  | -    |
    | CRAIRC | Critical air content for root aeration                   | SSo  | -    |
    | SM0    | Soil porosity                                            | SSo  | -    |
    | SMW    | Volumetric soil moisture at wilting point                | SSo  | -    |
    | SMFCF  | Volumetric soil moisture at field capacity               | SSo  | -    |

    **State variables**

    | Name  | Description                        | Pbl | Unit |
    |-------|------------------------------------|-----|------|
    | IDWST | Number of days with water stress   | N   | -    |
    | IDOST | Number of days with oxygen stress  | N   | -    |

    **Rate variables**

    | Name  | Description                                         | Pbl | Unit      |
    |-------|-----------------------------------------------------|-----|-----------|
    | EVWMX | Max evaporation rate from open water surface        | Y   | cm day⁻¹  |
    | EVSMX | Max evaporation rate from wet soil surface          | Y   | cm day⁻¹  |
    | TRAMX | Max transpiration rate from canopy                  | Y   | cm day⁻¹  |
    | TRA   | Actual transpiration rate from canopy               | Y   | cm day⁻¹  |
    | IDOS  | Indicates oxygen stress on this day (True|False)    | N   | -         |
    | IDWS  | Indicates water stress on this day (True|False)     | N   | -         |
    | RFWS  | Reduction factor for water stress                   | N   | -         |
    | RFOS  | Reduction factor for oxygen stress                  | N   | -         |
    | RFTRA | Combined reduction factor for transpiration         | Y   | -         |

    **External dependencies**

    | Name | Description                      | Provided by   | Unit |
    |------|----------------------------------|---------------|------|
    | DVS  | Crop development stage            | Phenology     | -    |
    | LAI  | Leaf area index                   | Leaf dynamics | -    |
    | SM   | Volumetric soil moisture content  | Waterbalance  | -    |
    """

    class Parameters(TensorParamTemplate):
        CFET = Tensor(-99.0)
        DEPNR = Tensor(-99.0)
        KDIFTB = AfgenTrait()
        IAIRDU = Tensor(-99.0)
        IOX = Tensor(-99.0)
        CRAIRC = Tensor(-99.0)
        SM0 = Tensor(-99.0)
        SMW = Tensor(-99.0)
        SMFCF = Tensor(-99.0)

    def initialize(
        self,
        day: datetime.date,
        kiosk: VariableKiosk,
        parvalues: ParameterProvider,
        shape: tuple | None = None,
    ) -> None:
        """Initialize the standard evapotranspiration module (no CO2 effects).

        Args:
            day (datetime.date): The starting date of the simulation.
            kiosk (VariableKiosk): A container for registering and publishing
                (internal and external) state variables. See PCSE documentation for
                details.
            parvalues (ParameterProvider): A dictionary-like container holding
                all parameter sets (crop, soil, site) as key/value. The values are
                arrays or scalars. See PCSE documentation for details.
            shape (tuple | torch.Size | None): Target shape for the state and rate variables.
        """
        self._initialize_base(
            day,
            kiosk,
            parvalues,
            publish_rates=["EVWMX", "EVSMX", "TRAMX", "TRA", "RFTRA"],
            shape=shape,
        )


class EvapotranspirationCO2(_BaseEvapotranspirationNonLayered):
    """Potential evaporation and crop transpiration with CO2 effect on TRAMX.

    **Simulation parameters**

    | Name     | Description                                            | Type | Unit |
    |----------|--------------------------------------------------------|------|------|
    | CFET     | Correction factor for potential transpiration rate      | SCr  | -    |
    | DEPNR    | Crop dependency number (drought sensitivity, 1..5)      | SCr  | -    |
    | KDIFTB   | Extinction coefficient for diffuse visible light vs DVS | TCr  | -    |
    | IAIRDU   | Switch airducts on (1) or off (0)                       | SCr  | -    |
    | IOX      | Switch oxygen stress on (1) or off (0)                  | SCr  | -    |
    | CRAIRC   | Critical air content for root aeration                  | SSo  | -    |
    | SM0      | Soil porosity                                           | SSo  | -    |
    | SMW      | Volumetric soil moisture at wilting point               | SSo  | -    |
    | SMFCF    | Volumetric soil moisture at field capacity              | SSo  | -    |
    | CO2      | Atmospheric CO2 concentration (used if not in drivers)  | SCr  | ppm  |
    | CO2TRATB | Reduction factor for TRAMX as function of CO2           | TCr  | -    |

    **State variables**

    | Name  | Description                        | Pbl | Unit |
    |-------|------------------------------------|-----|------|
    | IDWST | Number of days with water stress   | N   | -    |
    | IDOST | Number of days with oxygen stress  | N   | -    |

    **Rate variables**

    | Name  | Description                                         | Pbl | Unit      |
    |-------|-----------------------------------------------------|-----|-----------|
    | EVWMX | Max evaporation rate from open water surface        | Y   | cm day⁻¹  |
    | EVSMX | Max evaporation rate from wet soil surface          | Y   | cm day⁻¹  |
    | TRAMX | Max transpiration rate from canopy (CO2-adjusted)   | Y   | cm day⁻¹  |
    | TRA   | Actual transpiration rate from canopy               | Y   | cm day⁻¹  |
    | RFWS  | Reduction factor for water stress                   | N   | -         |
    | RFOS  | Reduction factor for oxygen stress                  | N   | -         |
    | RFTRA | Combined reduction factor for transpiration         | Y   | -         |

    **External dependencies**

    | Name | Description                      | Provided by   | Unit |
    |------|----------------------------------|---------------|------|
    | DVS  | Crop development stage            | Phenology     | -    |
    | LAI  | Leaf area index                   | Leaf dynamics | -    |
    | SM   | Volumetric soil moisture content  | Waterbalance  | -    |
    """

    class Parameters(TensorParamTemplate):
        CFET = Tensor(-99.0)
        DEPNR = Tensor(-99.0)
        KDIFTB = AfgenTrait()
        IAIRDU = Tensor(-99.0)
        IOX = Tensor(-99.0)
        CRAIRC = Tensor(-99.0)
        SM0 = Tensor(-99.0)
        SMW = Tensor(-99.0)
        SMFCF = Tensor(-99.0)
        CO2 = Tensor(-99.0)
        CO2TRATB = AfgenTrait()

    def initialize(
        self,
        day: datetime.date,
        kiosk: VariableKiosk,
        parvalues: ParameterProvider,
        shape: tuple | None = None,
    ) -> None:
        """Initialize the CO2-aware evapotranspiration module."""
        self._initialize_base(
            day,
            kiosk,
            parvalues,
            publish_rates=["EVWMX", "EVSMX", "TRAMX", "TRA", "TRALY", "RFTRA"],
            shape=shape,
        )

    def _rf_tramx_co2(self, drv: WeatherDataContainer, et0: torch.Tensor) -> torch.Tensor:
        """Calculate CO2 reduction factor for TRAMX based on atmospheric CO2 concentration."""
        p = self.params

        if hasattr(drv, "CO2") and drv.CO2 is not None:
            co2 = _get_drv(drv.CO2, self.params_shape, dtype=self.dtype, device=self.device)
        else:
            co2 = p.CO2
        return p.CO2TRATB(co2)


class EvapotranspirationCO2Layered(_BaseEvapotranspiration):
    """Layered-soil evapotranspiration with CO2 effect on TRAMX.

    This implementation expects a layered soil water balance.

    **Simulation parameters**

    | Name     | Description                                            | Type | Unit |
    |----------|--------------------------------------------------------|------|------|
    | CFET     | Correction factor for potential transpiration rate      | SCr  | -    |
    | DEPNR    | Crop dependency number (drought sensitivity, 1..5)      | SCr  | -    |
    | KDIFTB   | Extinction coefficient for diffuse visible light vs DVS | TCr  | -    |
    | IAIRDU   | Switch airducts on (1) or off (0)                       | SCr  | -    |
    | IOX      | Switch oxygen stress on (1) or off (0)                  | SCr  | -    |
    | CO2      | Atmospheric CO2 concentration (used if not in drivers)  | SCr  | ppm  |
    | CO2TRATB | Reduction factor for TRAMX as function of CO2           | TCr  | -    |

    Layer-specific soil parameters (SMW, SMFCF, SM0, CRAIRC, Thickness) are
    taken from `soil_profile` entries.

    **State variables**

    | Name  | Description                        | Pbl | Unit |
    |-------|------------------------------------|-----|------|
    | IDWST | Number of days with water stress   | N   | -    |
    | IDOST | Number of days with oxygen stress  | N   | -    |

    **Rate variables**

    | Name  | Description                                         | Pbl | Unit      |
    |-------|-----------------------------------------------------|-----|-----------|
    | EVWMX | Max evaporation rate from open water surface        | Y   | cm day⁻¹  |
    | EVSMX | Max evaporation rate from wet soil surface          | Y   | cm day⁻¹  |
    | TRAMX | Max transpiration rate from canopy (CO2-adjusted)   | Y   | cm day⁻¹  |
    | TRA   | Actual canopy transpiration (sum over layers)       | Y   | cm day⁻¹  |
    | TRALY | Transpiration per soil layer                        | Y   | cm day⁻¹  |
    | RFWS  | Water-stress reduction per layer                    | N   | -         |
    | RFOS  | Oxygen-stress reduction per layer                   | N   | -         |
    | RFTRA | Combined reduction factor for transpiration         | Y   | -         |

    **External dependencies**

    | Name | Description                      | Provided by   | Unit |
    |------|----------------------------------|---------------|------|
    | DVS  | Crop development stage            | Phenology     | -    |
    | LAI  | Leaf area index                   | Leaf dynamics | -    |
    | RD   | Rooting depth                     | Root dynamics | cm   |
    | SM   | Soil moisture per layer           | Waterbalance  | -    |
    """

    soil_profile = Any()

    class Parameters(TensorParamTemplate):
        CFET = Tensor(-99.0)
        DEPNR = Tensor(-99.0)
        KDIFTB = AfgenTrait()
        IAIRDU = Tensor(-99.0)
        IOX = Tensor(-99.0)
        CO2 = Tensor(-99.0)
        CO2TRATB = AfgenTrait()

    class RateVariables(TensorRatesTemplate):
        EVWMX = Tensor(0)
        EVSMX = Tensor(0)
        TRAMX = Tensor(0)
        TRA = Tensor(0)
        TRALY = Tensor(0)
        IDOS = Bool(False)
        IDWS = Bool(False)
        RFWS = Tensor(0)
        RFOS = Tensor(0)
        RFTRALY = Tensor(0)
        RFTRA = Tensor(0)

    class StateVariables(TensorStatesTemplate):
        IDOST = Tensor(-99.0)
        IDWST = Tensor(-99.0)

    def initialize(
        self,
        day: datetime.date,
        kiosk: VariableKiosk,
        parvalues: ParameterProvider,
        shape: tuple | None = None,
    ) -> None:
        """Initialize the layered-soil CO2-aware evapotranspiration module.

        Sets up layer-specific soil parameters and internal oxygen stress tracking.
        """
        self.soil_profile = parvalues["soil_profile"]
        self._initialize_base(
            day,
            kiosk,
            parvalues,
            publish_rates=["EVWMX", "EVSMX", "TRAMX", "TRA", "TRALY", "RFTRA"],
            shape=shape,
        )

        # Pre-stack layer soil properties as tensors
        n_layers = len(self.soil_profile)
        self._n_layers = n_layers
        self._layer_smw = torch.tensor(
            [layer.SMW for layer in self.soil_profile], dtype=self.dtype, device=self.device
        )
        self._layer_smfcf = torch.tensor(
            [layer.SMFCF for layer in self.soil_profile], dtype=self.dtype, device=self.device
        )
        self._layer_sm0 = torch.tensor(
            [layer.SM0 for layer in self.soil_profile], dtype=self.dtype, device=self.device
        )
        self._layer_crairc = torch.tensor(
            [layer.CRAIRC for layer in self.soil_profile], dtype=self.dtype, device=self.device
        )
        thicknesses = torch.tensor(
            [layer.Thickness for layer in self.soil_profile], dtype=self.dtype, device=self.device
        )
        self._layer_depth_hi = torch.cumsum(thicknesses, dim=0)
        self._layer_depth_lo = self._layer_depth_hi - thicknesses

        # Internal DSOS tracker for layered oxygen-stress response
        self._dsos = torch.zeros(self.params_shape, dtype=self.dtype, device=self.device)

    def _rf_tramx_co2(self, drv: WeatherDataContainer, et0: torch.Tensor) -> torch.Tensor:
        """Calculate CO2 reduction factor for TRAMX using CO2 from driver or parameters."""
        p = self.params
        if hasattr(drv, "CO2") and drv.CO2 is not None:
            co2 = _get_drv(drv.CO2, self.params_shape, dtype=self.dtype, device=self.device)
        else:
            co2 = p.CO2
        return p.CO2TRATB(co2)

    @prepare_rates
    def calc_rates(self, day: datetime.date = None, drv: WeatherDataContainer = None):
        """Calculate daily evapotranspiration rates per soil layer with CO2 effects.

        Computes transpiration and stress factors for each soil layer based on root
        distribution and layer-specific soil moisture conditions.
        """
        p = self.params
        r = self.rates
        k = self.kiosk

        dvs = k["DVS"]
        lai = k["LAI"]
        rd = k["RD"]

        pre_emergence = dvs < 0.0
        n_layers = self._n_layers

        et0 = _get_drv(drv.ET0, self.params_shape, dtype=self.dtype, device=self.device)
        e0 = _get_drv(drv.E0, self.params_shape, dtype=self.dtype, device=self.device)
        es0 = _get_drv(drv.ES0, self.params_shape, dtype=self.dtype, device=self.device)

        rf_tramx_co2 = self._rf_tramx_co2(drv, et0)

        if bool(torch.all(pre_emergence)):
            _z = torch.zeros_like(et0)
            _o = torch.ones_like(et0)
            _layered_shape = (n_layers,) + self.params_shape
            r.EVWMX = _z
            r.EVSMX = _z
            r.TRAMX = _z
            r.TRA = _z
            r.TRALY = torch.zeros(_layered_shape, dtype=self.dtype, device=self.device)
            r.RFWS = torch.ones(_layered_shape, dtype=self.dtype, device=self.device)
            r.RFOS = torch.ones(_layered_shape, dtype=self.dtype, device=self.device)
            r.RFTRA = _o
            r.IDWS = False
            r.IDOS = False
            return r.TRA, r.TRAMX

        et0_crop = torch.clamp(p.CFET * et0, min=0.0)
        kglob = 0.75 * p.KDIFTB(dvs)
        ekl = torch.exp(-kglob * lai)
        r.EVWMX = e0 * ekl
        r.EVSMX = torch.clamp(es0 * ekl, min=0.0)
        r.TRAMX = et0_crop * (1.0 - ekl) * rf_tramx_co2

        swdep = SWEAF(et0_crop, p.DEPNR)

        # Layered soil moisture can be provided as:
        # - torch.Tensor with shape (n_layers, *params_shape)
        # - list/tuple of length n_layers, each element scalar or tensor
        sm_layers = k["SM"]
        if isinstance(sm_layers, torch.Tensor):
            sm_layers_t = sm_layers.to(dtype=self.dtype, device=self.device)
        elif isinstance(sm_layers, (list, tuple)):
            if len(sm_layers) != n_layers:
                raise ValueError(
                    "Layered evapotranspiration expects SM with "
                    + f"{n_layers} layers, got {len(sm_layers)}."
                )
            sm_layers_t = torch.stack(
                [
                    _broadcast_to(sm_i, self.params_shape, dtype=self.dtype, device=self.device)
                    for sm_i in sm_layers
                ],
                dim=0,
            )
        else:
            sm_layers_t = torch.as_tensor(sm_layers, dtype=self.dtype, device=self.device)
            if sm_layers_t.dim() == 1:
                # Interpret as per-layer scalars
                if sm_layers_t.shape[0] != n_layers:
                    raise ValueError(
                        "Layered evapotranspiration expects SM with "
                        + f"{n_layers} layers, got {sm_layers_t.shape[0]}."
                    )
                sm_layers_t = torch.stack(
                    [
                        _broadcast_to(
                            sm_layers_t[i], self.params_shape, dtype=self.dtype, device=self.device
                        )
                        for i in range(n_layers)
                    ],
                    dim=0,
                )

        if sm_layers_t.shape[0] != n_layers:
            raise ValueError(
                "Layered evapotranspiration expects SM first dim to be "
                + f"{n_layers}, got {sm_layers_t.shape[0]}."
            )

        # Reshape pre-stacked layer properties for broadcasting against
        # (n_layers, *params_shape) tensors: (n_layers,) → (n_layers, 1, 1, ...)
        ndim = len(self.params_shape)
        expand = (-1,) + (1,) * ndim
        layer_smw = self._layer_smw.view(expand)
        layer_smfcf = self._layer_smfcf.view(expand)
        depth_lo = self._layer_depth_lo.view(expand)
        depth_hi = self._layer_depth_hi.view(expand)

        # Vectorised RFWS across all layers: (n_layers, *params_shape)
        smcr = (1.0 - swdep) * (layer_smfcf - layer_smw) + layer_smw
        denom = torch.where(
            (smcr - layer_smw).abs() > self._epsilon, smcr - layer_smw, self._epsilon
        )
        r.RFWS = torch.clamp((sm_layers_t - layer_smw) / denom, min=0.0, max=1.0)

        # Vectorised root fraction across all layers: (n_layers, *params_shape)
        root_len = torch.clamp(torch.minimum(rd, depth_hi) - depth_lo, min=0.0)
        root_fraction = torch.where(rd > self._epsilon, root_len / rd, 0.0)

        # Oxygen-stress reduction factor (sequential across layers due to
        # temporal _dsos accumulator that feeds forward between layers).
        r.RFOS = torch.ones_like(r.RFWS)
        mask_ox = (p.IAIRDU == 0) & (p.IOX == 1)
        if bool(torch.any(mask_ox)):
            layer_sm0 = self._layer_sm0.view(expand)
            layer_crairc = self._layer_crairc.view(expand)
            for i in range(n_layers):
                smair = layer_sm0[i] - layer_crairc[i]
                self._dsos = torch.where(
                    sm_layers_t[i] >= smair,
                    torch.clamp(self._dsos + 1.0, max=4.0),
                    0.0,
                )
                denom_ox = torch.where(
                    layer_crairc[i].abs() > self._epsilon, layer_crairc[i], self._epsilon
                )
                rfosmx = torch.clamp((layer_sm0[i] - sm_layers_t[i]) / denom_ox, min=0.0, max=1.0)
                r.RFOS[i] = rfosmx + (1.0 - torch.clamp(self._dsos, max=4.0) / 4.0) * (1.0 - rfosmx)

        # Transpiration per layer
        rftra = r.RFOS * r.RFWS
        r.TRALY = r.TRAMX * rftra * root_fraction
        r.TRA = r.TRALY.sum(dim=0)
        r.RFTRA = torch.where(r.TRAMX > self._epsilon, r.TRA / r.TRAMX, 1.0)

        if bool(torch.any(pre_emergence)):
            r.EVWMX = torch.where(pre_emergence, 0.0, r.EVWMX)
            r.EVSMX = torch.where(pre_emergence, 0.0, r.EVSMX)
            r.TRAMX = torch.where(pre_emergence, 0.0, r.TRAMX)
            r.TRA = torch.where(pre_emergence, 0.0, r.TRA)
            r.RFTRA = torch.where(pre_emergence, 1.0, r.RFTRA)

            pre_layers = pre_emergence.unsqueeze(0).expand_as(r.RFWS)
            r.RFWS = torch.where(pre_layers, 1.0, r.RFWS)
            r.RFOS = torch.where(pre_layers, 1.0, r.RFOS)
            r.TRALY = torch.where(pre_layers, 0.0, r.TRALY)

        r.IDWS = bool(torch.any(r.RFWS < 1.0))
        r.IDOS = bool(torch.any(r.RFOS < 1.0))
        return r.TRA, r.TRAMX

    def __call__(self, day: datetime.date = None, drv: WeatherDataContainer = None):
        """Callable interface for rate calculation."""
        return self.calc_rates(day, drv)

    @prepare_states
    def integrate(self, day: datetime.date = None, delt=1.0) -> None:
        """Accumulate stress-day counters based on any layer experiencing stress."""
        rfws_stress = (self.rates.RFWS < 1.0).any(dim=0).to(dtype=self.dtype)
        rfos_stress = (self.rates.RFOS < 1.0).any(dim=0).to(dtype=self.dtype)
        self.states.IDWST = self.states.IDWST + rfws_stress
        self.states.IDOST = self.states.IDOST + rfos_stress
