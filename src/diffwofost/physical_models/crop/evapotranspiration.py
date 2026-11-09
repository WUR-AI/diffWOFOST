import datetime
import torch
from pcse.base import SimulationObject
from pcse.base.parameter_providers import ParameterProvider
from pcse.base.variablekiosk import VariableKiosk
from pcse.traitlets import Any
from pcse.traitlets import Instance
from diffwofost.physical_models.base import TensorParamTemplate
from diffwofost.physical_models.base import TensorRatesTemplate
from diffwofost.physical_models.base import TensorStatesTemplate
from diffwofost.physical_models.config import ComputeConfig
from diffwofost.physical_models.kernels import kernel
from diffwofost.physical_models.traitlets import Tensor
from diffwofost.physical_models.utils import AfgenTrait
from diffwofost.physical_models.utils import _broadcast_to


def _stack_layer_property(soil_properties, soil_profile, dtype, device) -> torch.Tensor:
    """Stack per-layer soil properties without dropping autograd history.

    Each layer property must be a Python number or a single-element tensor
    (reshaped to ``()``). ``torch.tensor([...])`` would break the graph when a
    value already requires grad; ``stack`` of ``.to()`` tensors keeps it.
    """
    columns = []
    for layer in soil_profile:
        value = getattr(layer, soil_properties)
        if isinstance(value, torch.Tensor):
            columns.append(value.to(dtype=dtype, device=device).reshape(()))
        else:
            columns.append(torch.tensor(value, dtype=dtype, device=device))
    return torch.stack(columns)


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
    # curve for CGNR 5, and other curves at fixed distance below it
    sweaf = 1.0 / (A + B * ET0) - (5.0 - DEPNR) * 0.10
    # Correction for lower curves (CGNR less than 3)
    correction = (ET0 - 0.6) / (DEPNR * (DEPNR + 3.0))
    # NOTE: PCSE applies `correction` only when `DEPNR < 3` (hard switch), which
    # is non-differentiable at `DEPNR==3`.
    #
    # We use a Straight-Through Estimator (STE): the forward pass applies the
    # exact hard switch (identical to PCSE), while the backward pass routes
    # gradients through a smooth sigmoid surrogate so that autograd works.
    # TODO: sharpness can be exposed as a parameter
    _sigmoid_sharpness = 1000.0
    _sigmoid_epsilon = 1e-14

    # soft mask using sigmoid
    soft_mask = torch.sigmoid((3.0 - DEPNR - _sigmoid_epsilon) / _sigmoid_sharpness)

    # original hard mask
    hard_mask = (DEPNR < 3.0).to(dtype=DEPNR.dtype)

    # STE method: during forward pass the hard_mask is used, during
    # backpropagation the gradient is computed only through soft_mask.
    correction_mask = hard_mask.detach() + soft_mask - soft_mask.detach()
    sweaf = sweaf + correction * correction_mask
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
        elif "CO2" in parvalues and "CO2TRATB" in parvalues:
            self.etmodule = EvapotranspirationCO2(day, kiosk, parvalues, shape=shape)
        else:
            self.etmodule = Evapotranspiration(day, kiosk, parvalues, shape=shape)

    def calc_rates(self, day: datetime.date, drv: dict):
        """Delegate rate calculation to the selected evapotranspiration module.

        Args:
            day (datetime.date): The current date of the simulation.
            drv (dict): A container holding weather data elements as key/value. The values are
                arrays or scalars.
        """
        return self.etmodule.calc_rates(day, drv)

    def __call__(self, day: datetime.date, drv: dict):
        """Callable interface for rate calculation."""
        return self.calc_rates(day, drv)

    def integrate(self, day: datetime.date, delt: float = 1.0) -> None:
        """Delegate state integration to the selected evapotranspiration module.

        Args:
            day (datetime.date, optional): The current date of the simulation.
            delt (float, optional): The time step for integration. Defaults to 1.0.
        """
        return self.etmodule.integrate(day, delt)

    def finalize(self, day: datetime.date) -> None:
        """Delegate finalization to the selected evapotranspiration module."""
        self.etmodule.finalize(day)


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
        IDOS = Tensor(False, dtype=torch.bool)
        IDWS = Tensor(False, dtype=torch.bool)
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

        # helper variable for counting total days with water and oxygen stress (IDWST, IDOST)
        self._IDWST = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self._IDOST = torch.zeros(shape, dtype=self.dtype, device=self.device)

    def __call__(self, day: datetime.date, drv: dict):
        """Callable interface for rate calculation."""
        return self.calc_rates(day, drv)

    def integrate(self, day: datetime.date, delt: float = 1.0) -> None:
        """Accumulate stress-day counters for water and oxygen stress."""
        rfws_stress = (self.rates.RFWS < 1.0).to(dtype=self.dtype)
        rfos_stress = (self.rates.RFOS < 1.0).to(dtype=self.dtype)
        self._IDWST = self._IDWST + rfws_stress
        self._IDOST = self._IDOST + rfos_stress

    def finalize(self, day: datetime.date) -> None:
        """Finalize the evapotranspiration simulation."""
        self.states.IDWST = self._IDWST
        self.states.IDOST = self._IDOST
        SimulationObject.finalize(self, day)


class _BaseEvapotranspirationNonLayered(_BaseEvapotranspiration):
    """Shared implementation for non-layered evapotranspiration."""

    def _rf_tramx_co2(self, drv: dict, et0: torch.Tensor) -> torch.Tensor | float:
        """Return CO2 reduction factor for TRAMX (no CO2 effect in base implementation)."""
        return 1.0

    def calc_rates(self, day: datetime.date, drv: dict):
        p = self.params
        r = self.rates
        k = self.kiosk

        dvs = _broadcast_to(k["DVS"], self.params_shape, dtype=self.dtype, device=self.device)
        et0 = torch.as_tensor(drv["ET0"])
        e0 = torch.as_tensor(drv["E0"])
        es0 = torch.as_tensor(drv["ES0"])

        # reduction factor for CO2 on TRAMX
        rf_tramx_co2 = self._rf_tramx_co2(drv, et0)

        dsos = k["DSOS"] if "DSOS" in k else None

        (r.EVWMX, r.EVSMX, r.TRAMX, r.RFWS, r.RFOS, r.RFTRA, r.TRA) = _non_layered_rates(
            dvs,
            k["LAI"],
            k["SM"],
            dsos,
            et0,
            e0,
            es0,
            rf_tramx_co2,
            p.KDIFTB,
            p.CFET,
            p.DEPNR,
            p.SMFCF,
            p.SMW,
            p.IAIRDU,
            p.IOX,
            p.CRAIRC,
            p.SM0,
            self._epsilon,
        )

        # Counting stress days
        r.IDWS = r.RFWS < 1.0
        r.IDOS = r.RFOS < 1.0
        return r.TRA, r.TRAMX


@kernel
def _non_layered_rates(
    dvs, lai, sm, dsos, et0, e0, es0, rf_tramx_co2,
    KDIFTB, CFET, DEPNR, SMFCF, SMW, IAIRDU, IOX, CRAIRC, SM0, epsilon,
):  # fmt: skip
    """Evapotranspiration rates, see `_BaseEvapotranspirationNonLayered.calc_rates`."""
    # If DVS < 0, the crop has not yet emerged, so we zero the rates using a mask
    # A mask (1 if DVS >= 0, 0 if DVS < 0)
    dvs_mask = (dvs >= 0.0).to(dtype=dvs.dtype)

    kglob = 0.75 * KDIFTB(dvs)
    # crop specific correction on potential transpiration rate
    et0_crop = torch.clamp(CFET * et0, min=0.0)

    # maximum evaporation and transpiration rates
    ekl = torch.exp(-kglob * lai)

    EVWMX = dvs_mask * e0 * ekl
    EVSMX = dvs_mask * torch.clamp(es0 * ekl, min=0.0)
    TRAMX = dvs_mask * et0_crop * (1.0 - ekl) * rf_tramx_co2

    # Critical soil moisture
    swdep = SWEAF(et0_crop, DEPNR)
    smcr = (1.0 - swdep) * (SMFCF - SMW) + SMW

    # Reduction factor for transpiration in case of water shortage (RFWS)
    denom = torch.where((smcr - SMW).abs() > epsilon, (smcr - SMW), epsilon)
    RFWS = dvs_mask * torch.clamp((sm - SMW) / denom, min=0.0, max=1.0) + (1.0 - dvs_mask)

    # reduction in transpiration in case of oxygen shortage (RFOS)
    # for non-rice crops, and possibly deficient land drainage
    mask_ox = (IAIRDU == 0) & (IOX == 1)
    if dsos is None:
        dsos = torch.zeros_like(dvs)
    denom_ox = torch.where(CRAIRC.abs() > epsilon, CRAIRC, epsilon)
    rfosmx = torch.clamp((SM0 - sm) / denom_ox, min=0.0, max=1.0)

    # maximum reduction reached after 4 days
    rfos = rfosmx + (1.0 - torch.clamp(dsos, max=4.0) / 4.0) * (1.0 - rfosmx)
    RFOS = torch.where(mask_ox, rfos, torch.ones_like(et0))

    # Pre-emergence: RFOS = 1.0
    RFOS = dvs_mask * RFOS + (1.0 - dvs_mask)

    # Transpiration rate multiplied with reduction factors for oxygen and water
    RFTRA = RFOS * RFWS
    TRA = TRAMX * RFTRA
    return EVWMX, EVSMX, TRAMX, RFWS, RFOS, RFTRA, TRA


class Evapotranspiration(_BaseEvapotranspirationNonLayered):
    """Potential evaporation (water and soil) rates and crop transpiration rate.

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
    | IDOS  | Oxygen stress on this day, per element (True|False) | N   | -         |
    | IDWS  | Water stress on this day, per element (True|False)  | N   | -         |
    | RFWS  | Reduction factor for water stress                   | N   | -         |
    | RFOS  | Reduction factor for oxygen stress                  | N   | -         |
    | RFTRA | Combined reduction factor for transpiration         | Y   | -         |

    **External dependencies**

    | Name | Description                      | Provided by   | Unit |
    |------|----------------------------------|---------------|------|
    | DVS  | Crop development stage            | Phenology     | -    |
    | LAI  | Leaf area index                   | Leaf dynamics | -    |
    | SM   | Volumetric soil moisture content  | Waterbalance  | -    |


    **Outputs**

    | Name  | Description                                         | Pbl | Unit      |
    |-------|-----------------------------------------------------|-----|-----------|
    | TRA   | Actual transpiration rate from canopy               | Y   | cm day⁻¹  |
    | TRAMX | Max transpiration rate from canopy                  | Y   | cm day⁻¹  |
    | EVWMX | Max evaporation rate from open water surface        | Y   | cm day⁻¹  |
    | EVSMX | Max evaporation rate from wet soil surface          | Y   | cm day⁻¹  |
    | RFTRA | Combined reduction factor for transpiration         | Y   | -         |

    **Gradient mapping (which parameters have a gradient):**

    | Output | Parameters influencing it                          |
    |--------|----------------------------------------------------|
    | EVWMX  | KDIFTB                                             |
    | EVSMX  | KDIFTB                                             |
    | TRAMX  | CFET, KDIFTB                                       |
    | TRA    | CFET, KDIFTB, DEPNR, SMFCF, SMW, CRAIRC, SM0      |
    | RFTRA  | CFET, DEPNR, SMFCF, SMW, CRAIRC, SM0               |

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
    | CFET     | Correction factor for potential transpiration rate      | S  | -    |
    | DEPNR    | Dependency number for crop sensitivity to soil moisture stress | S  | -    |
    | KDIFTB   | Extinction coefficient for diffuse visible light vs DVS | T  | -    |
    | IAIRDU   | Switch airducts on (1) or off (0)                       | S  | -    |
    | IOX      | Switch oxygen stress on (1) or off (0)                  | S  | -    |
    | CRAIRC   | Critical air content for root aeration                  | S  | -    |
    | SM0      | Soil porosity                                           | S  | -    |
    | SMW      | Volumetric soil moisture at wilting point               | S  | -    |
    | SMFCF    | Volumetric soil moisture at field capacity              | S  | -    |
    | CO2      | Atmospheric CO2 concentration (used if not in drivers)  | S  | ppm  |
    | CO2TRATB | Reduction factor for TRAMX as function of CO2           | T  | -    |

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
    | IDOS  | Oxygen stress on this day, per element (True|False) | N   | -         |
    | IDWS  | Water stress on this day, per element (True|False)  | N   | -         |
    | RFWS  | Reduction factor for water stress                   | Y   | -         |
    | RFOS  | Reduction factor for oxygen stress                  | Y  | -         |
    | RFTRA | Combined reduction factor for transpiration         | Y   | -         |

    **External dependencies**

    | Name | Description                      | Provided by   | Unit |
    |------|----------------------------------|---------------|------|
    | DVS  | Crop development stage            | Phenology     | -    |
    | LAI  | Leaf area index                   | Leaf dynamics | -    |
    | SM   | Volumetric soil moisture content  | Waterbalance  | -    |

    **Gradient mapping (which parameters have a gradient):**

    | Output | Parameters influencing it                                      |
    |--------|----------------------------------------------------------------|
    | EVWMX  | KDIFTB                                                         |
    | EVSMX  | KDIFTB                                                         |
    | TRAMX  | CFET, KDIFTB, CO2, CO2TRATB                                    |
    | TRA    | CFET, KDIFTB, DEPNR, SMFCF, SMW, CRAIRC, SM0, CO2, CO2TRATB    |
    | RFTRA  | CFET, DEPNR, SMFCF, SMW, CRAIRC, SM0, CO2, CO2TRATB            |

    [!NOTE]
    ``DEPNR < 3`` uses the straight-through estimator in ``SWEAF``. ``IAIRDU`` and
    ``IOX`` are switches.
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
        """Initialize the CO2-aware evapotranspiration module.

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

    def _rf_tramx_co2(self, drv: dict, et0: torch.Tensor) -> torch.Tensor:
        """Calculate CO2 reduction factor for TRAMX based on atmospheric CO2 concentration."""
        if "CO2" in drv and drv["CO2"] is not None:
            co2 = drv["CO2"]
        else:
            co2 = self.params.CO2
        return self.params.CO2TRATB(co2)


class EvapotranspirationCO2Layered(_BaseEvapotranspiration):
    """Layered-soil evapotranspiration with CO2 effect on TRAMX.

    This implementation expects a layered soil water balance.

    **Simulation parameters**

    | Name     | Description                                            | Type | Unit |
    |----------|--------------------------------------------------------|------|------|
    | CFET     | Correction factor for potential transpiration rate      | S  | -    |
    | DEPNR    | Dependency number for crop sensitivity to soil moisture stress.     | S  | -    |
    | KDIFTB   | Extinction coefficient for diffuse visible light vs DVS | T  | -    |
    | IAIRDU   | Switch airducts on (1) or off (0)                       | S  | -    |
    | IOX      | Switch oxygen stress on (1) or off (0)                  | S  | -    |
    | CO2      | Atmospheric CO2 concentration (used if not in drivers)  | S  | ppm  |
    | CO2TRATB | Reduction factor for TRAMX as function of CO2           | T  | -    |

    Layer-specific soil parameters (SMW, SMFCF, SM0, CRAIRC, Thickness) are
    taken from `soil_profile` entries.
    **Soil parameters**

    | Name     | Description                                            | Type | Unit |
    |----------|--------------------------------------------------------|------|------|
    | SMW      | Volumetric soil moisture content at wilting  point     | S    | -    |
    | SMFCF    | Volumetric soil moisture content at field capacity     | S    | -    |
    | SM0      | Soil porosity                                          | S    | -    |
    | CRAIRC   | Critical air content for root aeration                 | S    | -    |

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
    | IDOS  | Oxygen stress on this day, per element (True|False) | N   | -         |
    | IDWS  | Water stress on this day, per element (True|False)  | N   | -         |
    | TRALY | Transpiration per soil layer                        | Y   | cm day⁻¹  |
    | RFWS  | Water-stress reduction per layer                    | Y   | -         |
    | RFOS  | Oxygen-stress reduction per layer                   | Y   | -         |
    | RFTRA | Combined reduction factor for transpiration         | Y   | -         |

    **External dependencies**

    | Name | Description                      | Provided by   | Unit |
    |------|----------------------------------|---------------|------|
    | DVS  | Crop development stage            | Phenology     | -    |
    | LAI  | Leaf area index                   | Leaf dynamics | -    |
    | RD   | Rooting depth                     | Root dynamics | cm   |
    | SM   | Soil moisture per layer           | Waterbalance  | -    |

    **Gradient mapping (which parameters have a gradient):**

    | Output | Parameters influencing it                                              |
    |--------|------------------------------------------------------------------------|
    | EVWMX  | KDIFTB                                                                 |
    | EVSMX  | KDIFTB                                                                 |
    | TRAMX  | CFET, KDIFTB, CO2, CO2TRATB                                            |
    | TRA    | CFET, KDIFTB, DEPNR, CO2, CO2TRATB, layer SMW, SMFCF, SM0, CRAIRC      |
    | RFTRA  | CFET, DEPNR, CO2, CO2TRATB, layer SMW, SMFCF, SM0, CRAIRC              |

    [!NOTE]
    Soil moisture thresholds come from the soil profile, not from crop parameters.
    ``DEPNR < 3`` uses the straight-through estimator in ``SWEAF``. ``IAIRDU`` and
    ``IOX`` are switches.
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
        IDOS = Tensor(False, dtype=torch.bool)
        IDWS = Tensor(False, dtype=torch.bool)
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

        Args:
            day (datetime.date): The starting date of the simulation.
            kiosk (VariableKiosk): A container for registering and publishing
                (internal and external) state variables. See PCSE documentation for
                details.
            parvalues (ParameterProvider): A dictionary-like container holding
                all parameter sets (crop, soil, site) as key/value. The values are
                arrays or scalars. See PCSE documentation for details.
            shape (tuple | torch.Size | None): Target shape for the states and rates.
        """
        self.soil_profile = parvalues["soil_profile"]
        self._initialize_base(
            day,
            kiosk,
            parvalues,
            publish_rates=["EVWMX", "EVSMX", "TRAMX", "TRA", "TRALY", "RFTRA"],
            shape=shape,
        )

        # Pre-stack layer soil properties as tensors. ``stack`` keeps the
        # autograd history when a layer property is already a tensor.
        n_layers = len(self.soil_profile)
        self._n_layers = n_layers
        self._layer_smw = _stack_layer_property("SMW", self.soil_profile, self.dtype, self.device)
        self._layer_smfcf = _stack_layer_property("SMFCF", self.soil_profile, self.dtype, self.device)
        self._layer_sm0 = _stack_layer_property("SM0", self.soil_profile, self.dtype, self.device)
        self._layer_crairc = _stack_layer_property("CRAIRC", self.soil_profile, self.dtype, self.device)
        thicknesses = _stack_layer_property("Thickness", self.soil_profile, self.dtype, self.device)

        self._layer_depth_hi = torch.cumsum(thicknesses, dim=0)
        self._layer_depth_lo = self._layer_depth_hi - thicknesses

        # Internal DSOS tracker for layered oxygen-stress response
        self._dsos = torch.zeros(self.params_shape, dtype=self.dtype, device=self.device)

    def _rf_tramx_co2(self, drv: dict, et0: torch.Tensor) -> torch.Tensor:
        """Calculate CO2 reduction factor for TRAMX using CO2 from driver or parameters."""
        if "CO2" in drv and drv["CO2"] is not None:
            co2 = drv["CO2"]
        else:
            co2 = self.params.CO2
        return self.params.CO2TRATB(co2)

    def calc_rates(self, day: datetime.date, drv: dict):
        """Calculate daily evapotranspiration rates per soil layer with CO2 effects.

        Computes transpiration and stress factors for each soil layer based on root
        distribution and layer-specific soil moisture conditions.
        """
        p = self.params
        r = self.rates
        k = self.kiosk
        params_shape = self.params_shape

        dvs = _broadcast_to(k["DVS"], self.params_shape, dtype=self.dtype, device=self.device)

        # reduction factor for CO2 on TRAMX
        et0 = torch.as_tensor(drv["ET0"])
        rf_tramx_co2 = self._rf_tramx_co2(drv, et0)

        r.EVWMX, r.EVSMX, r.TRAMX, r.RFTRA, r.TRA, r.TRALY,r.RFWS, r.RFOS = _layered_rates(
            dvs,
            p.IAIRDU,
            p.IOX,
            k["RD"],
            self._n_layers,
            k["SM"],
            self._dsos,
            self._layer_smw,
            self._layer_smfcf,
            self._layer_depth_lo,
            self._layer_depth_hi,
            self._layer_sm0,
            self._layer_crairc,
            params_shape,
            k["LAI"],
            et0,
            torch.as_tensor(drv["E0"]),
            torch.as_tensor(drv["ES0"]),
            rf_tramx_co2,
            p.CFET,
            p.DEPNR,
            p.KDIFTB,
            self._epsilon,
        )

        # Counting stress days
        r.IDWS = r.RFWS < 1.0
        r.IDOS = r.RFOS < 1.0
        return r.TRA, r.TRAMX

    def __call__(self, day: datetime.date, drv: dict):
        """Callable interface for rate calculation."""
        return self.calc_rates(day, drv)

    def integrate(self, day: datetime.date, delt: float = 1.0) -> None:
        """Accumulate stress-day counters based on any layer experiencing stress."""
        rfws_stress = (self.rates.RFWS < 1.0).any(dim=0).to(dtype=self.dtype)
        rfos_stress = (self.rates.RFOS < 1.0).any(dim=0).to(dtype=self.dtype)
        self._IDWST = self._IDWST + rfws_stress
        self._IDOST = self._IDOST + rfos_stress

@kernel
def _layered_rates(
    dvs,
    IAIRDU,
    IOX,
    rd,
    n_layers,
    sm_layers,
    _dsos,
    _layer_smw,
    _layer_smfcf,
    _layer_depth_lo,
    _layer_depth_hi,
    _layer_sm0,
    _layer_crairc,
    params_shape,
    lai,
    et0,
    e0,
    es0,
    rf_tramx_co2,
    CFET,
    DEPNR,
    KDIFTB,
    _epsilon,
):# fmt: skip

    # If DVS < 0, the crop has not yet emerged, so we zero the rates using a mask
    # A mask (1 if DVS >= 0, 0 if DVS < 0)
    dvs_mask = (dvs >= 0.0).to(dtype=dvs.dtype)

    # Layered mask: (n_layers, *params_shape)
    dvs_mask_layers = dvs_mask.unsqueeze(0).expand(n_layers, *params_shape)

    # crop specific correction on potential transpiration rate
    et0_crop = torch.clamp(CFET * et0, min=0.0)

    # maximum evaporation and transpiration rates
    kglob = 0.75 * KDIFTB(dvs)
    ekl = torch.exp(-kglob * lai)
    EVWMX = dvs_mask * e0 * ekl
    EVSMX = dvs_mask * torch.clamp(es0 * ekl, min=0.0)
    TRAMX = dvs_mask * et0_crop * (1.0 - ekl) * rf_tramx_co2

    # Critical soil moisture
    swdep = SWEAF(et0_crop, DEPNR)

    # Layered soil moisture can be provided as:
    # - torch.Tensor with shape (n_layers, *params_shape)
    # - list/tuple of length n_layers, each element scalar or tensor
    tensor_kwargs = {"dtype": dvs.dtype, "device": dvs.device}
    if not isinstance(sm_layers, (torch.Tensor, list, tuple)):
        sm_layers = torch.as_tensor(sm_layers, **tensor_kwargs)
        if sm_layers.dim() == 1:
            sm_layers = list(sm_layers)  # per-layer scalars

    if isinstance(sm_layers, (list, tuple)):
        sm_layers_t = torch.stack(
            [_broadcast_to(sm_i, params_shape, **tensor_kwargs) for sm_i in sm_layers],
            dim=0,
        )
    else:
        sm_layers_t = sm_layers.to(**tensor_kwargs)

    if sm_layers_t.shape[0] != n_layers:
        raise ValueError(
            f"Layered evapotranspiration expects SM with {n_layers} layers, "
            f"got {sm_layers_t.shape[0]}."
        )

    # Reshape pre-stacked layer properties for broadcasting against
    # (n_layers, *params_shape) tensors: (n_layers,) → (n_layers, 1, 1, ...)
    ndim = len(params_shape)
    expand = (-1,) + (1,) * ndim
    layer_smw = _layer_smw.view(expand)
    layer_smfcf = _layer_smfcf.view(expand)
    depth_lo = _layer_depth_lo.view(expand)
    depth_hi = _layer_depth_hi.view(expand)

    # Vectorised RFWS across all layers: (n_layers, *params_shape)
    smcr = (1.0 - swdep) * (layer_smfcf - layer_smw) + layer_smw
    denom = torch.where(
        (smcr - layer_smw).abs() > _epsilon, smcr - layer_smw, _epsilon
    )
    # Reduction factor for transpiration in case of water shortage (RFWS)
    RFWS = dvs_mask_layers * torch.clamp(
        (sm_layers_t - layer_smw) / denom, min=0.0, max=1.0
    ) + (1.0 - dvs_mask_layers)

    # Vectorised root fraction across all layers: (n_layers, *params_shape)
    root_len = torch.clamp(torch.minimum(rd, depth_hi) - depth_lo, min=0.0)
    root_fraction = torch.where(rd > _epsilon, root_len / rd, 0.0)

    # Oxygen-stress reduction factor (sequential across layers due to
    # temporal _dsos accumulator that feeds forward between layers).
    # reduction in transpiration in case of oxygen shortage (RFOS)
    # for non-rice crops, and possibly deficient land drainage
    RFOS = torch.ones_like(RFWS)
    mask_ox = (IAIRDU == 0) & (IOX == 1)
    if bool(torch.any(mask_ox)):
        layer_sm0 = _layer_sm0.view(expand)
        layer_crairc = _layer_crairc.view(expand)
        for i in range(n_layers):
            smair = layer_sm0[i] - layer_crairc[i]
            _dsos = torch.where(
                sm_layers_t[i] >= smair,
                torch.clamp(_dsos + 1.0, max=4.0),
                0.0,
            )
            denom_ox = torch.where(
                layer_crairc[i].abs() > _epsilon, layer_crairc[i], _epsilon
            )
            rfosmx = torch.clamp((layer_sm0[i] - sm_layers_t[i]) / denom_ox, min=0.0, max=1.0)
            RFOS[i] = rfosmx + (1.0 - torch.clamp(_dsos, max=4.0) / 4.0) * (1.0 - rfosmx)

    # Transpiration per layer
    rftra = RFOS * RFWS
    TRALY = TRAMX * rftra * root_fraction
    TRA = TRALY.sum(dim=0)
    tramx = torch.clamp(TRAMX, min=_epsilon)
    RFTRA = torch.where(TRAMX > _epsilon, TRA / tramx, 1.0)

    # Pre-emergence: RFOS = 1.0
    RFOS = dvs_mask_layers * RFOS + (1.0 - dvs_mask_layers)
    return EVWMX, EVSMX, TRAMX, RFTRA, TRA, TRALY, RFWS, RFOS
