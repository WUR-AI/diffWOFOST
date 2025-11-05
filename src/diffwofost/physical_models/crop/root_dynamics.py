from datetime import datetime
import torch
from pcse.base import ParamTemplate
from pcse.base import RatesTemplate
from pcse.base import SimulationObject
from pcse.base import StatesTemplate
from pcse.base.parameter_providers import ParameterProvider
from pcse.base.variablekiosk import VariableKiosk
from pcse.base.weather import WeatherDataContainer
from pcse.decorators import prepare_rates
from pcse.decorators import prepare_states
from pcse.traitlets import Any
from diffwofost.physical_models.utils import AfgenTrait

DTYPE = torch.float64  # Default data type for tensors in this module


class WOFOST_Root_Dynamics(SimulationObject):
    """Root biomass dynamics and rooting depth.

    Root growth and root biomass dynamics in WOFOST are separate processes,
    with the only exception that root growth stops when no more biomass is sent
    to the root system.

    Root biomass increase results from the assimilates partitioned to
    the root system. Root death is defined as the current root biomass
    multiplied by a relative death rate (`RDRRTB`). The latter as a function
    of the development stage (`DVS`).

    Increase in root depth is a simple linear expansion over time until the
    maximum rooting depth (`RDM`) is reached.

    **Simulation parameters**

    | Name   | Description                                         | Type | Unit      |
    |--------|-----------------------------------------------------|------|-----------|
    | RDI    | Initial rooting depth                               | SCr  | cm        |
    | RRI    | Daily increase in rooting depth                     | SCr  | cm day⁻¹  |
    | RDMCR  | Maximum rooting depth of the crop                   | SCR  | cm        |
    | RDMSOL | Maximum rooting depth of the soil                   | SSo  | cm        |
    | TDWI   | Initial total crop dry weight                       | SCr  | kg ha⁻¹   |
    | IAIRDU | Presence of air ducts in the root (1) or not (0)    | SCr  | -         |
    | RDRRTB | Relative death rate of roots as a function of development stage | TCr | - |

    **State variables**

    | Name | Description                                                                  | Pbl | Unit     |
    |------|------------------------------------------------------------------------------|-----|----------|
    | RD   | Current rooting depth                                                        | Y   | cm       |
    | RDM  | Maximum attainable rooting depth at the minimum of the soil and crop maximum rooting depth | N | cm |
    | WRT  | Weight of living roots                                                       | Y   | kg ha⁻¹  |
    | DWRT | Weight of dead roots                                                         | N   | kg ha⁻¹  |
    | TWRT | Total weight of roots                                                        | Y   | kg ha⁻¹  |

    **Rate variables**

    | Name | Description                 | Pbl | Unit         |
    |------|-----------------------------|-----|--------------|
    | RR   | Growth rate root depth      | N   | cm           |
    | GRRT | Growth rate root biomass    | N   | kg ha⁻¹ d⁻¹  |
    | DRRT | Death rate root biomass     | N   | kg ha⁻¹ d⁻¹  |
    | GWRT | Net change in root biomass  | N   | kg ha⁻¹ d⁻¹  |

    **Signals send or handled**

    None

    **External dependencies:**

    | Name | Description               | Provided by      | Unit         |
    |------|---------------------------|------------------|--------------|
    | DVS  | Crop development stage    | DVS_Phenology    | -            |
    | DMI  | Total dry matter increase | CropSimulation   | kg ha⁻¹ d⁻¹  |
    | FR   | Fraction biomass to roots | DVS_Partitioning | -            |

    **Outputs:**

    | Name | Description             | Provided by      | Unit         |
    |------|-------------------------|------------------|--------------|
    | RD   | Current rooting depth   | Y                | cm           |
    | TWRT | Total weight of roots   | Y                | kg ha⁻¹      |

    **Gradient mapping (which parameters have a gradient):**

    | Output | Parameters influencing it |
    |--------|----------------------------|
    | RD     | RDI, RRI, RDMCR, RDMSOL    |
    | TWRT   | TDWI, RDRRTB               |

    [!] Notice that the gradient ∂TWRT/∂RDRRTB is zero.

    **IMPORTANT NOTICE**

    Currently root development is linear and depends only on the fraction of assimilates
    send to the roots (FR) and not on the amount of assimilates itself. This means that
    roots also grow through the winter when there is no assimilation due to low
    temperatures. There has been a discussion to change this behaviour and make root growth
    dependent on the assimilates send to the roots: so root growth stops when there are
    no assimilates available for growth.

    Finally, we decided not to change the root model and keep the original WOFOST approach
    because of the following reasons:
    - A dry top layer in the soil could create a large drought stress that reduces the
      assimilates to zero. In this situation the roots would not grow if dependent on the
      assimilates, while water is available in the zone just below the root zone. Therefore
      a dependency on the amount of assimilates could create model instability in dry
      conditions (e.g. Southern-Mediterranean, etc.).
    - Other solutions to alleviate the problem above were explored: only put this limitation
      after a certain development stage, putting a dependency on soil moisture levels in the
      unrooted soil compartment. All these solutions were found to introduce arbitrary
      parameters that have no clear explanation. Therefore all proposed solutions were discarded.

    We conclude that our current knowledge on root development is insufficient to propose a
    better and more biophysical approach to root development in WOFOST.
    """  # noqa: E501

    class Parameters(ParamTemplate):
        RDI = Any(default_value=[torch.tensor(-99.0, dtype=DTYPE)])
        RRI = Any(default_value=[torch.tensor(-99.0, dtype=DTYPE)])
        RDMCR = Any(default_value=[torch.tensor(-99.0, dtype=DTYPE)])
        RDMSOL = Any(default_value=[torch.tensor(-99.0, dtype=DTYPE)])
        TDWI = Any(default_value=[torch.tensor(-99.0, dtype=DTYPE)])
        IAIRDU = Any(default_value=[torch.tensor(-99.0, dtype=DTYPE)])
        RDRRTB = AfgenTrait()

    class RateVariables(RatesTemplate):
        RR = Any(default_value=torch.tensor(0.0, dtype=DTYPE))
        GRRT = Any(default_value=torch.tensor(0.0, dtype=DTYPE))
        DRRT = Any(default_value=torch.tensor(0.0, dtype=DTYPE))
        GWRT = Any(default_value=torch.tensor(0.0, dtype=DTYPE))

    class StateVariables(StatesTemplate):
        RD = Any(default_value=[torch.tensor(-99.0, dtype=DTYPE)])
        RDM = Any(default_value=[torch.tensor(-99.0, dtype=DTYPE)])
        WRT = Any(default_value=[torch.tensor(-99.0, dtype=DTYPE)])
        DWRT = Any(default_value=[torch.tensor(-99.0, dtype=DTYPE)])
        TWRT = Any(default_value=[torch.tensor(-99.0, dtype=DTYPE)])

    def initialize(
        self, day: datetime.date, kiosk: VariableKiosk, parvalues: ParameterProvider
    ) -> None:
        """Initialize the model.

        Args:
            day (datetime.date): The starting date of the simulation.
            kiosk (VariableKiosk): A container for registering and publishing
                (internal and external) state variables. See PCSE documentation for
                details.
            parvalues (ParameterProvider): A dictionary-like container holding
                all parameter sets (crop, soil, site) as key/value. The values are
                arrays or scalars. See PCSE documentation for details.
        """
        self.params = self.Parameters(parvalues)
        self.rates = self.RateVariables(kiosk, publish=["DRRT", "GRRT"])
        self.kiosk = kiosk

        # INITIAL STATES
        params = self.params

        # Initial root depth states
        rdmax = torch.max(params.RDI, torch.min(params.RDMCR, params.RDMSOL))
        RDM = rdmax
        RD = params.RDI

        # initial root biomass states
        WRT = params.TDWI * self.kiosk.FR
        DWRT = torch.tensor(0.0, dtype=DTYPE)
        TWRT = WRT + DWRT

        self.states = self.StateVariables(
            kiosk, publish=["RD", "WRT", "TWRT"], RD=RD, RDM=RDM, WRT=WRT, DWRT=DWRT, TWRT=TWRT
        )

    @prepare_rates
    def calc_rates(self, day: datetime.date = None, drv: WeatherDataContainer = None) -> None:
        """Calculate the rates of change of the state variables.

        Args:
            day (datetime.date, optional): The current date of the simulation.
            drv (WeatherDataContainer, optional): A dictionary-like container holding
                weather data elements as key/value. The values are
                arrays or scalars. See PCSE documentation for details.
        """
        p = self.params
        r = self.rates
        s = self.states
        k = self.kiosk

        # If DVS < 0, the crop has not yet emerged, so we zerofy the rates using mask
        # Make a mask (0 if DVS < 0, 1 if DVS >= 0)
        DVS = torch.as_tensor(k["DVS"], dtype=DTYPE)
        dvs_mask = (DVS >= 0).to(dtype=DTYPE)

        # Increase in root biomass
        r.GRRT = dvs_mask * k.FR * k.DMI
        r.DRRT = dvs_mask * s.WRT * p.RDRRTB(k.DVS)
        r.GWRT = r.GRRT - r.DRRT

        # Increase in root depth
        r.RR = dvs_mask * torch.min((s.RDM - s.RD), p.RRI)

        # Do not let the roots growth if partioning to the roots
        # (variable FR) is zero.
        FR = torch.as_tensor(k["FR"], dtype=DTYPE)
        mask = (FR > 0.0).to(dtype=DTYPE)
        r.RR = r.RR * mask * dvs_mask

    @prepare_states
    def integrate(self, day: datetime.date = None, delt=1.0) -> None:
        """Integrate the state variables using the rates of change.

        Args:
            day (datetime.date, optional): The current date of the simulation.
            delt (float, optional): The time step for integration. Defaults to 1.0.
        """
        rates = self.rates
        states = self.states

        # Dry weight of living roots
        states.WRT = states.WRT + rates.GWRT

        # Dry weight of dead roots
        states.DWRT = states.DWRT + rates.DRRT

        # Total weight dry + living roots
        states.TWRT = states.WRT + states.DWRT

        # New root depth
        states.RD = states.RD + rates.RR
