import torch
from pcse.base import ParamTemplate
from pcse.base import RatesTemplate
from pcse.base import SimulationObject
from pcse.base import StatesTemplate
from pcse.decorators import prepare_rates
from pcse.decorators import prepare_states
from pcse.traitlets import Float
from pcse.traitlets import Any
from pcse.util import AfgenTrait

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

    =======  ============================================= =======  ============
     Name     Description                                   Type     Unit
    =======  ============================================= =======  ============
    RDI      Initial rooting depth                          SCr      cm
    RRI      Daily increase in rooting depth                SCr      |cm day-1|
    RDMCR    Maximum rooting depth of the crop              SCR      cm
    RDMSOL   Maximum rooting depth of the soil              SSo      cm
    TDWI     Initial total crop dry weight                  SCr      |kg ha-1|
    IAIRDU   Presence of air ducts in the root (1) or       SCr      -
             not (0)
    RDRRTB   Relative death rate of roots as a function     TCr      -
             of development stage
    =======  ============================================= =======  ============


    **State variables**

    =======  ================================================= ==== ============
     Name     Description                                      Pbl      Unit
    =======  ================================================= ==== ============
    RD       Current rooting depth                              Y     cm
    RDM      Maximum attainable rooting depth at the minimum    N     cm
             of the soil and crop maximum rooting depth
    WRT      Weight of living roots                             Y     |kg ha-1|
    DWRT     Weight of dead roots                               N     |kg ha-1|
    TWRT     Total weight of roots                              Y     |kg ha-1|
    =======  ================================================= ==== ============

    **Rate variables**

    =======  ================================================= ==== ============
     Name     Description                                      Pbl      Unit
    =======  ================================================= ==== ============
    RR       Growth rate root depth                             N    cm
    GRRT     Growth rate root biomass                           N   |kg ha-1 d-1|
    DRRT     Death rate root biomass                            N   |kg ha-1 d-1|
    GWRT     Net change in root biomass                         N   |kg ha-1 d-1|
    =======  ================================================= ==== ============

    **Signals send or handled**

    None

    **External dependencies:**

    =======  =================================== =================  ============
     Name     Description                         Provided by         Unit
    =======  =================================== =================  ============
    DVS      Crop development stage              DVS_Phenology       -
    DMI      Total dry matter                    CropSimulation     |kg ha-1 d-1|
             increase
    FR       Fraction biomass to roots           DVS_Partitioning    -
    =======  =================================== =================  ============

    **Outputs:**
    - RD, TWRT

    """
    """
    IMPORTANT NOTICE
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
    """

    class Parameters(ParamTemplate):
        RDI    = Float(-99.)
        RRI    = Float(-99.)
        RDMCR  = Float(-99.)
        RDMSOL = Float(-99.)
        TDWI   = Float(-99.)
        IAIRDU = Float(-99)
        RDRRTB = AfgenTrait() # FIXEME

    class RateVariables(RatesTemplate):
        RR   = Float(-99.)
        GRRT = Float(-99.)
        DRRT = Float(-99.)
        GWRT = Float(-99.)

    class StateVariables(StatesTemplate):
        RD   = Float(-99.)
        RDM  = Float(-99.)
        WRT  = Float(-99.)
        DWRT = Float(-99.)
        TWRT = Float(-99.)

    def initialize(self, day, kiosk, parvalues):
        """
        :param day: start date of the simulation
        :param kiosk: variable kiosk of this PCSE  instance
        :param parvalues: `ParameterProvider` object providing parameters as
                key/value pairs
        """

        self.params = self.Parameters(parvalues)
        self.rates = self.RateVariables(kiosk, publish=["DRRT", "GRRT"])
        self.kiosk = kiosk

        # INITIAL STATES
        params = self.params
        # Initial root depth states
        rdmax = max(params.RDI, min(params.RDMCR, params.RDMSOL))
        RDM = rdmax
        RD = params.RDI
        # initial root biomass states
        WRT  = params.TDWI * self.kiosk.FR
        DWRT = 0.
        TWRT = WRT + DWRT

        self.states = self.StateVariables(kiosk, publish=["RD","WRT", "TWRT"],
                                          RD=RD, RDM=RDM, WRT=WRT, DWRT=DWRT,
                                          TWRT=TWRT)

    @prepare_rates
    def calc_rates(self, day, drv):
        p = self.params
        r = self.rates
        s = self.states
        k = self.kiosk

        # Increase in root biomass
        r.GRRT = k.FR * k.DMI
        r.DRRT = s.WRT * p.RDRRTB(k.DVS)
        r.GWRT = r.GRRT - r.DRRT

        # Increase in root depth
        r.RR = min((s.RDM - s.RD), p.RRI)
        # Do not let the roots growth if partioning to the roots
        # (variable FR) is zero.
        if k.FR == 0.:
            r.RR = 0.

    @prepare_states
    def integrate(self, day, delt=1.0):
        rates = self.rates
        states = self.states

        # Dry weight of living roots
        states.WRT += rates.GWRT
        # Dry weight of dead roots
        states.DWRT += rates.DRRT
        # Total weight dry + living roots
        states.TWRT = states.WRT + states.DWRT

        # New root depth
        states.RD += rates.RR


    @prepare_states
    def _set_variable_WRT(self, nWRT):
        """Updates the value of WRT to to the new value provided as input.

        Related state variables will be updated as well and the increments
        to all adjusted state variables will be returned as a dict.
        """
        states = self.states

        # Store old values of states
        oWRT = states.WRT
        oTWRT = states.TWRT

        # Apply new root weight and adjust total (dead + live) root weight
        states.WRT = nWRT
        states.TWRT = states.WRT + states.DWRT

        increments = {"WRT": states.WRT - oWRT,
                      "TWLRT": states.TWRT - oTWRT}
        return increments
