"""Book-keeping of nitrogen in leaves, stems, roots and storage organs."""

import datetime
import torch
from pcse import exceptions as exc
from pcse.base import SimulationObject
from pcse.traitlets import Instance
from diffwofost.physical_models.base import TensorParamTemplate
from diffwofost.physical_models.base import TensorRatesTemplate
from diffwofost.physical_models.base import TensorStatesTemplate
from diffwofost.physical_models.crop.nutrients.n_demand_uptake import N_Demand_Uptake
from diffwofost.physical_models.traitlets import Tensor
from diffwofost.physical_models.utils import AfgenTrait


class N_Crop_Dynamics(SimulationObject):
    """Book-keeping of nitrogen in leaves, stems, roots and storage organs.

    This class keeps the nitrogen amount of each living organ. Daily uptake
    and translocation come from ``N_Demand_Uptake``. Nitrogen leaves an organ
    when that organ dies.

    **Simulation parameters**

    | Name      | Description                                           | Unit                  |
    |-----------|-------------------------------------------------------|-----------------------|
    | NMAXLV_TB | Maximum N concentration in leaves as function of DVS | kg N kg-1 dry biomass |
    | NMAXRT_FR | Maximum N in roots as a fraction of the leaf maximum | -                     |
    | NMAXST_FR | Maximum N in stems as a fraction of the leaf maximum | -                     |
    | NRESIDLV  | Residual N fraction in leaves                         | kg N kg-1 dry biomass |
    | NRESIDRT  | Residual N fraction in roots                          | kg N kg-1 dry biomass |
    | NRESIDST  | Residual N fraction in stems                          | kg N kg-1 dry biomass |

    **State variables**

    | Name         | Description                              | Unit      |
    |--------------|------------------------------------------|-----------|
    | NamountLV    | Actual N amount in living leaves         | kg N ha-1 |
    | NamountST    | Actual N amount in living stems          | kg N ha-1 |
    | NamountRT    | Actual N amount in living roots          | kg N ha-1 |
    | NamountSO    | Actual N amount in storage organs        | kg N ha-1 |
    | NuptakeTotal | Total N absorbed from the soil           | kg N ha-1 |
    | NfixTotal    | Total N supplied by biological fixation  | kg N ha-1 |
    | NlossesTotal | Total N lost with dying biomass          | kg N ha-1 |

    **Rate variables**

    | Name       | Description                         | Unit          |
    |------------|-------------------------------------|---------------|
    | RNamountLV | Net change of N in leaves           | kg N ha-1 d-1 |
    | RNamountST | Net change of N in stems            | kg N ha-1 d-1 |
    | RNamountRT | Net change of N in roots            | kg N ha-1 d-1 |
    | RNdeathLV  | N loss with dying leaves            | kg N ha-1 d-1 |
    | RNdeathST  | N loss with dying stems             | kg N ha-1 d-1 |
    | RNdeathRT  | N loss with dying roots             | kg N ha-1 d-1 |
    | RNloss     | Total N loss due to senescence      | kg N ha-1 d-1 |

    **Signals sent or handled**

    None

    **External dependencies**

    | Name | Description                 | Provided by          | Unit        |
    |------|-----------------------------|----------------------|-------------|
    | DVS  | Crop development stage      | DVS_Phenology        | -           |
    | WLV  | Dry weight of living leaves | WOFOST_Leaf_Dynamics | kg ha-1     |
    | WST  | Dry weight of living stems  | WOFOST_Stem_Dynamics | kg ha-1     |
    | WRT  | Dry weight of living roots  | WOFOST_Root_Dynamics | kg ha-1     |
    | DRLV | Death rate of leaves        | WOFOST_Leaf_Dynamics | kg ha-1 d-1 |
    | DRST | Death rate of stems         | WOFOST_Stem_Dynamics | kg ha-1 d-1 |
    | DRRT | Death rate of roots         | WOFOST_Root_Dynamics | kg ha-1 d-1 |

    **Gradient mapping (which parameters have a gradient):**

    | Output       | Parameters influencing it                    |
    |--------------|----------------------------------------------|
    | NamountLV    | NMAXLV_TB, NRESIDLV                          |
    | NamountST    | NMAXLV_TB, NMAXST_FR, NRESIDST               |
    | NamountRT    | NMAXLV_TB, NMAXRT_FR, NRESIDRT               |
    | NamountSO    | NMAXLV_TB                                    |
    | NuptakeTotal | the parameters of ``N_Demand_Uptake``        |

    [!NOTE]
    The daily change of each pool follows the uptake and translocation rates.
    ``NRESID*`` changes how much nitrogen can leave an organ, not the initial amount.
    The initial amount is the maximum concentration times the initial biomass.
    """

    demand_uptake = Instance(SimulationObject)
    NamountLVI = None
    NamountSTI = None
    NamountRTI = None
    NamountSOI = None

    class Parameters(TensorParamTemplate):
        NMAXLV_TB = AfgenTrait()
        NMAXST_FR = Tensor(-99.0)
        NMAXRT_FR = Tensor(-99.0)
        NRESIDLV = Tensor(-99.0)
        NRESIDST = Tensor(-99.0)
        NRESIDRT = Tensor(-99.0)

    class StateVariables(TensorStatesTemplate):
        NamountLV = Tensor(-99.0)
        NamountST = Tensor(-99.0)
        NamountSO = Tensor(-99.0)
        NamountRT = Tensor(-99.0)
        NuptakeTotal = Tensor(-99.0)
        NfixTotal = Tensor(-99.0)
        NlossesTotal = Tensor(-99.0)

    class RateVariables(TensorRatesTemplate):
        RNamountLV = Tensor(0.0)
        RNamountST = Tensor(0.0)
        RNamountRT = Tensor(0.0)
        RNamountSO = Tensor(0.0)
        RNdeathLV = Tensor(0.0)
        RNdeathST = Tensor(0.0)
        RNdeathRT = Tensor(0.0)
        RNloss = Tensor(0.0)

    def initialize(self, day, kiosk, parvalues, shape=None):
        """Start each organ at its maximum nitrogen concentration."""
        self.params = self.Parameters(parvalues, shape=shape)
        self.rates = self.RateVariables(kiosk, shape=shape)
        self.kiosk = kiosk
        self.demand_uptake = N_Demand_Uptake(day, kiosk, parvalues, shape=shape)
        params = self.params
        nmax_leaf = params.NMAXLV_TB(kiosk["DVS"])
        self.NamountLVI = kiosk["WLV"] * nmax_leaf
        self.NamountSTI = kiosk["WST"] * nmax_leaf * params.NMAXST_FR
        self.NamountRTI = kiosk["WRT"] * nmax_leaf * params.NMAXRT_FR
        self.NamountSOI = torch.zeros_like(self.NamountLVI)
        self.states = self.StateVariables(
            kiosk,
            publish=["NamountLV", "NamountST", "NamountRT", "NamountSO", "NuptakeTotal"],
            NamountLV=self.NamountLVI,
            NamountST=self.NamountSTI,
            NamountRT=self.NamountRTI,
            NamountSO=self.NamountSOI,
            NuptakeTotal=0.0,
            NfixTotal=0.0,
            NlossesTotal=0.0,
            shape=shape,
        )

    def calc_rates(self, day: datetime.date, drv):
        """Uptake minus translocation minus nitrogen lost with dying biomass."""
        self.demand_uptake.calc_rates(day, drv)
        rates = self.rates
        states = self.states
        kiosk = self.kiosk
        # Same unused-branch guard as N_Stress: the selected rate is still 0.
        leaf_weight = torch.clamp(kiosk["WLV"], min=1e-12)
        stem_weight = torch.clamp(kiosk["WST"], min=1e-12)
        root_weight = torch.clamp(kiosk["WRT"], min=1e-12)
        rates.RNdeathLV = torch.where(
            kiosk["WLV"] > 0, states.NamountLV / leaf_weight * kiosk["DRLV"], 0.0
        )
        rates.RNdeathST = torch.where(
            kiosk["WST"] > 0, states.NamountST / stem_weight * kiosk["DRST"], 0.0
        )
        rates.RNdeathRT = torch.where(
            kiosk["WRT"] > 0, states.NamountRT / root_weight * kiosk["DRRT"], 0.0
        )
        rates.RNamountLV = kiosk["RNuptakeLV"] - kiosk["RNtranslocationLV"] - rates.RNdeathLV
        rates.RNamountST = kiosk["RNuptakeST"] - kiosk["RNtranslocationST"] - rates.RNdeathST
        rates.RNamountRT = kiosk["RNuptakeRT"] - kiosk["RNtranslocationRT"] - rates.RNdeathRT
        rates.RNamountSO = kiosk["RNuptakeSO"] + kiosk["RNtranslocation"]
        rates.RNloss = rates.RNdeathLV + rates.RNdeathST + rates.RNdeathRT
        self._check_n_balance(day)

    def integrate(self, day: datetime.date, delt=1.0):
        """Add the daily nitrogen rates to the organ pools."""
        states = self.states
        rates = self.rates
        kiosk = self.kiosk
        states.NamountLV = states.NamountLV + rates.RNamountLV
        states.NamountST = states.NamountST + rates.RNamountST
        states.NamountRT = states.NamountRT + rates.RNamountRT
        states.NamountSO = states.NamountSO + rates.RNamountSO
        self.demand_uptake.integrate(day, delt)
        states.NuptakeTotal = states.NuptakeTotal + kiosk["RNuptake"]
        states.NfixTotal = states.NfixTotal + kiosk["RNfixation"]
        states.NlossesTotal = states.NlossesTotal + rates.RNloss

    def _check_n_balance(self, day):
        states = self.states
        checksum = (
            states.NuptakeTotal
            + states.NfixTotal
            + self.NamountLVI
            + self.NamountSTI
            + self.NamountRTI
            + self.NamountSOI
            - (
                states.NamountLV
                + states.NamountST
                + states.NamountRT
                + states.NamountSO
                + states.NlossesTotal
            )
        )
        if torch.any(torch.abs(checksum) >= 1.0):
            msg = f"N flows not balanced on day {day}. Checksum: {checksum}"
            raise exc.NutrientBalanceError(msg)
