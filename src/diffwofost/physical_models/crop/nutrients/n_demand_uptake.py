"""Crop nitrogen demand, soil uptake and translocation to storage organs."""

import datetime
from collections import namedtuple
import torch
from pcse.base import SimulationObject
from diffwofost.physical_models.base import TensorParamTemplate
from diffwofost.physical_models.base import TensorRatesTemplate
from diffwofost.physical_models.base import TensorStatesTemplate
from diffwofost.physical_models.traitlets import Tensor
from diffwofost.physical_models.utils import AfgenTrait

MaxNutrientConcentrations = namedtuple(
    "MaxNutrientConcentrations", ["NMAXLV", "NMAXST", "NMAXRT", "NMAXSO"]
)


class N_Demand_Uptake(SimulationObject):
    """Calculates the crop N demand and its uptake from the soil.

    Crop N demand is calculated as the difference between the actual N
    (kg N per kg biomass) in the vegetative plant organs (leaves, stems and roots)
    and the maximum N concentration for each organ. N uptake is then estimated as
    the minimum of supply from the soil and demand from the crop.

    Nitrogen fixation (leguminous plants) is calculated by assuming that a fixed
    fraction of the daily N demand is supplied by nitrogen fixation. The remaining
    part has to be supplied by the soil.

    The N demand of the storage organs is calculated in a somewhat different way
    because it is assumed that the demand from the storage organs is fulfilled by
    translocation of N from the leaves, stems and roots. Therefore the uptake of
    the storage organs is calculated as the minimum of the daily translocatable N
    supply and the demand from the storage organs. Below a transpiration reduction
    of 0.01 the crop takes up no nitrogen.

    **Simulation parameters**

    | Name        | Description                                            | Unit                  |
    |-------------|--------------------------------------------------------|-----------------------|
    | NMAXLV_TB   | Maximum N concentration in leaves as function of DVS  | kg N kg-1 dry biomass |
    | NMAXRT_FR   | Maximum N concentration in roots as fraction of leaves| -                     |
    | NMAXST_FR   | Maximum N concentration in stems as fraction of leaves| -                     |
    | NMAXSO      | Maximum N concentration in storage organs             | kg N kg-1 dry biomass |
    | TCNT        | Time coefficient for N translocation to storage organs| days                  |
    | NFIX_FR     | Fraction of N uptake supplied by biological fixation  | kg N kg-1 dry biomass |
    | RNUPTAKEMAX | Maximum rate of N uptake                              | kg N ha-1 d-1         |
    | DVS_N_TRANSL| Development stage at which translocation starts        | -                     |
    | NRESIDLV    | Residual N fraction in leaves                          | kg N kg-1 dry biomass |
    | NRESIDST    | Residual N fraction in stems                           | kg N kg-1 dry biomass |
    | NRESIDRT    | Residual N fraction in roots                           | kg N kg-1 dry biomass |

    **State variables**

    | Name              | Description                                      | Pbl | Unit      |
    |-------------------|--------------------------------------------------|-----|-----------|
    | Ntranslocatable   | Total N that can move to the storage organs      | Y   | kg N ha-1 |
    | NtranslocatableLV | Translocatable N in living leaves                | N   | kg N ha-1 |
    | NtranslocatableST | Translocatable N in living stems                 | N   | kg N ha-1 |
    | NtranslocatableRT | Translocatable N in living roots                 | N   | kg N ha-1 |

    **Rate variables**

    | Name              | Description                                      | Pbl | Unit          |
    |-------------------|--------------------------------------------------|-----|---------------|
    | RNuptakeLV        | Rate of N uptake in leaves                       | Y   | kg N ha-1 d-1 |
    | RNuptakeST        | Rate of N uptake in stems                        | Y   | kg N ha-1 d-1 |
    | RNuptakeRT        | Rate of N uptake in roots                        | Y   | kg N ha-1 d-1 |
    | RNuptakeSO        | Rate of N uptake in storage organs               | Y   | kg N ha-1 d-1 |
    | RNuptake          | Total rate of N uptake                           | Y   | kg N ha-1 d-1 |
    | RNfixation        | Rate of N fixation                               | Y   | kg N ha-1 d-1 |
    | RNtranslocation   | Total N translocation to storage organs          | Y   | kg N ha-1 d-1 |
    | RNtranslocationLV | N translocation rate from leaves                 | Y   | kg N ha-1 d-1 |
    | RNtranslocationST | N translocation rate from stems                  | Y   | kg N ha-1 d-1 |
    | RNtranslocationRT | N translocation rate from roots                  | Y   | kg N ha-1 d-1 |
    | NdemandLV         | N demand in living leaves                        | N   | kg N ha-1     |
    | NdemandST         | N demand in living stems                         | N   | kg N ha-1     |
    | NdemandRT         | N demand in living roots                         | N   | kg N ha-1     |
    | NdemandSO         | N demand in storage organs                       | N   | kg N ha-1     |
    | Ndemand           | Total crop N demand                              | N   | kg N ha-1     |

    **Signals sent or handled**

    None

    **External dependencies**

    | Name    | Description                         | Provided by        | Unit      |
    |---------|-------------------------------------|--------------------|-----------|
    | DVS     | Crop development stage              | DVS_Phenology      | -         |
    | RFTRA   | Transpiration reduction factor      | Evapotranspiration | -         |
    | NAVAIL  | Total available N from soil         | N_Soil_Dynamics    | kg ha-1   |
    | WLV     | Weight of living leaves             | WOFOST_Leaf_Dynamics | kg ha-1 |
    | WST     | Weight of living stems              | WOFOST_Stem_Dynamics | kg ha-1 |
    | WRT     | Weight of living roots              | WOFOST_Root_Dynamics | kg ha-1 |
    | WSO     | Weight of storage organs            | WOFOST_Storage_Organ_Dynamics | kg ha-1 |
    | NamountLV | N amount in living leaves         | N_Crop_Dynamics    | kg ha-1   |
    | NamountST | N amount in living stems          | N_Crop_Dynamics    | kg ha-1   |
    | NamountRT | N amount in living roots          | N_Crop_Dynamics    | kg ha-1   |
    | NamountSO | N amount in storage organs        | N_Crop_Dynamics    | kg ha-1   |
    | GRLV    | Growth rate of living leaves        | WOFOST_Leaf_Dynamics | kg ha-1 d-1 |
    | GRST    | Growth rate of living stems         | WOFOST_Stem_Dynamics | kg ha-1 d-1 |
    | GRRT    | Growth rate of living roots         | WOFOST_Root_Dynamics | kg ha-1 d-1 |
    | GRSO    | Growth rate of storage organs       | WOFOST_Storage_Organ_Dynamics | kg ha-1 d-1 |

    **Gradient mapping (which parameters have a gradient):**

    | Output           | Parameters influencing it                                |
    |------------------|----------------------------------------------------------|
    | Ndemand, RNuptake| NMAXLV_TB, NMAXST_FR, NMAXRT_FR, NMAXSO, RNUPTAKEMAX     |
    | RNfixation       | NFIX_FR, NMAXLV_TB, NMAXST_FR, NMAXRT_FR, NMAXSO         |
    | RNtranslocation  | TCNT, NMAXSO, NRESIDLV, NRESIDST, NRESIDRT               |

    [!NOTE]
    ``DVS_N_TRANSL`` is a hard development-stage switch, so its gradient is zero.
    The residual nitrogen fractions still have a gradient once translocation has started.
    The ``RFTRA > 0.01`` cutoff is a fixed constant, not a parameter.
    """

    class Parameters(TensorParamTemplate):
        NMAXLV_TB = AfgenTrait()
        DVS_N_TRANSL = Tensor(-99.0)
        NMAXRT_FR = Tensor(-99.0)
        NMAXST_FR = Tensor(-99.0)
        NMAXSO = Tensor(-99.0)
        TCNT = Tensor(-99.0)
        NFIX_FR = Tensor(-99.0)
        RNUPTAKEMAX = Tensor(-99.0)
        NRESIDLV = Tensor(-99.0)
        NRESIDST = Tensor(-99.0)
        NRESIDRT = Tensor(-99.0)

    class RateVariables(TensorRatesTemplate):
        RNtranslocationLV = Tensor(0.0)
        RNtranslocationST = Tensor(0.0)
        RNtranslocationRT = Tensor(0.0)
        RNtranslocation = Tensor(0.0)
        RNuptakeLV = Tensor(0.0)
        RNuptakeST = Tensor(0.0)
        RNuptakeRT = Tensor(0.0)
        RNuptakeSO = Tensor(0.0)
        RNuptake = Tensor(0.0)
        RNfixation = Tensor(0.0)
        NdemandLV = Tensor(0.0)
        NdemandST = Tensor(0.0)
        NdemandRT = Tensor(0.0)
        NdemandSO = Tensor(0.0)
        Ndemand = Tensor(0.0)

    class StateVariables(TensorStatesTemplate):
        NtranslocatableLV = Tensor(-99.0)
        NtranslocatableST = Tensor(-99.0)
        NtranslocatableRT = Tensor(-99.0)
        Ntranslocatable = Tensor(-99.0)

    def initialize(self, day, kiosk, parvalues, shape=None):
        """Publish uptake and translocation rates for the crop nitrogen balance."""
        self.params = self.Parameters(parvalues, shape=shape)
        self.kiosk = kiosk
        self.rates = self.RateVariables(
            kiosk,
            publish=[
                "RNtranslocationLV",
                "RNtranslocationST",
                "RNtranslocationRT",
                "RNtranslocation",
                "RNuptakeLV",
                "RNuptakeST",
                "RNuptakeRT",
                "RNuptakeSO",
                "RNuptake",
                "RNfixation",
            ],
            shape=shape,
        )
        self.states = self.StateVariables(
            kiosk,
            NtranslocatableLV=0.0,
            NtranslocatableST=0.0,
            NtranslocatableRT=0.0,
            Ntranslocatable=0.0,
            publish=["Ntranslocatable"],
            shape=shape,
        )

    def calc_rates(self, day: datetime.date, drv):
        """Demand, fixation, translocation and soil uptake for one day."""
        rates = self.rates
        params = self.params
        kiosk = self.kiosk
        states = self.states
        delt = 1.0
        maximum = self._maximum_concentrations()
        nutrient_limit = torch.where(kiosk["RFTRA"] > 0.01, 1.0, 0.0)

        rates.NdemandLV = _organ_demand(
            maximum.NMAXLV, kiosk["WLV"], kiosk["NamountLV"], kiosk["GRLV"], delt
        )
        rates.NdemandST = _organ_demand(
            maximum.NMAXST, kiosk["WST"], kiosk["NamountST"], kiosk["GRST"], delt
        )
        rates.NdemandRT = _organ_demand(
            maximum.NMAXRT, kiosk["WRT"], kiosk["NamountRT"], kiosk["GRRT"], delt
        )
        rates.NdemandSO = _organ_demand(
            maximum.NMAXSO, kiosk["WSO"], kiosk["NamountSO"], kiosk["GRSO"], delt
        )
        rates.Ndemand = rates.NdemandLV + rates.NdemandST + rates.NdemandRT + rates.NdemandSO
        rates.RNfixation = torch.clamp(params.NFIX_FR * rates.Ndemand, min=0.0) * nutrient_limit

        translocating = kiosk["DVS"] >= params.DVS_N_TRANSL
        states.NtranslocatableLV = torch.where(
            translocating,
            torch.clamp(kiosk["NamountLV"] - kiosk["WLV"] * params.NRESIDLV, min=0.0),
            torch.zeros_like(kiosk["WLV"]),
        )
        states.NtranslocatableST = torch.where(
            translocating,
            torch.clamp(kiosk["NamountST"] - kiosk["WST"] * params.NRESIDST, min=0.0),
            torch.zeros_like(kiosk["WST"]),
        )
        states.NtranslocatableRT = torch.where(
            translocating,
            torch.clamp(kiosk["NamountRT"] - kiosk["WRT"] * params.NRESIDRT, min=0.0),
            torch.zeros_like(kiosk["WRT"]),
        )
        states.Ntranslocatable = (
            states.NtranslocatableLV + states.NtranslocatableST + states.NtranslocatableRT
        )
        rates.RNtranslocation = torch.minimum(
            rates.NdemandSO / delt, states.Ntranslocatable / params.TCNT
        )
        # PCSE divides by the total only when it is not zero. Each organ pool is
        # clamped at zero, so the total cannot be negative and ``> 0`` matches
        # that test. ``torch.where`` still evaluates the division, so the
        # denominator is clamped; above the floor it equals the total.
        total_translocatable = states.Ntranslocatable
        share_ready = total_translocatable > 0
        safe_total = torch.clamp(total_translocatable, min=1e-12)
        no_share = torch.zeros_like(rates.RNtranslocation)
        rates.RNtranslocationLV = torch.where(
            share_ready,
            rates.RNtranslocation * states.NtranslocatableLV / safe_total,
            no_share,
        )
        rates.RNtranslocationST = torch.where(
            share_ready,
            rates.RNtranslocation * states.NtranslocatableST / safe_total,
            no_share,
        )
        rates.RNtranslocationRT = torch.where(
            share_ready,
            rates.RNtranslocation * states.NtranslocatableRT / safe_total,
            no_share,
        )

        soil_limited = torch.minimum(
            torch.clamp(rates.Ndemand - rates.RNfixation, min=0.0), kiosk["NAVAIL"]
        )
        # PCSE: max(0, min(demand - fixation, NAVAIL, RNUPTAKEMAX)).
        rates.RNuptake = (
            torch.clamp(torch.minimum(soil_limited, params.RNUPTAKEMAX), min=0.0) * nutrient_limit
        )
        supply = rates.RNuptake + rates.RNfixation
        has_demand = rates.Ndemand > 0
        safe_demand = torch.clamp(rates.Ndemand, min=1e-12)
        rates.RNuptakeLV = _organ_uptake(
            has_demand,
            rates.NdemandLV / delt + rates.RNtranslocationLV,
            supply,
            rates.NdemandLV,
            safe_demand,
        )
        rates.RNuptakeST = _organ_uptake(
            has_demand,
            rates.NdemandST / delt + rates.RNtranslocationST,
            supply,
            rates.NdemandST,
            safe_demand,
        )
        rates.RNuptakeRT = _organ_uptake(
            has_demand,
            rates.NdemandRT / delt + rates.RNtranslocationRT,
            supply,
            rates.NdemandRT,
            safe_demand,
        )
        rates.RNuptakeSO = _organ_uptake(
            has_demand,
            rates.NdemandSO / delt - rates.RNtranslocation,
            supply,
            rates.NdemandSO,
            safe_demand,
        )

    def integrate(self, day: datetime.date, delt=1.0):
        """Translocatable nitrogen is recomputed every day in ``calc_rates``."""
        return

    def _maximum_concentrations(self):
        nmax_leaf = self.params.NMAXLV_TB(self.kiosk["DVS"])
        return MaxNutrientConcentrations(
            NMAXLV=nmax_leaf,
            NMAXST=self.params.NMAXST_FR * nmax_leaf,
            NMAXRT=self.params.NMAXRT_FR * nmax_leaf,
            NMAXSO=self.params.NMAXSO,
        )


def _organ_demand(maximum, weight, amount, growth, delt):
    replenish = torch.clamp(maximum * weight - amount, min=0.0)
    new_growth = torch.clamp(growth * maximum, min=0.0) * delt
    return replenish + new_growth


def _organ_uptake(has_demand, requested, supply, organ_demand, total_demand):
    share = supply * requested / total_demand
    taken = torch.clamp(torch.minimum(requested, share), min=0.0)
    return torch.where(has_demand, taken, torch.zeros_like(taken))
