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
    """Nitrogen demand of each organ and uptake limited by soil supply.

    Uptake is the minimum of crop demand, soil mineral nitrogen and
    ``RNUPTAKEMAX``. Below a transpiration reduction of 0.01 the crop takes
    up no nitrogen. Storage organs are supplied by translocation from leaves,
    stems and roots once development passes ``DVS_N_TRANSL``.

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
