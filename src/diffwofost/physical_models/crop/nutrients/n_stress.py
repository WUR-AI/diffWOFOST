"""Nitrogen stress factors for WOFOST 8.1."""

import datetime
import torch
from pcse.base import SimulationObject
from diffwofost.physical_models.base import TensorParamTemplate
from diffwofost.physical_models.base import TensorRatesTemplate
from diffwofost.physical_models.traitlets import Tensor
from diffwofost.physical_models.utils import AfgenTrait


class N_Stress(SimulationObject):
    """Leaf-death and juvenile-growth reduction from the crop nitrogen status.

    ``NSLLV`` accelerates leaf ageing. ``RFRGRL`` reduces the relative leaf
    expansion rate while the canopy is still small. Both are table lookups of
    a nitrogen index, so they stay differentiable through ``Afgen``.
    """

    class Parameters(TensorParamTemplate):
        NMAXLV_TB = AfgenTrait()
        NSLLV_TB = AfgenTrait()
        NMAXRT_FR = Tensor(-99.0)
        NMAXST_FR = Tensor(-99.0)
        NRESIDLV = Tensor(-99.0)
        NRESIDST = Tensor(-99.0)
        NMAXSO = Tensor(-99.0)
        RGRLAI_MIN = Tensor(-99.0)
        RGRLAI = Tensor(-99.0)

    class RateVariables(TensorRatesTemplate):
        NSLLV = Tensor(0.0)
        RFRGRL = Tensor(0.0)

    def initialize(self, day, kiosk, parvalues, shape=None):
        """Read the nitrogen-stress parameters."""
        self.kiosk = kiosk
        self.params = self.Parameters(parvalues, shape=shape)
        self.rates = self.RateVariables(kiosk, publish=["NSLLV", "RFRGRL"], shape=shape)

    def calc_rates(self, day: datetime.date, drv):
        """Compute NSLLV and RFRGRL from current biomass and nitrogen amounts."""
        params = self.params
        rates = self.rates
        kiosk = self.kiosk
        nmax_leaf = params.NMAXLV_TB(kiosk["DVS"])
        nmax_stem = params.NMAXST_FR * nmax_leaf
        nitrogen_above = kiosk["NamountLV"] + kiosk["NamountST"] + kiosk["NamountSO"]
        nitrogen_above_max = (
            kiosk["WLV"] * nmax_leaf + kiosk["WST"] * nmax_stem + kiosk["WSO"] * params.NMAXSO
        )
        ratio = nitrogen_above_max / torch.clamp(nitrogen_above, min=1e-8)
        stress_index = torch.clamp(ratio, min=1.0, max=2.0)
        rates.NSLLV = params.NSLLV_TB(stress_index)

        leaf_concentration = torch.where(
            kiosk["WLV"] > 0, kiosk["NamountLV"] / kiosk["WLV"], torch.zeros_like(kiosk["WLV"])
        )
        growth_index = torch.clamp(
            (leaf_concentration - 0.9 * nmax_leaf) / (nmax_leaf - 0.9 * nmax_leaf),
            min=0.0,
            max=1.0,
        )
        rates.RFRGRL = (
            1.0 - (1.0 - growth_index) * (params.RGRLAI - params.RGRLAI_MIN) / params.RGRLAI
        )

    def __call__(self, day, drv):
        """PCSE calls this module directly."""
        return self.calc_rates(day, drv)
