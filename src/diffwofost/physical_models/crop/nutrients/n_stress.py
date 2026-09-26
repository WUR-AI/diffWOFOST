"""Nitrogen stress factors for WOFOST 8.1."""

import datetime
import torch
from pcse.base import SimulationObject
from diffwofost.physical_models.base import TensorParamTemplate
from diffwofost.physical_models.base import TensorRatesTemplate
from diffwofost.physical_models.traitlets import Tensor
from diffwofost.physical_models.utils import AfgenTrait


class N_Stress(SimulationObject):
    """Nitrogen stress factors for leaf death and juvenile leaf expansion.

    ``NSLLV`` is a multiplication factor for leaf ageing. ``RFRGRL`` reduces
    the relative growth rate of leaf area while the canopy is still in the
    exponential phase. Both are table lookups of a nitrogen index, so they
    stay differentiable through ``Afgen``.

    **Simulation parameters**

    | Name       | Description                                           | Unit                  |
    |------------|-------------------------------------------------------|-----------------------|
    | NMAXLV_TB  | Maximum N concentration in leaves as function of DVS | kg N kg-1 dry matter  |
    | NMAXRT_FR  | Maximum N in roots as a fraction of the leaf maximum | -                     |
    | NMAXST_FR  | Maximum N in stems as a fraction of the leaf maximum | -                     |
    | NMAXSO     | Maximum N concentration in storage organs            | kg N kg-1 dry matter  |
    | NRESIDLV   | Residual N fraction in leaves                         | kg N kg-1 dry matter  |
    | NRESIDST   | Residual N fraction in stems                          | kg N kg-1 dry matter  |
    | NSLLV_TB   | Leaf-death stress factor as function of the N index  | -                     |
    | RGRLAI     | Maximum relative growth rate of leaf area             | d-1                   |
    | RGRLAI_MIN | Relative growth rate of leaf area at maximum N stress| d-1                   |

    **Rate variables**

    These are not rates of a state. They are used directly when the leaf
    rates are calculated.

    | Name   | Description                                              | Pbl | Unit |
    |--------|----------------------------------------------------------|-----|------|
    | NSLLV  | Nitrogen stress factor for leaf death                    | Y   | -    |
    | RFRGRL | Reduction of relative leaf growth in the exponential phase| Y  | -    |

    **External dependencies**

    | Name      | Description                  | Provided by                   | Unit    |
    |-----------|------------------------------|-------------------------------|---------|
    | DVS       | Crop development stage       | DVS_Phenology                 | -       |
    | WLV       | Dry weight of living leaves  | WOFOST_Leaf_Dynamics          | kg ha-1 |
    | WST       | Dry weight of living stems   | WOFOST_Stem_Dynamics          | kg ha-1 |
    | WSO       | Dry weight of storage organs | WOFOST_Storage_Organ_Dynamics | kg ha-1 |
    | NamountLV | Amount of N in leaves        | N_Crop_Dynamics               | kg ha-1 |
    | NamountST | Amount of N in stems         | N_Crop_Dynamics               | kg ha-1 |
    | NamountSO | Amount of N in storage organs| N_Crop_Dynamics               | kg ha-1 |

    **Gradient mapping (which parameters have a gradient):**

    | Output | Parameters influencing it                          |
    |--------|----------------------------------------------------|
    | NSLLV  | NMAXLV_TB, NMAXST_FR, NMAXSO, NSLLV_TB             |
    | RFRGRL | NMAXLV_TB, RGRLAI, RGRLAI_MIN                      |

    [!NOTE]
    ``NMAXRT_FR``, ``NRESIDLV`` and ``NRESIDST`` are not read by this module.
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

        # ``torch.where`` still divides when WLV is zero. The clamp keeps that
        # unused branch finite; the selected value stays 0, as in PCSE.
        leaf_weight = torch.clamp(kiosk["WLV"], min=1e-12)
        leaf_concentration = torch.where(
            kiosk["WLV"] > 0, kiosk["NamountLV"] / leaf_weight, torch.zeros_like(kiosk["WLV"])
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
