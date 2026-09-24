"""SNOMIN soil carbon and nitrogen balance.

Torch port of ``pcse.soil.snomin``. Layered mineral nitrogen (NH4, NO3) and
organic amendments follow Berghuijs et al. (2024). Hard thresholds, such as
the critical water-filled pore space for denitrification, use ``torch.where``
so the forward pass matches PCSE and gradients flow on the active side of the
threshold.
"""

import datetime
import torch
from pcse import exceptions as exc
from pcse import signals
from pcse.base import SimulationObject
from diffwofost.physical_models.base import TensorParamTemplate
from diffwofost.physical_models.base import TensorRatesTemplate
from diffwofost.physical_models.base import TensorStatesTemplate
from diffwofost.physical_models.config import ComputeConfig
from diffwofost.physical_models.soil.soil_profile import _as_tensor
from diffwofost.physical_models.traitlets import Tensor

_M2_TO_HA = 1e-4
_CM_TO_M = 1e-2
_Y_TO_D = 365.25
_MG_TO_KG = 1e-6
_L_TO_M3 = 1e-3
_OM_TO_C = 0.58


class SNOMIN(SimulationObject):
    """Layered soil nitrogen module for mineral and organic nitrogen.

    **Gradient mapping (which parameters have a gradient):**

    | Output          | Parameters influencing it                                      |
    |-----------------|----------------------------------------------------------------|
    | NH4, NO3, NAVAIL| NH4I, NO3I, KNIT_REF, KDENIT_REF, FASDIS, CNRatioBio,          |
    |                 | MRCDIS, KSORP, A0SOM                                           |
    | RNO3DENITR      | KDENIT_REF, MRCDIS                                             |
    | RNH4NITR        | KNIT_REF                                                       |

    [!NOTE]
    ``WFPS_CRIT`` is a hard threshold, so its gradient is zero. Denitrification
    and nitrification still have a gradient through the rate coefficients on the
    active side of that threshold.
    """

    soiln_profile = None

    _LAYER_STATES = ["AGE0", "AGE", "ORGMAT", "CORG", "NORG", "NH4", "NO3"]
    _LAYER_RATES = [
        "RAGE",
        "RORGMAT",
        "RCORG",
        "RNORG",
        "RAGEAG",
        "RORGMATDIS",
        "RCORGDIS",
        "RNORGDIS",
        "RAGEAM",
        "RORGMATAM",
        "RCORGAM",
        "RNORGAM",
        "RNH4",
        "RNH4MIN",
        "RNH4NITR",
        "RNH4UP",
        "RNH4IN",
        "RNH4OUT",
        "RNH4AM",
        "RNH4DEPOS",
        "RNO3",
        "RNO3NITR",
        "RNO3DENITR",
        "RNO3UP",
        "RNO3IN",
        "RNO3OUT",
        "RNO3AM",
        "RNO3DEPOS",
    ]

    class Parameters(TensorParamTemplate):
        A0SOM = Tensor(-99.0)
        CNRatioBio = Tensor(-99.0)
        FASDIS = Tensor(-99.0)
        KDENIT_REF = Tensor(-99.0)
        KNIT_REF = Tensor(-99.0)
        KSORP = Tensor(-99.0)
        MRCDIS = Tensor(-99.0)
        NO3ConcR = Tensor(-99.0)
        NH4ConcR = Tensor(-99.0)
        NO3I = Tensor(-99.0)
        NH4I = Tensor(-99.0)
        WFPS_CRIT = Tensor(-99.0)

    class StateVariables(TensorStatesTemplate):
        AGE0 = Tensor(-99.0)
        AGE = Tensor(-99.0)
        ORGMAT = Tensor(-99.0)
        CORG = Tensor(-99.0)
        NORG = Tensor(-99.0)
        NH4 = Tensor(-99.0)
        NO3 = Tensor(-99.0)
        NAVAIL = Tensor(-99.0)
        NDENITCUM = Tensor(-99.0)
        NO3LEACHCUM = Tensor(-99.0)
        NH4LEACHCUM = Tensor(-99.0)
        NLOSSCUM = Tensor(-99.0)
        RORGMATDISTT = Tensor(-99.0)
        RORGMATAMTT = Tensor(-99.0)
        RCORGDISTT = Tensor(-99.0)
        RCORGAMTT = Tensor(-99.0)
        RNORGDISTT = Tensor(-99.0)
        RNORGAMTT = Tensor(-99.0)
        RNO3NITRTT = Tensor(-99.0)
        RNO3DENITRTT = Tensor(-99.0)
        RNO3UPTT = Tensor(-99.0)
        RNO3INTT = Tensor(-99.0)
        RNO3OUTTT = Tensor(-99.0)
        RNO3AMTT = Tensor(-99.0)
        RNO3DEPOSTT = Tensor(-99.0)
        RNH4MINTT = Tensor(-99.0)
        RNH4NITRTT = Tensor(-99.0)
        RNH4UPTT = Tensor(-99.0)
        RNH4INTT = Tensor(-99.0)
        RNH4OUTTT = Tensor(-99.0)
        RNH4AMTT = Tensor(-99.0)
        RNH4DEPOSTT = Tensor(-99.0)
        ORGMATT = Tensor(-99.0)
        CORGT = Tensor(-99.0)
        NORGT = Tensor(-99.0)
        RMINT = Tensor(-99.0)
        NH4T = Tensor(-99.0)
        NO3T = Tensor(-99.0)

    class RateVariables(TensorRatesTemplate):
        RAGE = Tensor(0.0)
        RORGMAT = Tensor(0.0)
        RCORG = Tensor(0.0)
        RNORG = Tensor(0.0)
        RAGEAG = Tensor(0.0)
        RORGMATDIS = Tensor(0.0)
        RCORGDIS = Tensor(0.0)
        RNORGDIS = Tensor(0.0)
        RAGEAM = Tensor(0.0)
        RORGMATAM = Tensor(0.0)
        RCORGAM = Tensor(0.0)
        RNORGAM = Tensor(0.0)
        RNH4 = Tensor(0.0)
        RNH4MIN = Tensor(0.0)
        RNH4NITR = Tensor(0.0)
        RNH4UP = Tensor(0.0)
        RNH4IN = Tensor(0.0)
        RNH4OUT = Tensor(0.0)
        RNH4AM = Tensor(0.0)
        RNH4DEPOS = Tensor(0.0)
        RNO3 = Tensor(0.0)
        RNO3NITR = Tensor(0.0)
        RNO3DENITR = Tensor(0.0)
        RNO3UP = Tensor(0.0)
        RNO3IN = Tensor(0.0)
        RNO3OUT = Tensor(0.0)
        RNO3AM = Tensor(0.0)
        RNO3DEPOS = Tensor(0.0)
        RNH4LEACHCUM = Tensor(0.0)
        RNO3LEACHCUM = Tensor(0.0)
        RNDENITCUM = Tensor(0.0)

    def initialize(self, day, kiosk, parvalues, shape=None):
        """Initialise organic matter and mineral nitrogen per soil layer."""
        self._device = ComputeConfig.get_device()
        self._dtype = ComputeConfig.get_dtype()
        self.kiosk = kiosk
        # Initial mineral nitrogen is one value per layer, not a batch axis.
        self.params = self.Parameters(parvalues, shape=shape, do_not_broadcast=["NH4I", "NO3I"])
        if "soil_profile" not in parvalues:
            msg = "SNOMIN requires the multi-layer water balance to create 'soil_profile' first."
            raise exc.PCSEError(msg)
        self.soiln_profile = parvalues["soil_profile"]
        n_layers = len(self.soiln_profile)
        params = self.params
        nh4 = torch.stack([params.NH4I[il] * _M2_TO_HA for il in range(n_layers)], dim=0)
        no3 = torch.stack([params.NO3I[il] * _M2_TO_HA for il in range(n_layers)], dim=0)
        age = []
        orgmat = []
        corg = []
        norg = []
        for layer in self.soiln_profile:
            age_il = params.A0SOM * _Y_TO_D
            organic = layer.RHOD_kg_per_m3 * layer.FSOMI * layer.Thickness_m
            carbon = organic * _OM_TO_C
            age.append(age_il)
            orgmat.append(organic)
            corg.append(carbon)
            norg.append(carbon / layer.CNRatioSOMI)
        # Amendment axis first, layer axis second: (1, n_layers, *batch).
        # Layer properties are identical for every batch member; parameters
        # such as A0SOM already carry the batch axis.
        batch = tuple(params.shape)
        self._batch_ndim = len(batch)
        age = _with_batch(torch.stack(age, dim=0).unsqueeze(0), batch)
        orgmat = _with_batch(torch.stack(orgmat, dim=0).unsqueeze(0), batch)
        corg = _with_batch(torch.stack(corg, dim=0).unsqueeze(0), batch)
        norg = _with_batch(torch.stack(norg, dim=0).unsqueeze(0), batch)
        nh4 = _with_batch(nh4, batch)
        no3 = _with_batch(no3, batch)
        zeros = params.A0SOM.new_zeros(params.shape)
        self.states = self.StateVariables(
            kiosk,
            publish=["NAVAIL", "ORGMATT", "CORGT", "NORGT"],
            do_not_broadcast=self._LAYER_STATES,
            AGE0=age,
            AGE=age,
            ORGMAT=orgmat,
            CORG=corg,
            NORG=norg,
            NH4=nh4,
            NO3=no3,
            RORGMATDISTT=zeros,
            RORGMATAMTT=zeros,
            RCORGDISTT=zeros,
            RCORGAMTT=zeros,
            RNORGDISTT=zeros,
            RNORGAMTT=zeros,
            RNO3NITRTT=zeros,
            RNO3DENITRTT=zeros,
            RNO3UPTT=zeros,
            RNO3INTT=zeros,
            RNO3OUTTT=zeros,
            RNO3AMTT=zeros,
            RNO3DEPOSTT=zeros,
            RNH4MINTT=zeros,
            RNH4NITRTT=zeros,
            RNH4UPTT=zeros,
            RNH4INTT=zeros,
            RNH4OUTTT=zeros,
            RNH4AMTT=zeros,
            RNH4DEPOSTT=zeros,
            CORGT=_sum_soil(corg, self._batch_ndim) / _M2_TO_HA,
            NORGT=_sum_soil(norg, self._batch_ndim) / _M2_TO_HA,
            ORGMATT=_sum_soil(orgmat, self._batch_ndim) / _M2_TO_HA,
            RMINT=zeros,
            NH4T=_sum_soil(nh4, self._batch_ndim) / _M2_TO_HA,
            NO3T=_sum_soil(no3, self._batch_ndim) / _M2_TO_HA,
            NAVAIL=zeros,
            NH4LEACHCUM=zeros,
            NO3LEACHCUM=zeros,
            NDENITCUM=zeros,
            NLOSSCUM=zeros,
            shape=shape,
        )
        self.rates = self.RateVariables(kiosk, do_not_broadcast=self._LAYER_RATES, shape=shape)
        self._RAGEAM = torch.zeros_like(age)
        self._RORGMATAM = torch.zeros_like(orgmat)
        self._RCORGAM = torch.zeros_like(corg)
        self._RNORGAM = torch.zeros_like(norg)
        self._RNH4AM = torch.zeros_like(nh4)
        self._RNO3AM = torch.zeros_like(no3)
        self._ORGMATI = orgmat
        self._CORGI = corg
        self._NORGI = norg
        self._NH4I = nh4
        self._NO3I = no3
        self._connect_signal(self._on_APPLY_N_SNOMIN, signals.apply_n_snomin)

    def calc_rates(self, day: datetime.date, drv):
        """Rates of ageing, mineralisation, nitrification, uptake and leaching."""
        params = self.params
        states = self.states
        rates = self.rates
        kiosk = self.kiosk
        profile = self.soiln_profile
        delt = 1.0
        temperature = _as_tensor(drv.TEMP if hasattr(drv, "TEMP") else drv["TEMP"])
        infiltration = kiosk["RIN"] * _CM_TO_M
        flow = kiosk["Flow"] * _CM_TO_M
        demand = kiosk["RNuptake"] * _M2_TO_HA if "RNuptake" in kiosk else _as_tensor(0.0)
        rooting_depth = kiosk["RD"] * _CM_TO_M if "RD" in kiosk else _as_tensor(0.0)
        soil_moisture = kiosk["SM"]
        pf = torch.stack(
            [layer.PFfromSM(soil_moisture[il]) for il, layer in enumerate(profile)], dim=0
        )
        ph = torch.stack([layer.Soil_pH for layer in profile], dim=0)

        rates.RAGEAM = self._RAGEAM
        rates.RORGMATAM = self._RORGMATAM
        rates.RCORGAM = self._RCORGAM
        rates.RNORGAM = self._RNORGAM
        rates.RNH4AM = self._RNH4AM
        rates.RNO3AM = self._RNO3AM
        self._RAGEAM = torch.zeros_like(states.AGE)
        self._RORGMATAM = torch.zeros_like(rates.RORGMATAM)
        self._RCORGAM = torch.zeros_like(rates.RCORGAM)
        self._RNORGAM = torch.zeros_like(rates.RNORGAM)
        self._RNH4AM = torch.zeros_like(states.NH4)
        self._RNO3AM = torch.zeros_like(states.NO3)

        rates.RAGEAG = _age_increase(states.AGE, delt, pf, ph, temperature)
        rates.RORGMATDIS, rates.RCORGDIS, rates.RNORGDIS = _dissimilation(
            states.AGE,
            states.ORGMAT,
            states.NORG,
            params.CNRatioBio,
            params.FASDIS,
            pf,
            ph,
            temperature,
        )
        rates.RAGE = rates.RAGEAG + rates.RAGEAM
        rates.RORGMAT = rates.RORGMATAM - rates.RORGMATDIS
        rates.RCORG = rates.RCORGAM - rates.RCORGDIS

        rates.RNH4UP, rates.RNO3UP = _uptake(
            profile,
            delt,
            params.KSORP,
            demand,
            states.NH4,
            states.NO3,
            rooting_depth,
            soil_moisture,
        )
        nh4_pre = states.NH4 - rates.RNH4UP * delt
        no3_pre = states.NO3 - rates.RNO3UP * delt
        # Amendment axis only. The result stays one mineralization rate per layer.
        rates.RNH4MIN = rates.RNORGDIS.sum(dim=0)
        rates.RNH4NITR = _nitrification(
            profile, params.KNIT_REF, params.KSORP, nh4_pre, soil_moisture, temperature
        )
        rates.RNH4MIN, rates.RNORGDIS = _limit_immobilisation(
            nh4_pre, rates.RNH4MIN, rates.RNH4NITR, rates.RNORGDIS, delt
        )
        rates.RNORG = rates.RNORGAM - rates.RNORGDIS
        rates.RNO3NITR = rates.RNH4NITR
        rates.RNO3DENITR = _denitrification(
            profile,
            params.KDENIT_REF,
            params.MRCDIS,
            params.WFPS_CRIT,
            no3_pre,
            rates.RCORGDIS.sum(dim=0),
            soil_moisture,
            temperature,
        )
        rates.RNH4DEPOS, rates.RNO3DEPOS = _deposition(
            infiltration, params.NH4ConcR, params.NO3ConcR, states.NH4
        )
        nh4_after = (
            nh4_pre + (rates.RNH4AM + rates.RNH4MIN + rates.RNH4DEPOS - rates.RNH4NITR) * delt
        )
        no3_after = (
            no3_pre + (rates.RNO3AM + rates.RNO3NITR + rates.RNO3DEPOS - rates.RNO3DENITR) * delt
        )
        conc_nh4 = _ammonium_concentration(profile, params.KSORP, nh4_after, soil_moisture)
        conc_no3 = _nitrate_concentration(profile, no3_after, soil_moisture)
        rates.RNH4IN, rates.RNH4OUT = _solute_flow(flow, conc_nh4)
        rates.RNO3IN, rates.RNO3OUT = _solute_flow(flow, conc_no3)
        rates.RNH4 = (
            rates.RNH4AM
            + rates.RNH4MIN
            + rates.RNH4DEPOS
            - rates.RNH4NITR
            - rates.RNH4UP
            + rates.RNH4IN
            - rates.RNH4OUT
        )
        rates.RNO3 = (
            rates.RNO3AM
            + rates.RNO3NITR
            + rates.RNO3DEPOS
            - rates.RNO3DENITR
            - rates.RNO3UP
            + rates.RNO3IN
            - rates.RNO3OUT
        )
        rates.RNH4LEACHCUM = rates.RNH4OUT[-1] / _M2_TO_HA
        rates.RNO3LEACHCUM = rates.RNO3OUT[-1] / _M2_TO_HA
        rates.RNDENITCUM = rates.RNO3DENITR.sum(dim=0) / _M2_TO_HA

    def integrate(self, day: datetime.date, delt=1.0):
        """Step organic and mineral pools and publish plant-available nitrogen."""
        states = self.states
        rates = self.rates
        params = self.params
        kiosk = self.kiosk
        states.AGE = states.AGE + rates.RAGE * delt
        states.ORGMAT = states.ORGMAT + rates.RORGMAT * delt
        states.CORG = states.CORG + rates.RCORG * delt
        states.NORG = states.NORG + rates.RNORG * delt
        states.NH4 = states.NH4 + rates.RNH4 * delt
        states.NO3 = states.NO3 + rates.RNO3 * delt
        rooting_depth = kiosk["RD"] * _CM_TO_M if "RD" in kiosk else _as_tensor(0.0)
        states.NAVAIL = (
            _available_nitrogen(
                self.soiln_profile, params.KSORP, states.NH4, states.NO3, rooting_depth, kiosk["SM"]
            )
            / _M2_TO_HA
        )
        self._check_mass_balances(day, delt)
        # Organic pools are (amendments, layers, *batch). Mineral pools are
        # (layers, *batch). Sum those leading axes only; a bare sum would add
        # the batch members together.
        ha = 1.0 / _M2_TO_HA
        batch_ndim = self._batch_ndim
        states.ORGMATT = _sum_soil(states.ORGMAT, batch_ndim) * ha
        states.CORGT = _sum_soil(states.CORG, batch_ndim) * ha
        states.NORGT = _sum_soil(states.NORG, batch_ndim) * ha
        states.RMINT = states.RMINT + _sum_soil(rates.RNORGDIS, batch_ndim) * ha
        states.NH4T = _sum_soil(states.NH4, batch_ndim) * ha
        states.NO3T = _sum_soil(states.NO3, batch_ndim) * ha
        states.NH4LEACHCUM = states.NH4LEACHCUM + rates.RNH4LEACHCUM * delt
        states.NO3LEACHCUM = states.NO3LEACHCUM + rates.RNO3LEACHCUM * delt
        states.NDENITCUM = states.NDENITCUM + rates.RNDENITCUM * delt
        states.NLOSSCUM = states.NH4LEACHCUM + states.NO3LEACHCUM + states.NDENITCUM

    def _on_APPLY_N_SNOMIN(
        self,
        amount=None,
        application_depth=None,
        cnratio=None,
        f_orgmat=None,
        f_NH4N=None,
        f_NO3N=None,
        initial_age=None,
    ):
        """Queue one fertiliser or manure amendment for the next rate calculation."""
        profile = self.soiln_profile
        depth = _as_tensor(application_depth)
        depth = torch.maximum(depth, profile[0].Thickness)
        amount_t = _as_tensor(amount)
        cn_ratio = _as_tensor(cnratio)
        organic_fraction = _as_tensor(f_orgmat)
        age = _as_tensor(initial_age) * _Y_TO_D
        nh4_am, no3_am = _application_mineral(
            profile, amount_t, depth, _as_tensor(f_NH4N), _as_tensor(f_NO3N)
        )
        org_am, corg_am, norg_am = _application_organic(
            profile, amount_t, depth, cn_ratio, organic_fraction
        )
        n_layers = len(profile)
        zeros = torch.zeros((1, n_layers), dtype=self._dtype, device=self._device)
        age_row = age.expand(1, n_layers)
        self.states.AGE0 = torch.cat((self.states.AGE0, age_row), dim=0)
        self.states.ORGMAT = torch.cat((self.states.ORGMAT, zeros), dim=0)
        self.states.CORG = torch.cat((self.states.CORG, zeros), dim=0)
        self.states.NORG = torch.cat((self.states.NORG, zeros), dim=0)
        self.states.AGE = torch.cat((self.states.AGE, zeros), dim=0)
        self._RAGEAM = torch.cat((self._RAGEAM, age_row), dim=0)
        self._RORGMATAM = torch.cat((self._RORGMATAM, org_am.unsqueeze(0) * _M2_TO_HA), dim=0)
        self._RCORGAM = torch.cat((self._RCORGAM, corg_am.unsqueeze(0) * _M2_TO_HA), dim=0)
        self._RNORGAM = torch.cat((self._RNORGAM, norg_am.unsqueeze(0) * _M2_TO_HA), dim=0)
        self._RNH4AM = nh4_am * _M2_TO_HA
        self._RNO3AM = no3_am * _M2_TO_HA

    def _check_mass_balances(self, day, delt):
        states = self.states
        rates = self.rates
        # Called from integrate. Organic rates sum amendments and layers.
        # Mineral rates sum layers. The trailing batch axis stays intact, so
        # each member keeps its own cumulative total.
        batch_ndim = self._batch_ndim

        def total(value):
            return _sum_soil(value, batch_ndim)

        states.RORGMATAMTT = states.RORGMATAMTT + delt * total(rates.RORGMATAM)
        states.RORGMATDISTT = states.RORGMATDISTT + delt * total(rates.RORGMATDIS)
        states.RCORGAMTT = states.RCORGAMTT + delt * total(rates.RCORGAM)
        states.RCORGDISTT = states.RCORGDISTT + delt * total(rates.RCORGDIS)
        states.RNORGAMTT = states.RNORGAMTT + delt * total(rates.RNORGAM)
        states.RNORGDISTT = states.RNORGDISTT + delt * total(rates.RNORGDIS)
        states.RNH4MINTT = states.RNH4MINTT + delt * total(rates.RNH4MIN)
        states.RNH4NITRTT = states.RNH4NITRTT + delt * total(rates.RNH4NITR)
        states.RNH4UPTT = states.RNH4UPTT + delt * total(rates.RNH4UP)
        states.RNH4INTT = states.RNH4INTT + delt * total(rates.RNH4IN)
        states.RNH4OUTTT = states.RNH4OUTTT + delt * total(rates.RNH4OUT)
        states.RNH4AMTT = states.RNH4AMTT + delt * total(rates.RNH4AM)
        states.RNH4DEPOSTT = states.RNH4DEPOSTT + delt * total(rates.RNH4DEPOS)
        states.RNO3NITRTT = states.RNO3NITRTT + delt * total(rates.RNO3NITR)
        states.RNO3DENITRTT = states.RNO3DENITRTT + delt * total(rates.RNO3DENITR)
        states.RNO3UPTT = states.RNO3UPTT + delt * total(rates.RNO3UP)
        states.RNO3INTT = states.RNO3INTT + delt * total(rates.RNO3IN)
        states.RNO3OUTTT = states.RNO3OUTTT + delt * total(rates.RNO3OUT)
        states.RNO3AMTT = states.RNO3AMTT + delt * total(rates.RNO3AM)
        states.RNO3DEPOSTT = states.RNO3DEPOSTT + delt * total(rates.RNO3DEPOS)

        organic = (
            total(self._ORGMATI) - total(states.ORGMAT) + states.RORGMATAMTT - states.RORGMATDISTT
        )
        carbon = total(self._CORGI) - total(states.CORG) + states.RCORGAMTT - states.RCORGDISTT
        nitrogen = total(self._NORGI) - total(states.NORG) + states.RNORGAMTT - states.RNORGDISTT
        ammonium = (
            total(self._NH4I)
            - total(states.NH4)
            + states.RNH4AMTT
            + states.RNH4INTT
            + states.RNH4MINTT
            + states.RNH4DEPOSTT
            - states.RNH4NITRTT
            - states.RNH4OUTTT
            - states.RNH4UPTT
        )
        nitrate = (
            total(self._NO3I)
            - total(states.NO3)
            + states.RNO3AMTT
            + states.RNO3NITRTT
            + states.RNO3INTT
            + states.RNO3DEPOSTT
            - states.RNO3DENITRTT
            - states.RNO3OUTTT
            - states.RNO3UPTT
        )
        if torch.any(torch.abs(organic) > 1e-4):
            raise exc.SoilOrganicMatterBalanceError(f"Organic matter balance on {day}: {organic}")
        if torch.any(torch.abs(carbon) > 1e-4):
            raise exc.SoilOrganicCarbonBalanceError(f"Organic carbon balance on {day}: {carbon}")
        if torch.any(torch.abs(nitrogen) > 1e-4):
            raise exc.SoilOrganicNitrogenBalanceError(
                f"Organic nitrogen balance on {day}: {nitrogen}"
            )
        if torch.any(torch.abs(ammonium) > 1e-4):
            raise exc.SoilAmmoniumBalanceError(f"NH4 balance on {day}: {ammonium}")
        if torch.any(torch.abs(nitrate) > 1e-4):
            raise exc.SoilNitrateBalanceError(f"NO3 balance on {day}: {nitrate}")


def _sum_soil(value: torch.Tensor, batch_ndim: int) -> torch.Tensor:
    """Sum every axis in front of the batch.

    An organic quantity is ranked (amendments, layers, *batch), so both soil
    axes are included. A mineral quantity is ranked (layers, *batch), so only
    the layer axis is included. The batch rank is the parameter shape stored
    at initialisation.
    """
    soil_rank = value.dim() - batch_ndim
    if soil_rank <= 0:
        return value
    return value.sum(dim=tuple(range(soil_rank)))


def _with_batch(value: torch.Tensor, batch: tuple) -> torch.Tensor:
    """Repeat a per-layer tensor along a trailing batch axis."""
    if not batch or value.shape[-len(batch) :] == batch:
        return value
    target = value.shape + batch
    expanded = value
    while expanded.dim() < len(target):
        expanded = expanded.unsqueeze(-1)
    return expanded.expand(target).clone()


def _moisture_response(pf: torch.Tensor) -> torch.Tensor:
    return torch.where(pf < 2.7, 1.0, torch.where(pf < 4.2, (4.2 - pf) / (4.2 - 2.7), 0.0))


def _temperature_response(temperature: torch.Tensor) -> torch.Tensor:
    cold = 0.09 * (temperature + 1)
    mild = 0.88 * torch.pow(2.0, (temperature - 9.0) / 9.0)
    return torch.where(
        temperature < -1,
        torch.zeros_like(temperature),
        torch.where(
            temperature < 9,
            cold,
            torch.where(temperature < 27, mild, 3.5 + torch.zeros_like(temperature)),
        ),
    )


def _ph_response(ph: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-1.5 * (ph - 4)))


def _layer_response(pf, ph, temperature):
    """pH, temperature and moisture factors, with a trailing batch axis when present.

    ``ph`` is one value per layer. ``pf`` is per layer, or per layer and batch
    member when soil moisture is batched. The pH factor is expanded on the
    right so it lines up with that batch axis.
    """
    moisture = _moisture_response(pf)
    acidity = _ph_response(ph)
    while acidity.dim() < moisture.dim():
        acidity = acidity.unsqueeze(-1)
    return acidity * _temperature_response(temperature) * moisture


def _align_to(response, reference):
    """Add leading axes so a layer response broadcasts onto an amendment tensor."""
    while response.dim() < reference.dim():
        response = response.unsqueeze(0)
    return response


def _relative_dissimilation(age, pf, ph, temperature):
    response = _align_to(_layer_response(pf, ph, temperature), age)
    safe_age = torch.clamp(age, min=1e-6)
    return response * 2.82 * torch.pow(safe_age / _Y_TO_D, -1.6) / _Y_TO_D


def _age_increase(age, delt, pf, ph, temperature):
    return _align_to(_layer_response(pf, ph, temperature), age) * delt


def _dissimilation(
    age, organic_matter, organic_nitrogen, cn_ratio_bio, fasdis, pf, ph, temperature
):
    rate_constant = _relative_dissimilation(age, pf, ph, temperature)
    active = organic_matter > 0
    organic_rate = torch.where(
        active, rate_constant * organic_matter, torch.zeros_like(organic_matter)
    )
    carbon_rate = torch.where(
        active, rate_constant * organic_matter * _OM_TO_C, torch.zeros_like(organic_matter)
    )
    carbon = organic_matter * _OM_TO_C
    conversion = torch.where(
        active,
        (carbon_rate * (1.0 + fasdis)) * organic_nitrogen / torch.clamp(carbon, min=1e-12),
        torch.zeros_like(organic_matter),
    )
    assimilation = torch.where(
        active, carbon_rate * fasdis / cn_ratio_bio, torch.zeros_like(organic_matter)
    )
    nitrogen_rate = torch.where(active, conversion - assimilation, torch.zeros_like(organic_matter))
    return organic_rate, carbon_rate, nitrogen_rate


def _limit_immobilisation(nh4_pre, mineralization, nitrification, organic_nitrogen_rate, delt):
    """Keep the PCSE assignment when net immobilisation would be used.

    PCSE sets the limited mineralization to the pre-uptake ammonium minus
    nitrification, then scales every amendment in that layer by the ratio of
    the limited rate to the original amendment sum.
    """
    immobilised = nh4_pre + (mineralization - nitrification) * delt < 0
    limited = torch.where(immobilised, nh4_pre - nitrification, mineralization)
    mineral_n = organic_nitrogen_rate.sum(dim=0)
    scale = torch.where(
        immobilised,
        limited / torch.clamp(mineral_n, min=1e-12),
        torch.ones_like(mineral_n),
    )
    return limited, organic_nitrogen_rate * scale.unsqueeze(0)


def _layer_fraction(depth, z_min, z_max, thickness):
    covered = depth > z_max
    partial = (depth > z_min) & (depth <= z_max)
    fraction = torch.where(
        covered,
        torch.ones_like(depth),
        torch.where(partial, (depth - z_min) / thickness, torch.zeros_like(depth)),
    )
    return fraction


def _available_ammonium(ksorp, ammonium, rooting_depth, bulk_density, moisture, z_max, z_min):
    thickness = z_max - z_min
    soluble = moisture / (ksorp * bulk_density + moisture)
    return _layer_fraction(rooting_depth, z_min, z_max, thickness) * soluble * ammonium


def _available_nitrate(nitrate, rooting_depth, z_max, z_min):
    thickness = z_max - z_min
    return _layer_fraction(rooting_depth, z_min, z_max, thickness) * nitrate


def _available_nitrogen(profile, ksorp, ammonium, nitrate, rooting_depth, moisture):
    total = ammonium.new_zeros(ammonium.shape[1:])
    z_min = ammonium.new_zeros(())
    for il, layer in enumerate(profile):
        z_max = z_min + layer.Thickness_m
        total = total + _available_ammonium(
            ksorp, ammonium[il], rooting_depth, layer.RHOD_kg_per_m3, moisture[il], z_max, z_min
        )
        total = total + _available_nitrate(nitrate[il], rooting_depth, z_max, z_min)
        z_min = z_max
    return total


def _uptake(profile, delt, ksorp, demand, ammonium, nitrate, rooting_depth, moisture):
    remaining = demand
    nh4_up = []
    no3_up = []
    z_min = ammonium.new_zeros(())
    for il, layer in enumerate(profile):
        z_max = z_min + layer.Thickness_m
        nh4_av = _available_ammonium(
            ksorp, ammonium[il], rooting_depth, layer.RHOD_kg_per_m3, moisture[il], z_max, z_min
        )
        nh4_rate = torch.minimum(remaining, nh4_av)
        remaining = remaining - nh4_rate * delt
        no3_av = _available_nitrate(nitrate[il], rooting_depth, z_max, z_min)
        no3_rate = torch.minimum(torch.clamp(remaining, min=0.0), no3_av)
        remaining = remaining - no3_rate * delt
        nh4_up.append(nh4_rate)
        no3_up.append(no3_rate)
        z_min = z_max
    return torch.stack(nh4_up, dim=0), torch.stack(no3_up, dim=0)


def _ammonium_concentration(profile, ksorp, ammonium, moisture):
    concentration = []
    for il, layer in enumerate(profile):
        concentration.append(
            ammonium[il] / ((ksorp * layer.RHOD_kg_per_m3 + moisture[il]) * layer.Thickness_m)
        )
    return torch.stack(concentration, dim=0)


def _nitrate_concentration(profile, nitrate, moisture):
    concentration = []
    for il, layer in enumerate(profile):
        concentration.append(
            nitrate[il] / (layer.Thickness_m * torch.clamp(moisture[il], min=1e-8))
        )
    return torch.stack(concentration, dim=0)


def _nitrification(profile, knit_ref, ksorp, ammonium, moisture, temperature):
    rates = []
    temperature_factor = 1.0 / (1.0 + torch.exp(-0.26 * (temperature - 17.0))) - 1.0 / (
        1.0 + torch.exp(-0.77 * (temperature - 41.9))
    )
    for il, layer in enumerate(profile):
        concentration = ammonium[il] / (
            (ksorp * layer.RHOD_kg_per_m3 + moisture[il]) * layer.Thickness_m
        )
        water_filled = moisture[il] / layer.SM0
        moisture_factor = (
            0.9 / (1.0 + torch.exp(-15 * (water_filled - 0.45)))
            + 0.1
            - 1.0 / (1.0 + torch.exp(-50.0 * (water_filled - 0.95)))
        )
        rates.append(
            moisture_factor
            * temperature_factor
            * knit_ref
            * moisture[il]
            * concentration
            * layer.Thickness_m
        )
    return torch.stack(rates, dim=0)


def _denitrification(
    profile, kdenit_ref, mrcdis, wfps_crit, nitrate, carbon_rate, moisture, temperature
):
    rates = []
    temperature_factor = 1.0 / (1.0 + torch.exp(-0.26 * (temperature - 17.0))) - 1.0 / (
        1.0 + torch.exp(-0.77 * (temperature - 41.9))
    )
    for il, layer in enumerate(profile):
        # Carbon rate is already summed over amendments and is one value per layer.
        layer_carbon = carbon_rate[il]
        respiration_factor = layer_carbon / (mrcdis + layer_carbon)
        concentration = nitrate[il] / (layer.Thickness_m * torch.clamp(moisture[il], min=1e-8))
        water_filled = moisture[il] / layer.SM0
        wet = water_filled >= wfps_crit
        moisture_factor = torch.where(
            wet,
            torch.pow((water_filled - wfps_crit) / (1.0 - wfps_crit), 2),
            torch.zeros_like(water_filled),
        )
        rates.append(
            moisture_factor
            * temperature_factor
            * respiration_factor
            * kdenit_ref
            * moisture[il]
            * concentration
            * layer.Thickness_m
        )
    return torch.stack(rates, dim=0)


def _deposition(infiltration, nh4_conc, no3_conc, ammonium):
    """Rain adds mineral nitrogen to the top layer only."""
    flux = (_MG_TO_KG / _L_TO_M3) * infiltration
    zeros = torch.zeros_like(ammonium[0])
    nh4_layers = [flux * nh4_conc] + [zeros for _ in range(ammonium.shape[0] - 1)]
    no3_layers = [flux * no3_conc] + [torch.zeros_like(zeros) for _ in range(ammonium.shape[0] - 1)]
    return torch.stack(nh4_layers, dim=0), torch.stack(no3_layers, dim=0)


def _solute_flow(flow, concentration):
    """Advect a solute with the layered water flow, matching the PCSE indexing."""
    n_layers = concentration.shape[0]
    incoming = [concentration.new_zeros(concentration.shape[1:]) for _ in range(n_layers)]
    outgoing = [concentration.new_zeros(concentration.shape[1:]) for _ in range(n_layers)]
    downward = flow >= 0
    for il in range(1, n_layers):
        flux = torch.where(
            downward[il],
            flow[il] * concentration[il - 1],
            torch.zeros_like(concentration[il]),
        )
        incoming[il] = incoming[il] + flux
        outgoing[il - 1] = outgoing[il - 1] + flux
    bottom = torch.where(
        downward[n_layers - 1],
        flow[n_layers] * concentration[n_layers - 1],
        torch.zeros_like(concentration[n_layers - 1]),
    )
    outgoing[n_layers - 1] = outgoing[n_layers - 1] + bottom
    upward = flow < 0
    for il in range(n_layers - 2, -1, -1):
        flux = torch.where(
            upward[il + 1],
            -flow[il + 1] * concentration[il + 1],
            torch.zeros_like(concentration[il]),
        )
        incoming[il] = incoming[il] + flux
        outgoing[il + 1] = outgoing[il + 1] + flux
    top = torch.where(upward[0], -flow[0] * concentration[0], torch.zeros_like(concentration[0]))
    outgoing[0] = outgoing[0] + top
    return torch.stack(incoming, dim=0), torch.stack(outgoing, dim=0)


def _depth_share(application_depth, thickness, z_max, z_min):
    full = application_depth > z_max
    partial = (z_min <= application_depth) & (application_depth <= z_max)
    return torch.where(
        full,
        thickness / application_depth,
        torch.where(
            partial,
            (application_depth - z_min) / application_depth,
            torch.zeros_like(application_depth),
        ),
    )


def _application_mineral(profile, amount, application_depth, f_nh4, f_no3):
    nh4 = []
    no3 = []
    z_min = _as_tensor(0.0)
    for layer in profile:
        z_max = z_min + layer.Thickness
        share = _depth_share(application_depth, layer.Thickness, z_max, z_min)
        nh4.append(share * f_nh4 * amount)
        no3.append(share * f_no3 * amount)
        z_min = z_max
    return torch.stack(nh4, dim=0), torch.stack(no3, dim=0)


def _application_organic(profile, amount, application_depth, cn_ratio, organic_fraction):
    organic = []
    z_min = _as_tensor(0.0)
    for layer in profile:
        z_max = z_min + layer.Thickness
        share = _depth_share(application_depth, layer.Thickness, z_max, z_min)
        organic.append(share * organic_fraction * amount)
        z_min = z_max
    organic = torch.stack(organic, dim=0)
    carbon = organic * _OM_TO_C
    nitrogen = torch.where(
        cn_ratio == 0, torch.zeros_like(carbon), carbon / torch.clamp(cn_ratio, min=1e-8)
    )
    return organic, carbon, nitrogen
