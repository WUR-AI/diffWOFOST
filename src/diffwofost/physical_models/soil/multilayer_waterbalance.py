"""Multi-layer soil water balance.

Torch port of ``pcse.soil.multilayer_waterbalance``. The layer loop and the
bisection for the dry-flow limit are the same algorithm as PCSE. Comparisons
that depend on a continuous state use ``torch.where`` so a batch of simulations
can run together and gradients flow through the selected branch.
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
from diffwofost.physical_models.soil.soil_profile import SoilProfile
from diffwofost.physical_models.soil.soil_profile import _as_tensor
from diffwofost.physical_models.traitlets import Tensor
from diffwofost.physical_models.utils import Afgen


class WaterBalanceLayered(SimulationObject):
    """Layered water balance for water-limited production.

    Downward flow is the larger of a dry (matric-flux) flow and a wet
    (gravity) flow, limited so a layer cannot pass saturation or fall below
    field capacity. Upward flow is the dry flow when it is negative, limited
    to half of the amount that would equalise the two layers.
    """

    MaxFlowIter = 50
    TinyFlow = 0.001
    UpwardFlowLimit = 0.50

    # Declared so SimulationObject.__setattr__ accepts the assignment.
    soil_profile = None
    parameter_provider = None
    NINFTB = None

    class Parameters(TensorParamTemplate):
        IFUNRN = Tensor(-99.0)
        NOTINF = Tensor(-99.0)
        SSI = Tensor(-99.0)
        SSMAX = Tensor(-99.0)
        SMLIM = Tensor(-99.0)
        WAV = Tensor(-99.0)

    _LAYER_STATES = ["SM", "WC"]
    _LAYER_RATES = ["Flow", "WTRALY", "DWC"]

    class StateVariables(TensorStatesTemplate):
        WTRAT = Tensor(-99.0)
        EVST = Tensor(-99.0)
        EVWT = Tensor(-99.0)
        TSR = Tensor(-99.0)
        RAINT = Tensor(-99.0)
        WDRT = Tensor(-99.0)
        TOTINF = Tensor(-99.0)
        TOTIRR = Tensor(-99.0)
        CRT = Tensor(-99.0)
        SM = Tensor(-99.0)
        SM_MEAN = Tensor(-99.0)
        WC = Tensor(-99.0)
        W = Tensor(-99.0)
        WLOW = Tensor(-99.0)
        WWLOW = Tensor(-99.0)
        WBOT = Tensor(-99.0)
        WAVUPP = Tensor(-99.0)
        WAVLOW = Tensor(-99.0)
        WAVBOT = Tensor(-99.0)
        SS = Tensor(-99.0)
        BOTTOMFLOWT = Tensor(-99.0)

    class RateVariables(TensorRatesTemplate):
        Flow = Tensor(0.0)
        RIN = Tensor(0.0)
        WTRALY = Tensor(0.0)
        WTRA = Tensor(0.0)
        EVS = Tensor(0.0)
        EVW = Tensor(0.0)
        RIRR = Tensor(0.0)
        DWC = Tensor(0.0)
        DRAINT = Tensor(0.0)
        DSS = Tensor(0.0)
        DTSR = Tensor(0.0)
        BOTTOMFLOW = Tensor(0.0)

    def initialize(self, day, kiosk, parvalues, shape=None):
        """Build the soil profile and the initial moisture distribution."""
        self._device = ComputeConfig.get_device()
        self._dtype = ComputeConfig.get_dtype()
        self.soil_profile = SoilProfile(parvalues)
        parvalues._soildata["soil_profile"] = self.soil_profile
        if self.soil_profile.GroundWater:
            raise NotImplementedError("Groundwater influence not yet implemented.")

        self._RDM = self.soil_profile.get_max_rootable_depth()
        self.soil_profile.validate_max_rooting_depth(self._RDM)
        self.params = self.Parameters(parvalues, shape=shape)
        self.parameter_provider = parvalues
        self._default_RD = _as_tensor(10.0)
        self.soil_profile.determine_rooting_status(self._default_RD, self._RDM)
        self.NINFTB = Afgen([0.0, 0.0, 0.5, 0.0, 1.5, 1.0])

        sm, wc, totals = self._initial_moisture()
        n_layers = len(self.soil_profile)
        batch = self.params.shape
        flow = torch.zeros((n_layers + 1, *batch), dtype=self._dtype, device=self._device)
        self.states = self.StateVariables(
            kiosk,
            publish=["WC", "SM", "EVST"],
            do_not_broadcast=self._LAYER_STATES,
            WTRAT=0.0,
            EVST=0.0,
            EVWT=0.0,
            TSR=0.0,
            WDRT=0.0,
            TOTINF=0.0,
            TOTIRR=0.0,
            BOTTOMFLOWT=0.0,
            CRT=0.0,
            RAINT=0.0,
            WLOW=totals["WLOW"],
            W=totals["W"],
            WC=wc,
            SM=sm,
            SS=self.params.SSI,
            WWLOW=totals["W"] + totals["WLOW"],
            WBOT=0.0,
            SM_MEAN=totals["W"] / self._default_RD,
            WAVUPP=totals["WAVUPP"],
            WAVLOW=totals["WAVLOW"],
            WAVBOT=0.0,
            shape=shape,
        )
        self._WCI = wc.sum(dim=0)
        self.rates = self.RateVariables(
            kiosk, publish=["RIN", "Flow", "EVS"], do_not_broadcast=self._LAYER_RATES, shape=shape
        )
        self.rates.Flow = flow
        self._RINold = _as_tensor(0.0)
        self._RIRR = _as_tensor(0.0)
        self._RAIN = _as_tensor(0.0)
        self._RDold = self._default_RD
        top = self.soil_profile[0]
        half_wet = top.SMW + 0.5 * (top.SMFCF - top.SMW)
        self._DSLR = torch.where(sm[0] <= half_wet, _as_tensor(5.0), _as_tensor(1.0))
        self._crop_ready = False

        self._connect_signal(self._on_CROP_START, signals.crop_start)
        self._connect_signal(self._on_CROP_FINISH, signals.crop_finish)
        self._connect_signal(self._on_IRRIGATE, signals.irrigate)
        # diffWOFOST builds the soil after AgroManager has already emitted
        # CROP_START, so apply the crop rooting depth here when it is known.
        if "RDMCR" in parvalues:
            self._setup_new_crop()

    def _initial_moisture(self):
        """Distribute initial available water (WAV) over the rooted profile."""
        params = self.params
        profile = self.soil_profile
        avmax = []
        top_limit = params.WAV.new_zeros(())
        low_limit = params.WAV.new_zeros(())
        for layer in profile:
            rooted = float(layer.Wtop) > 0
            potential = float(layer.Wpot) > 0 and not rooted
            if rooted:
                sm_limit = torch.clamp(params.SMLIM, min=layer.SMW, max=layer.SM0)
                capacity = (sm_limit - layer.SMW) * layer.Thickness
                avmax.append(capacity)
                top_limit = top_limit + capacity
            elif potential:
                capacity = (layer.SM0 - layer.SMW) * layer.Thickness
                avmax.append(capacity)
                low_limit = low_limit + capacity
            else:
                break

        wav = params.WAV
        safe_top = torch.clamp(top_limit, min=1e-8)
        safe_low = torch.clamp(low_limit, min=1e-8)
        top_reduction = torch.where(
            wav <= 0,
            wav.new_zeros(()),
            torch.where(wav <= top_limit, wav / safe_top, wav.new_ones(())),
        )
        low_reduction = torch.where(
            wav <= top_limit,
            wav.new_zeros(()),
            torch.where(
                wav < top_limit + low_limit,
                (wav - top_limit) / safe_low,
                wav.new_ones(()),
            ),
        )

        water = wav.new_zeros(())
        water_low = wav.new_zeros(())
        available_top = wav.new_zeros(())
        available_low = wav.new_zeros(())
        sm_layers = []
        wc_layers = []
        il_capacity = 0
        for layer in profile:
            rooted = float(layer.Wtop) > 0
            potential = float(layer.Wpot) > 0 and not rooted
            if rooted or potential:
                reduction = top_reduction if rooted else low_reduction
                sm_il = layer.SMW + avmax[il_capacity] * reduction / layer.Thickness
                il_capacity += 1
            else:
                sm_il = layer.SMW
            sm_layers.append(sm_il)
            wc_layers.append(sm_il * layer.Thickness)
            if rooted or potential:
                water = water + sm_il * layer.Thickness * layer.Wtop
                water_low = water_low + sm_il * layer.Thickness * layer.Wpot
                available_top = available_top + (sm_il - layer.SMW) * layer.Thickness * layer.Wtop
                available_low = available_low + (sm_il - layer.SMW) * layer.Thickness * layer.Wpot
        sm = torch.stack(sm_layers, dim=0)
        wc = torch.stack(wc_layers, dim=0)
        totals = {
            "W": water,
            "WLOW": water_low,
            "WAVUPP": available_top,
            "WAVLOW": available_low,
        }
        return sm, wc, totals

    def calc_rates(self, day: datetime.date, drv):
        """Compute infiltration, layer flows and surface runoff for one day."""
        params = self.params
        states = self.states
        rates = self.rates
        kiosk = self.kiosk
        profile = self.soil_profile
        delt = 1.0
        rain = _as_tensor(drv.RAIN if hasattr(drv, "RAIN") else drv["RAIN"])

        rates.RIRR = self._RIRR
        self._RIRR = _as_tensor(0.0)
        self._RAIN = rain

        # Layered transpiration is published as one value per soil layer once
        # the crop has emerged. Before that the kiosk still holds the scalar
        # rate-template default, which is the same situation as PCSE having no
        # TRALY yet: evaporate at the potential soil and water rates.
        layered_transpiration = kiosk["TRALY"] if "TRALY" in kiosk else None
        if (
            isinstance(layered_transpiration, torch.Tensor)
            and layered_transpiration.ndim >= 1
            and layered_transpiration.shape[0] == len(profile)
        ):
            wtraly = layered_transpiration
            rates.WTRA = kiosk["TRA"]
            evwmx = kiosk["EVWMX"]
            evsmx = kiosk["EVSMX"]
        else:
            wtraly = torch.zeros_like(states.SM)
            rates.WTRA = _as_tensor(0.0)
            evwmx = _as_tensor(drv.E0 if hasattr(drv, "E0") else drv["E0"])
            evsmx = _as_tensor(drv.ES0 if hasattr(drv, "ES0") else drv["ES0"])
        rates.WTRALY = wtraly

        heavy_infiltration = self._RINold >= 1
        evaporative_demand = evsmx * (torch.sqrt(self._DSLR + 1) - torch.sqrt(self._DSLR))
        soil_evaporation = torch.minimum(evsmx, evaporative_demand + self._RINold)
        surface_water = states.SS > 1
        rates.EVW = torch.where(surface_water, evwmx, _as_tensor(0.0))
        rates.EVS = torch.where(
            surface_water, _as_tensor(0.0), torch.where(heavy_infiltration, evsmx, soil_evaporation)
        )
        self._DSLR = torch.where(
            surface_water,
            self._DSLR,
            torch.where(heavy_infiltration, _as_tensor(1.0), self._DSLR + 1),
        )

        pf, conductivity, matric_flux = self._hydraulic_state(states.SM)
        # IFUNRN is 0 or 1 per batch element. torch.where keeps both formulas
        # defined when members disagree, matching the classic water balance.
        rin_fixed = (1.0 - params.NOTINF) * rain
        rin_storm = (1.0 - params.NOTINF * self.NINFTB(rain)) * rain
        rin_pre = torch.where(params.IFUNRN == 0, rin_fixed, rin_storm)
        rin_pre = rin_pre + rates.RIRR + states.SS
        available = rin_pre + rates.RIRR - rates.EVW
        rin_pre = torch.where(
            states.SS > 0.1,
            torch.minimum(profile.SurfaceConductivity, available),
            rin_pre,
        )

        flow_max = self._maximum_boundary_flow(
            states.WC, wtraly, pf, conductivity, matric_flux, delt
        )
        rates.RIN = torch.minimum(rin_pre, flow_max[0])
        evap_layer, rates.EVS = self._evaporation_by_layer(
            states.WC, wtraly, rates.EVS, rates.RIN, delt
        )
        flow, dwc = self._apply_flow_limits(
            states.WC, wtraly, flow_max, evap_layer, rates.RIN, delt
        )
        rates.Flow = flow
        rates.DWC = dwc
        rates.BOTTOMFLOW = flow[-1]

        not_infiltrated = rain + rates.RIRR - rates.EVW - rates.RIN
        rates.DSS = torch.minimum(not_infiltrated, params.SSMAX - states.SS)
        rates.DTSR = not_infiltrated - rates.DSS
        rates.DRAINT = rain
        self._RINold = rates.RIN

    def integrate(self, day: datetime.date, delt=1.0):
        """Update layer moisture and the profile water totals."""
        states = self.states
        rates = self.rates
        profile = self.soil_profile
        wc = states.WC + rates.DWC * delt
        sm = torch.stack([wc[il] / layer.Thickness for il, layer in enumerate(profile)], dim=0)
        states.WC = wc
        states.SM = sm
        states.WTRAT = states.WTRAT + rates.WTRA * delt
        states.EVWT = states.EVWT + rates.EVW * delt
        states.EVST = states.EVST + rates.EVS * delt
        states.RAINT = states.RAINT + self._RAIN
        states.TOTINF = states.TOTINF + rates.RIN * delt
        states.TOTIRR = states.TOTIRR + rates.RIRR * delt
        states.SS = states.SS + rates.DSS * delt
        states.TSR = states.TSR + rates.DTSR * delt
        states.BOTTOMFLOWT = states.BOTTOMFLOWT + rates.BOTTOMFLOW * delt
        states.CRT = _as_tensor(0.0)

        rooting_depth = self._determine_rooting_depth()
        if torch.any(torch.abs(rooting_depth - self._RDold) > 0.001):
            profile.determine_rooting_status(rooting_depth, self._RDM)
        water = sm.new_zeros(self.params.shape)
        water_low = torch.zeros_like(water)
        water_bottom = torch.zeros_like(water)
        available_top = torch.zeros_like(water)
        available_low = torch.zeros_like(water)
        available_bottom = torch.zeros_like(water)
        for il, layer in enumerate(profile):
            water = water + states.WC[il] * layer.Wtop
            water_low = water_low + states.WC[il] * layer.Wpot
            water_bottom = water_bottom + states.WC[il] * layer.Wund
            available_top = available_top + (states.WC[il] - layer.WCW) * layer.Wtop
            available_low = available_low + (states.WC[il] - layer.WCW) * layer.Wpot
            available_bottom = available_bottom + (states.WC[il] - layer.WCW) * layer.Wund
        states.W = water
        states.WLOW = water_low
        states.WWLOW = states.W + states.WLOW
        states.WBOT = water_bottom
        states.WAVUPP = available_top
        states.WAVLOW = available_low
        states.WAVBOT = available_bottom
        self._RDold = rooting_depth
        states.SM_MEAN = states.W / torch.clamp(rooting_depth, min=1e-6)

    def finalize(self, day: datetime.date):
        """Check that the free-drainage water balance closes."""
        states = self.states
        checksum = (
            self.params.SSI
            - states.SS
            + self._WCI
            - states.WC.sum(dim=0)
            + states.RAINT
            + states.TOTIRR
            - states.WTRAT
            - states.EVWT
            - states.EVST
            - states.TSR
            - states.BOTTOMFLOWT
        )
        if torch.any(torch.abs(checksum) > 0.0001):
            msg = f"Waterbalance not closing on {day} with checksum: {checksum}"
            raise exc.WaterBalanceError(msg)

    def _hydraulic_state(self, soil_moisture):
        n_layers = soil_moisture.shape[0]
        pf = []
        conductivity = []
        matric_flux = []
        for il, layer in enumerate(self.soil_profile):
            pf_il = layer.PFfromSM(soil_moisture[il])
            pf.append(pf_il)
            conductivity.append(10.0 ** layer.CONDfromPF(pf_il))
            matric_flux.append(layer.MFPfromPF(pf_il))
        return (
            (
                torch.stack(pf, dim=0),
                torch.stack(conductivity, dim=0),
                torch.stack(matric_flux, dim=0),
            )
            if n_layers
            else (soil_moisture, soil_moisture, soil_moisture)
        )

    def _maximum_boundary_flow(
        self, water_content, transpiration, pf, conductivity, matric_flux, delt
    ):
        profile = self.soil_profile
        n_layers = len(profile)
        thickness = [layer.Thickness for layer in profile]
        flow_max = [None] * (n_layers + 1)
        flow_max[n_layers] = torch.maximum(profile[-1].CondFC, conductivity[-1])
        for il in reversed(range(n_layers)):
            if il == 0:
                limit_wet = profile.SurfaceConductivity
                limit_dry = _as_tensor(0.0)
                equal_amount = _as_tensor(0.0)
            else:
                limit_wet = (thickness[il - 1] + thickness[il]) / (
                    thickness[il - 1] / conductivity[il - 1] + thickness[il] / conductivity[il]
                )
                same_soil = profile[il - 1] == profile[il]
                if same_soil:
                    limit_dry = (
                        2.0
                        * (matric_flux[il - 1] - matric_flux[il])
                        / (thickness[il - 1] + thickness[il])
                    )
                    mean_moisture = (water_content[il - 1] + water_content[il]) / (
                        thickness[il - 1] + thickness[il]
                    )
                    equal_amount = water_content[il - 1] - thickness[il - 1] * mean_moisture
                else:
                    limit_dry = self._bisect_boundary_flow(
                        profile[il - 1],
                        profile[il],
                        pf[il - 1],
                        pf[il],
                        matric_flux[il - 1],
                        matric_flux[il],
                        thickness[il - 1],
                        thickness[il],
                    )
                    equal_amount = self._bisect_equal_potential(
                        profile[il - 1],
                        profile[il],
                        water_content[il - 1],
                        water_content[il],
                        thickness[il - 1],
                        thickness[il],
                    )
            upward = limit_dry < 0
            flow_up = torch.maximum(limit_dry, equal_amount * self.UpwardFlowLimit)
            field_capacity = profile[il - 1].WCFC if il > 0 else flow_up
            target_limit = (
                transpiration[il - 1] + field_capacity - water_content[il - 1] / delt
                if il > 0
                else flow_up
            )
            dry_target = target_limit > 0 if il > 0 else upward
            flow_up = torch.where(
                dry_target if il > 0 else upward,
                torch.maximum(flow_up, -target_limit) if il > 0 else flow_up,
                flow_up,
            )
            # Upward flow cannot empty the source layer.
            flow_up = torch.maximum(
                flow_up, flow_max[il + 1] + transpiration[il] - water_content[il] / delt
            )
            reject_upward = upward & ~dry_target if il > 0 else upward & False
            flow_down_max = torch.maximum(limit_dry, limit_wet)
            saturation_limit = (
                flow_max[il + 1] + (profile[il].WC0 - water_content[il]) / delt + transpiration[il]
            )
            downward = torch.minimum(flow_down_max, saturation_limit)
            use_upward = (
                upward & ~reject_upward if il > 0 else torch.zeros_like(upward, dtype=torch.bool)
            )
            if il == 0:
                use_upward = torch.zeros_like(limit_wet, dtype=torch.bool)
            flow_max[il] = torch.where(use_upward, flow_up, downward)
        return flow_max

    def _bisect_boundary_flow(
        self,
        layer_above,
        layer_below,
        pf_above,
        pf_below,
        mfp_above,
        mfp_below,
        thickness_above,
        thickness_below,
    ):
        """Find the pF at the interface of two different soils by bisection."""
        bound_above = pf_above
        bound_below = pf_below
        flow_above = mfp_above
        flow_below = mfp_below
        for _ in range(self.MaxFlowIter):
            midpoint = (bound_above + bound_below) / 2.0
            flow_above = 2.0 * (mfp_above - layer_above.MFPfromPF(midpoint)) / thickness_above
            flow_below = 2.0 * (-mfp_below + layer_below.MFPfromPF(midpoint)) / thickness_below
            converged = torch.abs(flow_above - flow_below) < self.TinyFlow
            shift_toward_above = torch.abs(flow_above) > torch.abs(flow_below)
            shift_toward_below = torch.abs(flow_above) < torch.abs(flow_below)
            bound_below = torch.where(converged | ~shift_toward_above, bound_below, midpoint)
            bound_above = torch.where(converged | ~shift_toward_below, bound_above, midpoint)
        return (flow_above + flow_below) / 2.0

    def _bisect_equal_potential(
        self,
        layer_above,
        layer_below,
        water_above,
        water_below,
        thickness_above,
        thickness_below,
    ):
        """Water amount that would put two neighbouring layers at equal potential.

        PCSE evaluates ``SMfromPF`` at a moisture content here. That call is
        kept so the port matches the PCSE Python implementation.
        """
        low = -water_below
        high = torch.zeros_like(water_below)
        amount = (low + high) / 2.0
        for _ in range(self.MaxFlowIter):
            amount = (low + high) / 2.0
            moisture_above = (water_above - amount) / thickness_above
            moisture_below = (water_below + amount) / thickness_below
            pf_above = layer_above.SMfromPF(moisture_above)
            pf_below = layer_below.SMfromPF(moisture_below)
            converged = torch.abs(low - high) < self.TinyFlow
            high = torch.where(converged, high, torch.where(pf_above > pf_below, amount, high))
            low = torch.where(converged, low, torch.where(pf_above > pf_below, low, amount))
        return amount

    def _evaporation_by_layer(self, water_content, transpiration, evaporation, infiltration, delt):
        remaining = evaporation
        taken = []
        for il, layer in enumerate(self.soil_profile):
            if il == 0:
                take = torch.minimum(
                    evaporation,
                    (water_content[il] - layer.WCW) / delt + infiltration - transpiration[il],
                )
            else:
                available = torch.clamp(
                    (water_content[il] - layer.WCW) / delt - transpiration[il], min=0.0
                )
                take = torch.minimum(available, torch.clamp(remaining, min=0.0))
            remaining = remaining - take if il else evaporation - take
            taken.append(take)
        profile_take = torch.stack(taken, dim=0)
        reduced = evaporation - torch.clamp(remaining, min=0.0)
        return profile_take, reduced

    def _apply_flow_limits(
        self, water_content, transpiration, flow_max, evaporation_layer, infiltration, delt
    ):
        n_layers = len(self.soil_profile)
        evap_flow = [None] * (n_layers + 1)
        evap_flow[0] = self.rates.EVS
        for il in range(1, n_layers):
            evap_flow[il] = evap_flow[il - 1] - evaporation_layer[il - 1]
        evap_flow[n_layers] = _as_tensor(0.0)

        flow = [None] * (n_layers + 1)
        dwc = [None] * n_layers
        flow[0] = infiltration - evap_flow[0]
        for il, layer in enumerate(self.soil_profile):
            maximum_loss = (water_content[il] - layer.WCFC) / delt
            excess = torch.clamp(maximum_loss + flow[il] - transpiration[il], min=0.0)
            flow[il + 1] = torch.minimum(flow_max[il + 1], excess - evap_flow[il + 1])
            dwc[il] = flow[il] - flow[il + 1] - transpiration[il]
        return torch.stack(flow, dim=0), torch.stack(dwc, dim=0)

    def _determine_rooting_depth(self):
        if "RD" in self.kiosk:
            return self.kiosk["RD"]
        return self._default_RD

    def _on_CROP_START(self):
        self._crop_ready = True

    def _on_CROP_FINISH(self):
        return

    def _on_IRRIGATE(self, amount, efficiency):
        self._RIRR = _as_tensor(amount) * _as_tensor(efficiency)

    def _setup_new_crop(self):
        self._RDM = _as_tensor(self.parameter_provider["RDMCR"])
        self.soil_profile.validate_max_rooting_depth(self._RDM)
        self.soil_profile.determine_rooting_status(self._default_RD, self._RDM)
