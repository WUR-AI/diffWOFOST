"""Layered soil profile used by the multi-layer water balance and SNOMIN.

This is the torch port of ``pcse.soil.soil_profile``. Hydraulic lookup tables
go through diffWOFOST's differentiable ``Afgen``. Layer thickness and the pF
curves are structural inputs; volumetric thresholds derived from those curves
stay in the autograd graph.
"""

import torch
from pcse import exceptions as exc
from pcse.util import DotMap
from diffwofost.physical_models.config import ComputeConfig
from diffwofost.physical_models.utils import Afgen


def _as_tensor(value, dtype=None, device=None) -> torch.Tensor:
    dtype = ComputeConfig.get_dtype() if dtype is None else dtype
    device = ComputeConfig.get_device() if device is None else device
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype, device=device)
    return torch.tensor(value, dtype=dtype, device=device)


class MFPCurve(Afgen):
    """Matric flux potential as a function of pF, integrated from the pF curves."""

    elog10 = 2.302585092994
    Pgauss = (0.0469100770, 0.2307653449, 0.5000000000, 0.7692346551, 0.9530899230)
    Wgauss = (0.1184634425, 0.2393143352, 0.2844444444, 0.2393143352, 0.1184634425)

    def __init__(self, sm_from_pf, cond_from_pf):
        sm_from_pf = _as_tensor(sm_from_pf)
        conductivity = Afgen(cond_from_pf)
        # Build the table by rebinding list entries so autograd is not broken
        # by in-place writes. Odd slots hold MFP, even slots hold pF.
        n_points = sm_from_pf.shape[0]
        mfp_values = [sm_from_pf.new_zeros(()) for _ in range(n_points)]
        mfp_values[-1] = sm_from_pf.new_zeros(())
        mfp_values[-2] = sm_from_pf[-2]
        for ip in range(n_points - 3, 0, -2):
            mfp_values[ip - 1] = sm_from_pf[ip - 1]
            delta_pf = sm_from_pf[ip + 1] - sm_from_pf[ip - 1]
            added = sm_from_pf.new_zeros(())
            for gauss, weight in zip(self.Pgauss, self.Wgauss, strict=True):
                pfg = sm_from_pf[ip - 1] + gauss * delta_pf
                conductivity_value = 10.0 ** conductivity(pfg)
                added = added + conductivity_value * (10.0**pfg) * self.elog10 * weight
            mfp_values[ip] = added * delta_pf + mfp_values[ip + 2]
        Afgen.__init__(self, torch.stack(mfp_values))


class SoilLayer:
    """Intrinsic and derived properties of one soil layer."""

    def __init__(self, layer, pf_field_capacity, pf_wilting_point):
        sm_table = list(layer.SMfromPF)
        cond_table = list(layer.CONDfromPF)
        self.SMfromPF = Afgen(sm_table)
        self.CONDfromPF = Afgen(cond_table)
        self.PFfromSM = Afgen(_invert_pf_table(sm_table))
        self.MFPfromPF = MFPCurve(sm_table, cond_table)

        self.CNRatioSOMI = _as_tensor(layer.CNRatioSOMI)
        self.FSOMI = _as_tensor(layer.FSOMI)
        self.RHOD = _as_tensor(layer.RHOD)
        self.CRAIRC = _as_tensor(layer.CRAIRC)
        self.Soil_pH = _as_tensor(layer.Soil_pH)

        thickness = float(layer.Thickness)
        if not 5 <= thickness <= 250:
            msg = (
                "Soil layer should have thickness between 5 and 250 cm. "
                f"Current value: {thickness}"
            )
            raise exc.PCSEError(msg)
        self.Thickness = _as_tensor(thickness)

        pf_fc = _as_tensor(pf_field_capacity)
        pf_wp = _as_tensor(pf_wilting_point)
        self.SM0 = self.SMfromPF(_as_tensor(-1.0))
        self.SMFCF = self.SMfromPF(pf_fc)
        self.SMW = self.SMfromPF(pf_wp)
        self.WC0 = self.SM0 * self.Thickness
        self.WCW = self.SMW * self.Thickness
        self.WCFC = self.SMFCF * self.Thickness
        self.CondFC = 10.0 ** self.CONDfromPF(pf_fc)
        self.CondK0 = 10.0 ** self.CONDfromPF(_as_tensor(-1.0))
        self.Wtop = _as_tensor(0.0)
        self.Wpot = _as_tensor(0.0)
        self.Wund = _as_tensor(0.0)
        self._hash = hash((tuple(sm_table), tuple(cond_table)))

    @property
    def Thickness_m(self) -> torch.Tensor:
        """Layer thickness in metres."""
        return self.Thickness * 1e-2

    @property
    def RHOD_kg_per_m3(self) -> torch.Tensor:
        """Bulk density in kg m-3."""
        return self.RHOD * 1e3

    def __eq__(self, other):
        return isinstance(other, SoilLayer) and self._hash == other._hash

    def __hash__(self):
        return self._hash


class SoilProfile(list):
    """Soil column as a list of ``SoilLayer`` objects plus profile-level attributes."""

    def __init__(self, parvalues):
        super().__init__()
        description = DotMap(parvalues["SoilProfileDescription"])
        for layer_properties in description.SoilLayers:
            self.append(
                SoilLayer(
                    layer_properties, description.PFFieldCapacity, description.PFWiltingPoint
                )
            )
        for attr, value in description.items():
            if attr == "SoilLayers":
                continue
            if attr == "SubSoilType":
                value = SoilLayer(value, description.PFFieldCapacity, description.PFWiltingPoint)
            if attr == "SurfaceConductivity" and value is not None:
                value = _as_tensor(value)
            setattr(self, attr, value)
        if not hasattr(self, "GroundWater"):
            self.GroundWater = None

    def determine_rooting_status(self, rooting_depth, maximum_rooting_depth) -> None:
        """Set rooted / potentially rooted weights for the current rooting depth.

        The weight of a layer that the root front sits in is linear in rooting
        depth, so that boundary stays differentiable. Crossing into another
        layer is a hard switch, as in PCSE.
        """
        depth = _as_tensor(rooting_depth)
        maximum = _as_tensor(maximum_rooting_depth)
        upper = torch.zeros_like(depth)
        for layer in self:
            lower = upper + layer.Thickness
            rooted = lower <= depth
            partial = (upper < depth) & (depth < lower)
            potential = (depth <= upper) & (lower <= maximum)
            # A layer cannot be both partially rooted and potentially rooted.
            potential = potential & ~partial & ~rooted
            partial_weight = 1.0 - (lower - depth) / layer.Thickness
            layer.Wtop = torch.where(rooted, 1.0, torch.where(partial, partial_weight, 0.0))
            layer.Wpot = torch.where(
                rooted,
                0.0,
                torch.where(partial, 1.0 - layer.Wtop, torch.where(potential, 1.0, 0.0)),
            )
            layer.Wund = torch.where(rooted | partial | potential, 0.0, 1.0)
            upper = lower

    def validate_max_rooting_depth(self, maximum_rooting_depth) -> None:
        """Require the maximum rooting depth to fall on a layer boundary."""
        maximum = _as_tensor(maximum_rooting_depth).reshape(-1)
        tiny = 0.01
        for rdm in maximum:
            lower = rdm.new_zeros(())
            for layer in self:
                lower = lower + layer.Thickness
                if torch.abs(rdm - lower) < tiny:
                    break
            else:
                msg = (
                    "Current maximum rooting depth "
                    f"({float(rdm)}) does not coincide with a layer boundary!"
                )
                raise exc.PCSEError(msg)

    def get_max_rootable_depth(self) -> torch.Tensor:
        """Lower boundary of the deepest soil layer, in cm."""
        depth = self[0].Thickness.new_zeros(())
        for layer in self:
            depth = depth + layer.Thickness
        return depth


def _invert_pf_table(sm_from_pf) -> list:
    """Invert an SMfromPF table so it can be queried as pF from soil moisture."""
    pairs = list(zip(reversed(sm_from_pf[1::2]), reversed(sm_from_pf[0::2]), strict=True))
    inverted = []
    for moisture, pf in pairs:
        inverted.extend((moisture, pf))
    return inverted
