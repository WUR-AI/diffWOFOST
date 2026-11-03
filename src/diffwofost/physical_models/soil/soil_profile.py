"""Layered soil profile used by the multi-layer water balance and SNOMIN.

This is the torch port of ``pcse.soil.soil_profile``. Hydraulic lookup tables
go through diffWOFOST's differentiable ``Afgen``. Layer thickness and the pF
curves are structural inputs; volumetric thresholds derived from those curves
stay in the autograd graph.

**Gradient mapping (which parameters have a gradient):**

| Quantity | Parameters influencing it                          |
|----------|----------------------------------------------------|
| SM0, SMFCF, SMW | SMTAB, PFFieldCapacity, PFWiltingPoint   |
| MFP, conductivity | CONDTAB                                   |
| Wtop     | Rooting depth while the root front is inside the layer |

[!NOTE]
``Wtop`` is linear in rooting depth while the root front sits inside a layer.
Crossing into the next layer is a hard switch, so the gradient through that
branch is zero. Layer thickness is different: ``float(layer.Thickness)`` builds
a new tensor, so there is no gradient path back to the soil-data value.
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
        # Zero MFP at the highest pF. The paired slot keeps that pF value.
        mfp_values[-1] = sm_from_pf.new_zeros(())
        mfp_values[-2] = sm_from_pf[-2]
        for ip in range(n_points - 3, 0, -2):
            # Copy the corresponding pF value.
            mfp_values[ip - 1] = sm_from_pf[ip - 1]
            # Integral of conductivity over this pF interval.
            delta_pf = sm_from_pf[ip + 1] - sm_from_pf[ip - 1]
            added = sm_from_pf.new_zeros(())
            for gauss, weight in zip(self.Pgauss, self.Wgauss, strict=True):
                pfg = sm_from_pf[ip - 1] + gauss * delta_pf
                conductivity_value = 10.0 ** conductivity(pfg)
                added = added + conductivity_value * (10.0**pfg) * self.elog10 * weight
            mfp_values[ip] = added * delta_pf + mfp_values[ip + 2]
        Afgen.__init__(self, torch.stack(mfp_values))


class SoilLayer:
    """Intrinsic and derived properties of one soil layer.

    Required for each layer:

    | Name        | Description                                      | Unit                    |
    |-------------|--------------------------------------------------|-------------------------|
    | CONDfromPF  | Log10 of unsaturated conductivity versus pF     | log10(cm d-1), -        |
    | SMfromPF    | Volumetric moisture content versus pF           | m3 m-3, -               |
    | Thickness   | Layer thickness                                  | cm                      |
    | FSOMI       | Initial fraction of soil organic matter          | kg OM kg-1 soil         |
    | CNRatioSOMI | Initial C:N ratio of soil organic matter         | kg C kg-1 N             |
    | RHOD        | Bulk density                                     | g cm-3                  |
    | Soil_pH     | Soil pH                                          | -                       |
    | CRAIRC      | Critical air content for root aeration           | m3 air m-3 soil         |

    Derived from those inputs:

    | Name     | Description                                      | Unit           |
    |----------|--------------------------------------------------|----------------|
    | PFfromSM | Inverted moisture curve                          | -, m3 m-3      |
    | MFPfromPF| Matric flux potential versus pF                  | cm2 d-1        |
    | SM0      | Moisture content at saturation (pF = -1)         | m3 m-3         |
    | SMW      | Moisture content at wilting point                | m3 m-3         |
    | SMFCF    | Moisture content at field capacity               | m3 m-3         |
    | WC0      | Water amount at saturation                       | cm             |
    | WCW      | Water amount at wilting point                    | cm             |
    | WCFC     | Water amount at field capacity                   | cm             |
    | CondFC   | Conductivity at field capacity                   | cm d-1         |
    | CondK0   | Conductivity at saturation                       | cm d-1         |
    """

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
                f"Soil layer should have thickness between 5 and 250 cm. Current value: {thickness}"
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
        # compute hash value of this layer based on pF curves for conductivity and SM
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
    """Soil column used by the layered water balance and SNOMIN.

    The description is read from ``SoilProfileDescription`` on the parameter
    provider. This is a container of ``SoilLayer`` objects, plus the rooting
    status used for root growth and root water extraction. Layer properties
    are documented on ``SoilLayer``.

    A layer is rooted when the roots have passed its lower boundary, partially
    rooted when the root front sits inside it, potentially rooted when it lies
    between the current roots and the maximum rooting depth, and never rooted
    below that depth. The maximum rooting depth must fall on a layer boundary.

    The profile can be written as YAML. Layer types are defined once and then
    reused. In the example below, two layers of 10 cm use ``TopSoil``, three
    layers of 10, 20 and 30 cm use ``MidSoil``, a lower layer of 45 cm uses
    ``SubSoil``, and ``SubSoilType`` is 200 cm of the same subsoil. Only the
    top three layers contain organic matter (``FSOMI``).

    ```yaml
    SoilLayerTypes:
        TopSoil: &TopSoil
            SMfromPF: [-1.0, 0.366, 1.0, 0.338, 1.3, 0.304, 1.7, 0.233,
                       2.0, 0.179, 2.3, 0.135, 2.4, 0.123, 2.7, 0.094,
                       3.0, 0.073, 3.3, 0.059, 3.7, 0.046, 4.0, 0.039,
                       4.17, 0.037, 4.2, 0.036, 6.0, 0.02]
            CONDfromPF: [-1.0, 1.8451, 1.0, 1.02119, 1.3, 0.51055,
                         1.7, -0.52288, 2.0, -1.50864, 2.3, -2.56864,
                         2.4, -2.92082, 2.7, -4.01773, 3.0, -5.11919,
                         3.3, -6.22185, 3.7, -7.69897, 4.0, -8.79588,
                         4.17, -9.4318, 4.2, -9.5376, 6.0, -11.5376]
            CRAIRC: 0.090
            CNRatioSOMI: 9.0
            RHOD: 1.406
            Soil_pH: 7.4
        MidSoil: &MidSoil
            SMfromPF: [-1.0, 0.366, 1.0, 0.338, 1.3, 0.304, 1.7, 0.233,
                       2.0, 0.179, 2.3, 0.135, 2.4, 0.123, 2.7, 0.094,
                       3.0, 0.073, 3.3, 0.059, 3.7, 0.046, 4.0, 0.039,
                       4.17, 0.037, 4.2, 0.036, 6.0, 0.02]
            CONDfromPF: [-1.0, 1.8451, 1.0, 1.02119, 1.3, 0.51055,
                         1.7, -0.52288, 2.0, -1.50864, 2.3, -2.56864,
                         2.4, -2.92082, 2.7, -4.01773, 3.0, -5.11919,
                         3.3, -6.22185, 3.7, -7.69897, 4.0, -8.79588,
                         4.17, -9.4318, 4.2, -9.5376, 6.0, -11.5376]
            CRAIRC: 0.090
            CNRatioSOMI: 9.0
            RHOD: 1.406
            Soil_pH: 7.4
        SubSoil: &SubSoil
            SMfromPF: [-1.0, 0.366, 1.0, 0.338, 1.3, 0.304, 1.7, 0.233,
                       2.0, 0.179, 2.3, 0.135, 2.4, 0.123, 2.7, 0.094,
                       3.0, 0.073, 3.3, 0.059, 3.7, 0.046, 4.0, 0.039,
                       4.17, 0.037, 4.2, 0.036, 6.0, 0.02]
            CONDfromPF: [-1.0, 1.8451, 1.0, 1.02119, 1.3, 0.51055,
                         1.7, -0.52288, 2.0, -1.50864, 2.3, -2.56864,
                         2.4, -2.92082, 2.7, -4.01773, 3.0, -5.11919,
                         3.3, -6.22185, 3.7, -7.69897, 4.0, -8.79588,
                         4.17, -9.4318, 4.2, -9.5376, 6.0, -11.5376]
            CRAIRC: 0.090
            CNRatioSOMI: 9.0
            RHOD: 1.406
            Soil_pH: 7.4
    SoilProfileDescription:
        PFWiltingPoint: 4.2
        PFFieldCapacity: 2.0
        SurfaceConductivity: 70.0  # cm day-1
        SoilLayers:
        -   <<: *TopSoil
            Thickness: 10
            FSOMI: 0.02
        -   <<: *TopSoil
            Thickness: 10
            FSOMI: 0.02
        -   <<: *MidSoil
            Thickness: 10
            FSOMI: 0.01
        -   <<: *MidSoil
            Thickness: 20
            FSOMI: 0.00
        -   <<: *MidSoil
            Thickness: 30
            FSOMI: 0.00
        -   <<: *SubSoil
            Thickness: 45
            FSOMI: 0.00
        SubSoilType:
            <<: *SubSoil
            Thickness: 200
        GroundWater: null
    ```
    """

    def __init__(self, parvalues):
        super().__init__()
        description = DotMap(parvalues["SoilProfileDescription"])
        for layer_properties in description.SoilLayers:
            self.append(
                SoilLayer(layer_properties, description.PFFieldCapacity, description.PFWiltingPoint)
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
        """Set rooted, partially rooted, potentially rooted, or never-rooted weights.

        A layer whose lower boundary is above the rooting depth is rooted. The
        layer that contains the root front is partially rooted. Layers below
        the roots and above the maximum rooting depth are potentially rooted.
        The rest are never rooted. The maximum rooting depth must fall on a
        layer boundary.

        The weight of the partially rooted layer is linear in rooting depth, so
        that boundary stays differentiable. Crossing into the next layer is a
        hard switch, as in PCSE.
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
