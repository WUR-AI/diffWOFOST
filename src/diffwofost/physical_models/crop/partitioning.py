# Copyright (c) 2004-2024 Wageningen Environmental Research, Wageningen-UR
# Allard de Wit (allard.dewit@wur.nl), March 2024
"""Assimilate partitioning models for WOFOST crop growth.

This module implements assimilate partitioning based on development stage (DVS)
and environmental stress factors.
"""

from collections import namedtuple
from warnings import warn
import torch
from pcse import exceptions as exc
from pcse.base import ParamTemplate
from pcse.base import SimulationObject
from pcse.base import StatesTemplate
from pcse.decorators import prepare_states
from pcse.traitlets import Any
from pcse.traitlets import Instance
from diffwofost.physical_models.utils import AfgenTrait
from diffwofost.physical_models.utils import _broadcast_to
from diffwofost.physical_models.utils import _get_params_shape


# Template for namedtuple containing partitioning factors
class PartioningFactors(namedtuple("partitioning_factors", "FR FL FS FO")):
    pass


class WOFOST_Partitioning(SimulationObject):
    """Class for assimilate partitioning based on development stage (DVS).

    `DVS_Partitioning` calculates the partitioning of the assimilates to roots,
    stems, leaves and storage organs using fixed partitioning tables as a
    function of crop development stage. The available assimilates are first
    split into below-ground and aboveground using the values in FRTB. In a
    second stage they are split into leaves (FLTB), stems (FSTB) and storage
    organs (FOTB).

    Since the partitioning fractions are derived from the state variable DVS
    they are regarded state variables as well.

    **Simulation parameters** (To be provided in cropdata dictionary):

    | Name | Description                                                    | Type | Unit |
    |------|----------------------------------------------------------------|------|------|
    | FRTB | Partitioning to roots as a function of development stage        | TCr  | -    |
    | FSTB | Partitioning to stems as a function of development stage        | TCr  | -    |
    | FLTB | Partitioning to leaves as a function of development stage       | TCr  | -    |
    | FOTB | Partitioning to storage organs as a function of development    | TCr  | -    |
    |      | stage                                                            |      |      |

    **State variables**

    | Name | Description                              | Pbl | Unit |
    |------|------------------------------------------|-----|------|
    | FR   | Fraction partitioned to roots            | Y   | -    |
    | FS   | Fraction partitioned to stems            | Y   | -    |
    | FL   | Fraction partitioned to leaves           | Y   | -    |
    | FO   | Fraction partitioned to storage organs   | Y   | -    |
    | PF   | Partitioning factors packed in tuple     | N   | -    |

    **Rate variables**

    None

    **External dependencies:**

    | Name | Description              | Provided by   | Unit |
    |------|--------------------------|---------------|------|
    | DVS  | Crop development stage   | DVS_Phenology | -    |

    *Exceptions raised*

    A PartitioningError is raised if the partitioning coefficients to leaves,
    stems and storage organs on a given day do not add up to 1.
    """

    params_shape = None  # Shape of the parameters tensors

    # Default values that can be overridden before instantiation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    class Parameters(ParamTemplate):
        FRTB = AfgenTrait()
        FLTB = AfgenTrait()
        FSTB = AfgenTrait()
        FOTB = AfgenTrait()

        def __init__(self, parvalues, dtype=None, device=None):
            # Get dtype and device from parent class if not provided
            if dtype is None:
                dtype = WOFOST_Partitioning.dtype
            if device is None:
                device = WOFOST_Partitioning.device

            # Call parent init
            super().__init__(parvalues)

    class StateVariables(StatesTemplate):
        FR = Any()
        FL = Any()
        FS = Any()
        FO = Any()
        PF = Instance(PartioningFactors)

        def __init__(self, kiosk, publish=None, dtype=None, device=None, **kwargs):
            # Get dtype and device from parent class if not provided
            if dtype is None:
                dtype = WOFOST_Partitioning.dtype
            if device is None:
                device = WOFOST_Partitioning.device

            # Set default values using the provided dtype and device if not in kwargs
            if "FR" not in kwargs:
                kwargs["FR"] = torch.tensor(-99.0, dtype=dtype, device=device)
            if "FL" not in kwargs:
                kwargs["FL"] = torch.tensor(-99.0, dtype=dtype, device=device)
            if "FS" not in kwargs:
                kwargs["FS"] = torch.tensor(-99.0, dtype=dtype, device=device)
            if "FO" not in kwargs:
                kwargs["FO"] = torch.tensor(-99.0, dtype=dtype, device=device)

            # Call parent init
            super().__init__(kiosk, publish=publish, **kwargs)

    def initialize(self, day, kiosk, parvalues):
        """Initialize the DVS_Partitioning simulation object.

        Args:
            day: Start date of the simulation.
            kiosk (VariableKiosk): Variable kiosk of this PCSE instance.
            parvalues (ParameterProvider): Object providing parameters as
                key/value pairs.
        """
        self.params = self.Parameters(parvalues, dtype=self.dtype, device=self.device)
        self.kiosk = kiosk
        self.params_shape = _get_params_shape(self.params)

        # initial partitioning factors (pf)
        DVS = torch.as_tensor(self.kiosk["DVS"], dtype=self.dtype, device=self.device)
        FR = self.params.FRTB(DVS).to(dtype=self.dtype, device=self.device)
        FL = self.params.FLTB(DVS).to(dtype=self.dtype, device=self.device)
        FS = self.params.FSTB(DVS).to(dtype=self.dtype, device=self.device)
        FO = self.params.FOTB(DVS).to(dtype=self.dtype, device=self.device)

        # Broadcast to params_shape
        FR = _broadcast_to(FR, self.params_shape, dtype=self.dtype, device=self.device)
        FL = _broadcast_to(FL, self.params_shape, dtype=self.dtype, device=self.device)
        FS = _broadcast_to(FS, self.params_shape, dtype=self.dtype, device=self.device)
        FO = _broadcast_to(FO, self.params_shape, dtype=self.dtype, device=self.device)

        # Pack partitioning factors into tuple
        PF = PartioningFactors(FR, FL, FS, FO)

        # Initial states
        self.states = self.StateVariables(
            kiosk,
            publish=["FR", "FL", "FS", "FO"],
            FR=FR,
            FL=FL,
            FS=FS,
            FO=FO,
            PF=PF,
            dtype=self.dtype,
            device=self.device,
        )
        self._check_partitioning()

    def _check_partitioning(self):
        """Check for partitioning errors."""
        FR = self.states.FR
        FL = self.states.FL
        FS = self.states.FS
        FO = self.states.FO
        checksum = FR + (FL + FS + FO) * (1.0 - FR) - 1.0
        if torch.any(torch.abs(checksum) >= 0.0001):
            # Extract scalar values for error message
            cs = checksum.item() if checksum.dim() == 0 else checksum[0].item()
            fr = FR.item() if FR.dim() == 0 else FR[0].item()
            fl = FL.item() if FL.dim() == 0 else FL[0].item()
            fs = FS.item() if FS.dim() == 0 else FS[0].item()
            fo = FO.item() if FO.dim() == 0 else FO[0].item()
            msg = f"Error in partitioning!\nChecksum: {cs:.6f}, FR: {fr:.3f}, "
            msg += f"FL: {fl:.3f}, FS: {fs:.3f}, FO: {fo:.3f}\n"
            self.logger.error(msg)
            warn(msg)

    #             raise exc.PartitioningError(msg)

    @prepare_states
    def integrate(self, day, delt=1.0):
        """Update partitioning factors based on development stage (DVS)."""
        params = self.params

        DVS = torch.as_tensor(self.kiosk["DVS"], dtype=self.dtype, device=self.device)
        self.states.FR = _broadcast_to(
            params.FRTB(DVS).to(dtype=self.dtype, device=self.device),
            self.params_shape,
            dtype=self.dtype,
            device=self.device,
        )
        self.states.FL = _broadcast_to(
            params.FLTB(DVS).to(dtype=self.dtype, device=self.device),
            self.params_shape,
            dtype=self.dtype,
            device=self.device,
        )
        self.states.FS = _broadcast_to(
            params.FSTB(DVS).to(dtype=self.dtype, device=self.device),
            self.params_shape,
            dtype=self.dtype,
            device=self.device,
        )
        self.states.FO = _broadcast_to(
            params.FOTB(DVS).to(dtype=self.dtype, device=self.device),
            self.params_shape,
            dtype=self.dtype,
            device=self.device,
        )

        # Pack partitioning factors into tuple
        self.states.PF = PartioningFactors(
            self.states.FR, self.states.FL, self.states.FS, self.states.FO
        )

        self._check_partitioning()

    def calc_rates(self, day, drv):
        """Return partitioning factors based on current DVS.

        Rate calculation does nothing for partitioning as it is a derived state.
        """
        return self.states.PF


class WOFOST_Partitioning_N(SimulationObject):
    """Class for assimilate partitioning based on development stage (DVS) with N stress.

    `DVS_Partitioning_N` calculates the partitioning of the assimilates to roots,
    stems, leaves and storage organs using fixed partitioning tables as a
    function of crop development stage. The only difference with the normal
    partitioning class is the effect of nitrogen stress on partitioning to
    leaves. The available assimilates are first split into below-ground and
    aboveground using the values in FRTB. In a second stage they are split into
    leaves (FLTB), stems (FSTB) and storage organs (FOTB).

    Since the partitioning fractions are derived from the state variable DVS
    they are regarded state variables as well.

    **Simulation parameters** (To be provided in cropdata dictionary):

    | Name | Description                                                    | Type | Unit |
    |------|----------------------------------------------------------------|------|------|
    | FRTB | Partitioning to roots as a function of development stage        | TCr  | -    |
    | FSTB | Partitioning to stems as a function of development stage        | TCr  | -    |
    | FLTB | Partitioning to leaves as a function of development stage       | TCr  | -    |
    | FOTB | Partitioning to storage organs as a function of development    | TCr  | -    |
    |      | stage                                                            |      |      |

    **State variables**

    | Name | Description                              | Pbl | Unit |
    |------|------------------------------------------|-----|------|
    | FR   | Fraction partitioned to roots            | Y   | -    |
    | FS   | Fraction partitioned to stems            | Y   | -    |
    | FL   | Fraction partitioned to leaves           | Y   | -    |
    | FO   | Fraction partitioned to storage organs   | Y   | -    |
    | PF   | Partitioning factors packed in tuple     | N   | -    |

    **Rate variables**

    None

    **External dependencies:**

    | Name  | Description                                    | Provided by              | Unit |
    |-------|------------------------------------------------|--------------------------|------|
    | DVS   | Crop development stage                         | DVS_Phenology            | -    |
    | RFTRA | Reduction factor for transpiration (water &    | Water & Oxygen dynamics  | -    |
    |       | oxygen stress)                                 |                          |      |

    *Exceptions raised*

    A PartitioningError is raised if the partitioning coefficients to leaves,
    stems and storage organs on a given day do not add up to 1.
    """

    params_shape = None  # Shape of the parameters tensors

    # Default values that can be overridden before instantiation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    class Parameters(ParamTemplate):
        FRTB = AfgenTrait()
        FLTB = AfgenTrait()
        FSTB = AfgenTrait()
        FOTB = AfgenTrait()

        def __init__(self, parvalues, dtype=None, device=None):
            # Get dtype and device from parent class if not provided
            if dtype is None:
                dtype = WOFOST_Partitioning_N.dtype
            if device is None:
                device = WOFOST_Partitioning_N.device

            # Call parent init
            super().__init__(parvalues)

    class StateVariables(StatesTemplate):
        FR = Any()
        FL = Any()
        FS = Any()
        FO = Any()
        PF = Instance(PartioningFactors)

        def __init__(self, kiosk, publish=None, dtype=None, device=None, **kwargs):
            # Get dtype and device from parent class if not provided
            if dtype is None:
                dtype = WOFOST_Partitioning_N.dtype
            if device is None:
                device = WOFOST_Partitioning_N.device

            # Set default values using the provided dtype and device if not in kwargs
            if "FR" not in kwargs:
                kwargs["FR"] = torch.tensor(-99.0, dtype=dtype, device=device)
            if "FL" not in kwargs:
                kwargs["FL"] = torch.tensor(-99.0, dtype=dtype, device=device)
            if "FS" not in kwargs:
                kwargs["FS"] = torch.tensor(-99.0, dtype=dtype, device=device)
            if "FO" not in kwargs:
                kwargs["FO"] = torch.tensor(-99.0, dtype=dtype, device=device)

            # Call parent init
            super().__init__(kiosk, publish=publish, **kwargs)

    def initialize(self, day, kiosk, parameters):
        """Initialize the DVS_Partitioning_N simulation object.

        Args:
            day: Start date of the simulation.
            kiosk (VariableKiosk): Variable kiosk of this PCSE instance.
            parameters (ParameterProvider): Dictionary with WOFOST cropdata
                key/value pairs.
        """
        self.params = self.Parameters(parameters, dtype=self.dtype, device=self.device)
        self.kiosk = kiosk
        self.params_shape = _get_params_shape(self.params)

        # initial partitioning factors (pf)
        DVS = torch.as_tensor(self.kiosk["DVS"], dtype=self.dtype, device=self.device)
        FR = self.params.FRTB(DVS).to(dtype=self.dtype, device=self.device)
        FL = self.params.FLTB(DVS).to(dtype=self.dtype, device=self.device)
        FS = self.params.FSTB(DVS).to(dtype=self.dtype, device=self.device)
        FO = self.params.FOTB(DVS).to(dtype=self.dtype, device=self.device)

        # Broadcast to params_shape
        FR = _broadcast_to(FR, self.params_shape, dtype=self.dtype, device=self.device)
        FL = _broadcast_to(FL, self.params_shape, dtype=self.dtype, device=self.device)
        FS = _broadcast_to(FS, self.params_shape, dtype=self.dtype, device=self.device)
        FO = _broadcast_to(FO, self.params_shape, dtype=self.dtype, device=self.device)

        # Pack partitioning factors into tuple
        PF = PartioningFactors(FR, FL, FS, FO)

        # Initial states
        self.states = self.StateVariables(
            kiosk,
            publish=["FR", "FL", "FS", "FO"],
            FR=FR,
            FL=FL,
            FS=FS,
            FO=FO,
            PF=PF,
            dtype=self.dtype,
            device=self.device,
        )
        self._check_partitioning()

    def _check_partitioning(self):
        """Check for partitioning errors."""
        FR = self.states.FR
        FL = self.states.FL
        FS = self.states.FS
        FO = self.states.FO
        checksum = FR + (FL + FS + FO) * (1.0 - FR) - 1.0
        if torch.any(torch.abs(checksum) >= 0.0001):
            cs = checksum.item() if checksum.dim() == 0 else checksum[0].item()
            fr = FR.item() if FR.dim() == 0 else FR[0].item()
            fl = FL.item() if FL.dim() == 0 else FL[0].item()
            fs = FS.item() if FS.dim() == 0 else FS[0].item()
            fo = FO.item() if FO.dim() == 0 else FO[0].item()
            msg = f"Error in partitioning!\nChecksum: {cs:.6f}, FR: {fr:.3f}, "
            msg += f"FL: {fl:.3f}, FS: {fs:.3f}, FO: {fo:.3f}\n"
            self.logger.error(msg)
            raise exc.PartitioningError(msg)

    @prepare_states
    def integrate(self, day, delt=1.0):
        """Update partitioning factors based on DVS and water/oxygen stress."""
        p = self.params
        s = self.states
        k = self.kiosk

        # Get RFTRA from kiosk and ensure it's a tensor
        RFTRA = torch.as_tensor(k["RFTRA"], dtype=self.dtype, device=self.device)
        DVS = torch.as_tensor(k["DVS"], dtype=self.dtype, device=self.device)

        # Calculate FRTMOD with water/oxygen stress effect
        FRTMOD = torch.max(torch.ones_like(RFTRA), 1.0 / (RFTRA + 0.5))

        # Update partitioning fractions
        s.FR = _broadcast_to(
            torch.min(
                torch.full_like(FRTMOD, 0.6),
                (p.FRTB(DVS).to(dtype=self.dtype, device=self.device) * FRTMOD),
            ),
            self.params_shape,
            dtype=self.dtype,
            device=self.device,
        )
        s.FL = _broadcast_to(
            p.FLTB(DVS).to(dtype=self.dtype, device=self.device),
            self.params_shape,
            dtype=self.dtype,
            device=self.device,
        )
        s.FS = _broadcast_to(
            p.FSTB(DVS).to(dtype=self.dtype, device=self.device),
            self.params_shape,
            dtype=self.dtype,
            device=self.device,
        )
        s.FO = _broadcast_to(
            p.FOTB(DVS).to(dtype=self.dtype, device=self.device),
            self.params_shape,
            dtype=self.dtype,
            device=self.device,
        )

        # Pack partitioning factors into tuple
        s.PF = PartioningFactors(s.FR, s.FL, s.FS, s.FO)

        self._check_partitioning()

    def calc_rates(self, day, drv):
        """Return partitioning factors based on current DVS.

        Rate calculation does nothing for partitioning as it is a derived state.
        """
        return self.states.PF
