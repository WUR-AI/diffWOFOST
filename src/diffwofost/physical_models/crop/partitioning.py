from collections import namedtuple
from warnings import warn
import torch
from pcse import exceptions as exc
from pcse.base import SimulationObject
from pcse.decorators import prepare_states
from pcse.traitlets import Instance
from diffwofost.physical_models.base import TensorParamTemplate
from diffwofost.physical_models.base import TensorStatesTemplate
from diffwofost.physical_models.config import ComputeConfig
from diffwofost.physical_models.traitlets import Tensor
from diffwofost.physical_models.utils import AfgenTrait
from diffwofost.physical_models.utils import _broadcast_to


# Template for namedtuple containing partitioning factors
class PartioningFactors(namedtuple("partitioning_factors", "FR FL FS FO")):
    pass


def _first_tensor_item(x: torch.Tensor) -> float:
    """Returns the first element of a tensor as a python float (for logging)."""
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    if x.dim() == 0:
        return x.item()
    return x.reshape(-1)[0].item()


class _BaseDVSPartitioning(SimulationObject):
    """Shared implementation for DVS-based partitioning.

    This is intentionally private: it exists to avoid code duplication between
    the public partitioning classes.
    """

    @property
    def device(self):
        """Get device from ComputeConfig."""
        return ComputeConfig.get_device()

    @property
    def dtype(self):
        """Get dtype from ComputeConfig."""
        return ComputeConfig.get_dtype()

    class Parameters(TensorParamTemplate):
        FRTB = AfgenTrait()
        FLTB = AfgenTrait()
        FSTB = AfgenTrait()
        FOTB = AfgenTrait()

    class StateVariables(TensorStatesTemplate):
        FR = Tensor(-99.0)
        FL = Tensor(-99.0)
        FS = Tensor(-99.0)
        FO = Tensor(-99.0)
        PF = Instance(PartioningFactors)

    def _handle_partitioning_error(self, msg: str) -> None:
        """Hook for error handling (warn vs raise)."""
        warn(msg)

    def _format_partitioning_error(self, checksum, FR, FL, FS, FO) -> str:
        cs = _first_tensor_item(checksum)
        fr = _first_tensor_item(FR)
        fl = _first_tensor_item(FL)
        fs = _first_tensor_item(FS)
        fo = _first_tensor_item(FO)
        msg = f"Error in partitioning!\nChecksum: {cs:.6f}, FR: {fr:.3f}, "
        msg += f"FL: {fl:.3f}, FS: {fs:.3f}, FO: {fo:.3f}\n"
        return msg

    def _check_partitioning(self):
        """Check for partitioning errors."""
        FR = self.states.FR
        FL = self.states.FL
        FS = self.states.FS
        FO = self.states.FO
        checksum = FR + (FL + FS + FO) * (1.0 - FR) - 1.0
        if torch.any(torch.abs(checksum) >= 0.0001):
            msg = self._format_partitioning_error(checksum, FR, FL, FS, FO)
            self.logger.error(msg)
            self._handle_partitioning_error(msg)

    def _set_partitioning_states(self, FR, FL, FS, FO):
        self.states.FR = FR
        self.states.FL = FL
        self.states.FS = FS
        self.states.FO = FO
        self.states.PF = PartioningFactors(FR, FL, FS, FO)

    def _compute_partitioning_from_tables(self, DVS):
        p = self.params
        FR = p.FRTB(DVS)
        FL = p.FLTB(DVS)
        FS = p.FSTB(DVS)
        FO = p.FOTB(DVS)
        return FR, FL, FS, FO

    def _initialize_from_tables(self, kiosk, parvalues, shape=None):
        self.params = self.Parameters(parvalues, shape=shape)
        self.kiosk = kiosk
        DVS = _broadcast_to(self.kiosk["DVS"], self.params.shape)
        FR, FL, FS, FO = self._compute_partitioning_from_tables(DVS)
        self.states = self.StateVariables(
            kiosk,
            publish=["FR", "FL", "FS", "FO"],
            FR=FR,
            FL=FL,
            FS=FS,
            FO=FO,
            PF=PartioningFactors(FR, FL, FS, FO),
            shape=shape,
        )
        self._check_partitioning()

    def _update_from_tables(self):
        DVS = _broadcast_to(self.kiosk["DVS"], self.params.shape)
        FR, FL, FS, FO = self._compute_partitioning_from_tables(DVS)
        self._set_partitioning_states(FR, FL, FS, FO)
        self._check_partitioning()


class DVS_Partitioning(_BaseDVSPartitioning):
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
    | FOTB | Partitioning to storage organs as a function of development  stage   | TCr  | -    |

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

    **Outputs**

    | Name | Description                            | Pbl | Unit |
    |------|----------------------------------------|-----|------|
    | FR   | Fraction partitioned to roots          | Y   | -    |
    | FL   | Fraction partitioned to leaves         | Y   | -    |
    | FS   | Fraction partitioned to stems          | Y   | -    |
    | FO   | Fraction partitioned to storage organs | Y   | -    |

    **Gradient mapping (which parameters have a gradient):**

    | Output | Parameters influencing it |
    |--------|---------------------------|
    | FR     | FRTB, DVS                 |
    | FL     | FLTB, DVS                 |
    | FS     | FSTB, DVS                 |
    | FO     | FOTB, DVS                 |

    *Exceptions raised*

    A PartitioningError is raised if the partitioning coefficients to leaves,
    stems and storage organs on a given day do not add up to 1.
    """

    def initialize(self, day, kiosk, parvalues, shape=None):
        """Initialize the DVS_Partitioning simulation object.

        Args:
            day: Start date of the simulation.
            kiosk (VariableKiosk): Variable kiosk of this PCSE instance.
            parvalues (ParameterProvider): Object providing parameters as
                key/value pairs.
            shape (tuple | torch.Size | None): Target shape for the state and rate variables.
        """
        self._initialize_from_tables(kiosk, parvalues, shape=shape)

    @prepare_states
    def integrate(self, day, delt=1.0):
        """Update partitioning factors based on development stage (DVS)."""
        self._update_from_tables()

    def calc_rates(self, day, drv):
        """Return partitioning factors based on current DVS.

        Rate calculation does nothing for partitioning as it is a derived state.
        """
        return self.states.PF


# This class is used in `wofost81` and has NOT been tested, see #41
class DVS_Partitioning_N(_BaseDVSPartitioning):
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
    | FOTB | Partitioning to storage organs as a function of development stage    | TCr  | -    |

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

    **Outputs**

    | Name | Description                            | Pbl | Unit |
    |------|----------------------------------------|-----|------|
    | FR   | Fraction partitioned to roots          | Y   | -    |
    | FL   | Fraction partitioned to leaves         | Y   | -    |
    | FS   | Fraction partitioned to stems          | Y   | -    |
    | FO   | Fraction partitioned to storage organs | Y   | -    |

    **Gradient mapping (which parameters have a gradient):**

    | Output | Parameters influencing it  |
    |--------|----------------------------|
    | FR     | FRTB, DVS, RFTRA           |
    | FL     | FLTB, DVS                  |
    | FS     | FSTB, DVS                  |
    | FO     | FOTB, DVS                  |

    *Exceptions raised*

    A PartitioningError is raised if the partitioning coefficients to leaves,
    stems and storage organs on a given day do not add up to 1.
    """

    def _handle_partitioning_error(self, msg: str) -> None:
        raise exc.PartitioningError(msg)

    def initialize(self, day, kiosk, parameters, shape=None):
        """Initialize the DVS_Partitioning_N simulation object.

        Args:
            day: Start date of the simulation.
            kiosk (VariableKiosk): Variable kiosk of this PCSE instance.
            parameters (ParameterProvider): Dictionary with WOFOST cropdata
                key/value pairs.
            shape (tuple | torch.Size | None): Target shape for the state and rate variables.
        """
        self._initialize_from_tables(kiosk, parameters, shape=shape)

    def _calculate_stressed_fr(self, DVS: torch.Tensor, RFTRA: torch.Tensor) -> torch.Tensor:
        """Computes the FR partitioning fraction under water/oxygen stress."""
        FRTMOD = torch.max(torch.ones_like(RFTRA), 1.0 / (RFTRA + 0.5))
        return torch.min(torch.full_like(FRTMOD, 0.6), (self.params.FRTB(DVS) * FRTMOD))

    @prepare_states
    def integrate(self, day, delt=1.0):
        """Update partitioning factors based on DVS and water/oxygen stress."""
        DVS = _broadcast_to(self.kiosk["DVS"], self.params.shape)
        RFTRA = _broadcast_to(self.kiosk["RFTRA"], self.params.shape)
        FR = self._calculate_stressed_fr(DVS, RFTRA)
        FL = self.params.FLTB(DVS)
        FS = self.params.FSTB(DVS)
        FO = self.params.FOTB(DVS)
        self._set_partitioning_states(FR, FL, FS, FO)
        self._check_partitioning()

    def calc_rates(self, day, drv):
        """Return partitioning factors based on current DVS.

        Rate calculation does nothing for partitioning as it is a derived state.
        """
        return self.states.PF
