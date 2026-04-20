import torch
from pcse.traitlets import Instance
from diffwofost.physical_models.config import ComputeConfig
from diffwofost.physical_models.crop.partitioning import PartioningFactors
from diffwofost.physical_models.crop.partitioning import _BaseDVSPartitioning
from diffwofost.physical_models.utils import _broadcast_to


class PartitioningMLP(torch.nn.Module):
    """One-hidden-layer MLP for DVS-based partitioning.

    `PartitioningMLP` maps the crop development stage `DVS` directly to the
    four partitioning factors `FR`, `FL`, `FS`, and `FO` through a compact
    one-hidden-layer MLP. It is intentionally simpler than `PartitioningNN` and
    serves as a lightweight baseline for comparing NN architectures.

    The network predicts four logits. The root fraction `FR` is mapped through
    a sigmoid so it remains in the interval [0, 1]. The remaining logits are
    mapped through a softmax so that `FL + FS + FO = 1`.
    """

    def __init__(self, hidden_size=8):
        """Initialize the baseline partitioning network.

        Args:
            hidden_size (int): Width of the hidden layer. Defaults to 8.
        """
        super().__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_size),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, 4),
        )

        with torch.no_grad():
            torch.nn.init.xavier_uniform_(self.network[0].weight)
            self.network[0].bias.zero_()
            torch.nn.init.xavier_uniform_(self.network[2].weight)
            self.network[2].bias.zero_()

        self.to(device=ComputeConfig.get_device(), dtype=ComputeConfig.get_dtype())

    def forward(self, dvs):
        """Compute partitioning factors from the development stage.

        Args:
            dvs (float | torch.Tensor): Crop development stage.

        Returns:
            PartioningFactors: Named tuple containing `FR`, `FL`, `FS`, and `FO`.
        """
        dvs = torch.as_tensor(
            dvs, dtype=ComputeConfig.get_dtype(), device=ComputeConfig.get_device()
        )
        is_scalar = dvs.ndim == 0
        if is_scalar:
            dvs = dvs.unsqueeze(0)

        outputs = self.network(dvs.unsqueeze(-1))
        fr = torch.sigmoid(outputs[..., 0])
        above_ground = torch.softmax(outputs[..., 1:], dim=-1)

        if is_scalar:
            fr = fr.squeeze(0)
            above_ground = above_ground.squeeze(0)

        return PartioningFactors(
            FR=fr,
            FL=above_ground[..., 0],
            FS=above_ground[..., 1],
            FO=above_ground[..., 2],
        )


class PartitioningNN(torch.nn.Module):
    """DVS-based partitioning network with lifted stage features and two heads.

    `PartitioningNN` is a more structured alternative to `PartitioningMLP`.
    It first expands `DVS` into a small set of smooth stage-dependent features,
    processes those with a shared nonlinear trunk, and then predicts the root
    fraction `FR` and the above-ground fractions `FL`, `FS`, and `FO` with
    separate output heads.

    This architecture is better aligned with the original process-based
    partitioning logic, where different growth stages can lead to distinct
    allocation regimes.

    Args:
        hidden_size (int): Width of the hidden layers. Defaults to 32.
    """

    def __init__(self, hidden_size=32):
        """Initialize the structured partitioning network.

        Args:
            hidden_size (int): Width of the shared hidden layers. Defaults to 32.
        """
        super().__init__()

        head_hidden_size = max(hidden_size // 2, 8)
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(4, hidden_size),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.SiLU(),
        )
        self.fr_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, head_hidden_size),
            torch.nn.SiLU(),
            torch.nn.Linear(head_hidden_size, 1),
        )
        self.shoot_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, head_hidden_size),
            torch.nn.SiLU(),
            torch.nn.Linear(head_hidden_size, 3),
        )

        with torch.no_grad():
            torch.nn.init.xavier_uniform_(self.trunk[0].weight)
            self.trunk[0].bias.zero_()
            torch.nn.init.xavier_uniform_(self.trunk[2].weight)
            self.trunk[2].bias.zero_()
            torch.nn.init.xavier_uniform_(self.fr_head[0].weight)
            self.fr_head[0].bias.zero_()
            torch.nn.init.xavier_uniform_(self.fr_head[2].weight)
            self.fr_head[2].bias.zero_()
            torch.nn.init.xavier_uniform_(self.shoot_head[0].weight)
            self.shoot_head[0].bias.zero_()
            torch.nn.init.xavier_uniform_(self.shoot_head[2].weight)
            self.shoot_head[2].bias.zero_()

        self.to(device=ComputeConfig.get_device(), dtype=ComputeConfig.get_dtype())

    def forward(self, dvs):
        """Compute partitioning factors from the development stage.

        Args:
            dvs (float | torch.Tensor): Crop development stage.

        Returns:
            PartioningFactors: Named tuple containing `FR`, `FL`, `FS`, and `FO`.
        """
        dvs = torch.as_tensor(
            dvs,
            dtype=ComputeConfig.get_dtype(),
            device=ComputeConfig.get_device(),
        )
        is_scalar = dvs.ndim == 0
        if is_scalar:
            dvs = dvs.unsqueeze(0)

        features = torch.stack(
            (
                dvs,
                dvs**2,
                torch.relu(dvs - 1.0),
                torch.relu(1.0 - dvs),
            ),
            dim=-1,
        )
        latent = self.trunk(features)
        fr = torch.sigmoid(self.fr_head(latent)[..., 0])
        above_ground = torch.softmax(self.shoot_head(latent), dim=-1)

        if is_scalar:
            fr = fr.squeeze(0)
            above_ground = above_ground.squeeze(0)

        return PartioningFactors(
            FR=fr,
            FL=above_ground[..., 0],
            FS=above_ground[..., 1],
            FO=above_ground[..., 2],
        )


class DVS_Partitioning_NN(_BaseDVSPartitioning):
    """Drop-in wrapper for using a neural-network model as a partitioning component.

    `DVS_Partitioning_NN` mirrors the interface of `DVS_Partitioning` so that a
    neural-network model can be plugged into `Wofost72` anywhere the standard
    rule-based partitioning module is expected.

    The wrapped model must accept `DVS` as input and return a
    :class:`PartioningFactors` namedtuple containing `FR`, `FL`, `FS`, and `FO`.
    The wrapper handles publishing these values to the kiosk and storing them in
    the same state layout as the physical partitioning implementation.

    **External dependencies:**

    | Name | Description              | Provided by   | Unit |
    |------|--------------------------|---------------|------|
    | DVS  | Crop development stage   | DVS_Phenology | -    |

    **State variables**

    | Name | Description                              | Pbl | Unit |
    |------|------------------------------------------|-----|------|
    | FR   | Fraction partitioned to roots            | Y   | -    |
    | FS   | Fraction partitioned to stems            | Y   | -    |
    | FL   | Fraction partitioned to leaves           | Y   | -    |
    | FO   | Fraction partitioned to storage organs   | Y   | -    |
    | PF   | Partitioning factors packed in tuple     | N   | -    |
    """

    nn_model = Instance(torch.nn.Module)

    def initialize(self, day, kiosk, nn_model, shape=None):
        """Initialize the DVS_Partitioning_NN simulation object.

        Args:
            day: Start date of the simulation.
            kiosk (VariableKiosk): Variable kiosk of this PCSE instance.
            nn_model (torch.nn.Module): Network that maps DVS → PartioningFactors.
            shape (tuple | torch.Size | None): Target shape for batch simulations.
        """
        self._device = ComputeConfig.get_device()
        self._dtype = ComputeConfig.get_dtype()
        self.nn_model = nn_model
        self.kiosk = kiosk

        DVS = _broadcast_to(self.kiosk["DVS"], shape or ())
        pf = self.nn_model(DVS)
        self.states = self.StateVariables(
            kiosk,
            publish=["FR", "FL", "FS", "FO"],
            FR=pf.FR,
            FL=pf.FL,
            FS=pf.FS,
            FO=pf.FO,
            PF=PartioningFactors(pf.FR, pf.FL, pf.FS, pf.FO),
            shape=shape,
        )

    def integrate(self, day, delt=1.0):
        """Update partitioning factors by running the network on the current DVS.

        Args:
            day: Current simulation day.
            delt (float): Integration step size. Included for interface
                compatibility and ignored in this implementation.
        """
        DVS = _broadcast_to(self.kiosk["DVS"], self.states.shape)
        pf = self.nn_model(DVS)
        self._set_partitioning_states(pf.FR, pf.FL, pf.FS, pf.FO)

    def calc_rates(self, day, drv):
        """Return the current partitioning factors.

        Args:
            day: Current simulation day.
            drv: Driving variables object. Included for interface compatibility.

        Returns:
            PartioningFactors: Current partitioning factors stored in the state.
        """
        return self.states.PF
