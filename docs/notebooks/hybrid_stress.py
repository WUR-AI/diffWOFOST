"""Tutorial-specific hybrid stress plug-in for WOFOST72.

These classes belong with ``hybrid_stress_correction.ipynb``, not with the
diffwofost library. They show how to replace the evapotranspiration stress
factor ``RFTRA`` with a tiny MLP and train it through the differentiable
engine.

``StressNN`` maps a daily feature vector to ``RFTRA ∈ [0, 1]``.
``NNStressFactor`` is the WOFOST evapotranspiration component that calls it
on every simulated day.
"""

import torch
from pcse.traitlets import Instance
from diffwofost.physical_models.config import ComputeConfig
from diffwofost.physical_models.crop.evapotranspiration import Evapotranspiration


class StressNN(torch.nn.Module):
    """Tiny MLP mapping per-day features to an `RFTRA`-style stress factor.

    Architecture: `n_features → hidden_size → 1` with a `SiLU` activation in
    between and a `sigmoid` output. The output layer is biased at
    initialization toward `sigmoid(2.2) ≈ 0.90`, so an untrained network
    starts at a *mild stress* level (about 90% of PP). This avoids two
    pathologies: (a) collapsing to zero biomass before the optimizer can
    learn anything, and (b) starting in the saturated tail of the sigmoid
    where the gradient `sigmoid'(5) ≈ 0.007` would dampen all updates by
    two orders of magnitude. At bias=2.2 the gradient is `sigmoid'(2.2) ≈
    0.087` — roughly 13× larger — so the optimizer can actually move.

    Args:
        n_features (int): Length of the input feature vector.
        hidden_size (int): Width of the hidden layer. Defaults to 16.
        init_no_stress (bool): If True, biases the output toward ~0.90 at
            initialization (mild stress, with usable gradient). Defaults to
            True.
    """

    def __init__(self, n_features, hidden_size=16, init_no_stress=True):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(n_features, hidden_size),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, 1),
        )
        if init_no_stress:
            with torch.no_grad():
                # bias=2.2 -> sigmoid(2.2) ~ 0.90, well outside the saturated
                # tail of the sigmoid so backprop gradients aren't dampened.
                # Output-layer weights are left at xavier defaults so the
                # network has full dynamic range to deviate from the bias.
                self.layers[-1].bias.fill_(2.2)
        self.to(device=ComputeConfig.get_device(), dtype=ComputeConfig.get_dtype())

    def forward(self, features):
        """Compute the stress factor from a feature vector.

        Args:
            features (torch.Tensor): 1-D tensor of input features.

        Returns:
            torch.Tensor: Scalar tensor in `[0, 1]` representing the stress
            factor (`RFTRA`).
        """
        out = self.layers(features)
        return torch.sigmoid(out).squeeze(-1)


class NNStressFactor(Evapotranspiration):
    """Standard non-layered evapotranspiration with NN-overridden `RFTRA`.

    Runs the parent's `calc_rates` to populate `TRAMX`, `EVWMX`, `EVSMX`,
    `RFWS`, `RFOS`, then overrides `RFTRA` with `nn_model(features)` and
    recomputes `TRA = TRAMX × RFTRA`. The crop module reads `k.RFTRA` from
    the kiosk and applies it as `GASS = PGASS × k.RFTRA`, so the stress
    factor enters assimilation transparently.

    The feature builder is supplied as a callable rather than baked in,
    so the same engine class can be used with different feature sets.

    **Inputs from kiosk** (read indirectly by the parent's calc_rates and
    by the supplied `feature_builder`):

    | Name | Description                              |
    |------|------------------------------------------|
    | DVS  | Crop development stage                   |
    | LAI  | Leaf area index                          |
    | SM   | Soil moisture content                    |

    **Outputs to kiosk** (overridden):

    | Name  | Description                              |
    |-------|------------------------------------------|
    | RFTRA | NN-learned stress factor on assimilation |
    | TRA   | Crop transpiration rate (TRAMX × RFTRA)  |
    """

    nn_model = Instance(torch.nn.Module)
    feature_builder = Instance(object)

    def initialize(self, day, kiosk, parvalues, nn_model, feature_builder, shape=None):
        """Initialize using the parent's ET logic and attach the NN.

        Args:
            day: Start date of the simulation.
            kiosk (VariableKiosk): Variable kiosk of this PCSE instance.
            parvalues (ParameterProvider): Parameter provider for the ET module.
            nn_model (torch.nn.Module): Network mapping per-day feature
                vector to `RFTRA` in `[0, 1]`. Must return a scalar tensor.
            feature_builder (callable): Called daily as
                `feature_builder(day, drv, kiosk)` to produce a 1-D feature
                tensor for `nn_model`.
            shape (tuple | None): Target shape for state and rate variables.
        """
        super().initialize(day, kiosk, parvalues, shape=shape)
        self.nn_model = nn_model
        self.feature_builder = feature_builder

    def calc_rates(self, day=None, drv=None):
        """Compute ET rates with `RFTRA` replaced by the NN output."""
        super().calc_rates(day, drv)
        features = self.feature_builder(day, drv, self.kiosk)
        rftra = self.nn_model(features)
        if rftra.dim() > 0 and rftra.numel() == 1:
            rftra = rftra.reshape(())
        self.rates.RFTRA = rftra
        self.rates.TRA = self.rates.TRAMX * rftra
