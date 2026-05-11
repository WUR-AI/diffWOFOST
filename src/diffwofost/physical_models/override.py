from dataclasses import dataclass
from typing import Any

import torch
from pcse.base import SimulationObject

@dataclass(frozen=True)
class ComponentOverride:
    """Type-safe representation of a component override."""
    component_class: type[SimulationObject] | None = None
    model: type[torch.nn.Module] = None
    kwargs: dict[str, Any] | None = None

    def get_kwargs(self) -> dict[str, Any]:
        return self.kwargs or {}


def normalize_components(
    crop_components: dict | None,
    crop_component_specs: dict,
) -> dict[str, ComponentOverride]:
    """Convert user-facing component overrides into ComponentOverride instances.

    Args:
        crop_components: Raw override mapping from the configuration.
        crop_component_specs: Mapping of canonical component names to specs.

    Returns:
        Dictionary keyed by canonical component names with ComponentOverride values.
        containing:
        - "component_class": The class to use for the component.
        - "model": The model to use for the component, if specified in the override.
        - "kwargs": Any additional keyword arguments to pass to the component constructor.

    Raises:
        KeyError: If an unknown component name is provided.
    """
    normalized_overrides = {}

    for component_name, override in crop_components.items():
        if component_name not in crop_component_specs:
            msg = (
                f"Unknown crop component override: {component_name}. "
                f"Valid components are: {list(crop_component_specs.keys())}"
            )
            raise KeyError(msg)

        if isinstance(override, dict):
            override_dict = dict(override)
            component_class = override_dict.pop("class")
            model = override_dict.pop("model", None)
            explicit_kwargs = override_dict.pop("kwargs", {})
            constructor_kwargs = {**(explicit_kwargs or {}), **override_dict}
        else:
            component_class = override
            model = None
            constructor_kwargs = {}

        normalized_overrides[component_name] = ComponentOverride(
            component_class=component_class,
            model=model,
            kwargs=constructor_kwargs or None,
        )

    # Add defaults for any components not in overrides
    for component_name, (_, default_class) in crop_component_specs.items():
        if component_name not in normalized_overrides:
            normalized_overrides[component_name] = ComponentOverride(
                component_class=default_class,
                model=None,
                kwargs=None,
            )

    return normalized_overrides
