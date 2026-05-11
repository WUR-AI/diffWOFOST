from pcse.base import SimulationObject as PCSE_SimulationObject
from typing import Dict


def initialize_component(
    component_spec: dict,
    day,
    kiosk,
    parvalues,
    shape=None,
):
    """Build one embedded model component from the override definition.

    Args:
        component_name: Canonical component name to instantiate.
        component_specs: Mapping of canonical component names to
            ``(attribute_name, default_class)`` pairs (typically
            ``Wofost72.COMPONENT_SPECS``).
        day: Current simulation day.
        kiosk: Variable kiosk shared across crop components.
        parvalues: Physical-model parameter provider.
        shape: Optional tensor broadcast shape for the component.
        component_overrides: Normalized component override mapping.

    Returns:
        The instantiated simulation component.

    The constructor call depends on whether the override provides a
    ``model``. Default physical components expect the parameter provider as
    their third positional argument, whereas ML-backed wrappers typically
    expect a model object there instead. This function centralizes that
    dispatch so callers only need to describe the override declaratively.

    """

    constructor_kwargs = dict(component_spec["kwargs"])
    component_class = component_spec["component_class"]
    if shape is not None:
        constructor_kwargs["shape"] = shape

    if component_spec["model"] is not None:
        return component_class(day, kiosk, component_spec["model"], **constructor_kwargs)

    return component_class(day, kiosk, parvalues, **constructor_kwargs)


class SimulationObject(PCSE_SimulationObject):
    """Base class for simulation objects with component management."""

    COMPONENT_SPECS: Dict[str, tuple] = {}

    def initialize_components(
        self,
        day,
        kiosk,
        parvalues,
        shape,
        component_overrides: dict | None = None,
    ) -> None:
        """Generic crop component initialization for any SimulationObject.

        Args:
            day: Start date of the simulation.
            kiosk: Variable kiosk used to read and publish crop state.
            parvalues: Parameter provider containing the physical-model
                parameters for the crop.
            shape: Target tensor shape for state and rate variables.
            component_overrides: Mapping used to replace one or more
                internal components (e.g. in WOFOST) at initialization time.
                The order of components in the mapping matter because these are
                physical models to be initialized one by one, and some
                components may depend on previously initialized ones.
        """
        for component_name, (attribute_name, default_component_spec) in self.COMPONENT_SPECS.items():
            if component_overrides is None:
                component_spec = {
                    "component_class": default_component_spec,
                    "model": None,
                    "kwargs": {},
                }
            else:
                component_spec = component_overrides[component_name]
            component = initialize_component(
                component_spec,
                day,
                kiosk,
                parvalues,
                shape=shape,
            )
            print(type(component))
            1/0
            setattr(self, attribute_name, component)
