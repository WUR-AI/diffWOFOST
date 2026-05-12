from pcse.base import SimulationObject
from diffwofost.physical_models.override import ComponentOverride


def initialize_component(
    component_spec: ComponentOverride,
    day,
    kiosk,
    parvalues,
    shape=None,
):
    """Build one embedded model component from the override definition.

    Args:
        component_spec: Specification of the component to be initialized.
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
    constructor_kwargs = dict(component_spec.kwargs or {})
    component_class = component_spec.component_class
    if shape is not None:
        constructor_kwargs["shape"] = shape

    if component_spec.model is not None:
        return component_class(day, kiosk, component_spec.model, **constructor_kwargs)

    return component_class(day, kiosk, parvalues, **constructor_kwargs)


def initialize_components(
    simulation_object: SimulationObject,
    day,
    kiosk,
    parvalues,
    shape=None,
    component_overrides: dict | None = None,
) -> None:
    """Generic crop component initialization for any SimulationObject.

    Args:
        simulation_object: The SimulationObject for which to initialize the
            components.
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
    for component_name, (attribute_name, default_spec) in simulation_object.COMPONENT_SPECS.items():
        if component_overrides is None:
            component_spec = ComponentOverride.from_default(default_spec)
        else:
            component_spec = component_overrides[component_name]
        component = initialize_component(
            component_spec,
            day,
            kiosk,
            parvalues,
            shape=shape,
        )
        setattr(simulation_object, attribute_name, component)
