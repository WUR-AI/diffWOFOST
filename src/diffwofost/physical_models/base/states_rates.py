from pcse.base import ParamTemplate
from pcse.base import RatesTemplate
from pcse.base import StatesTemplate
from pcse.traitlets import HasTraits
from ..traitlets import Tensor
from ..utils import AfgenTrait


class TensorContainer(HasTraits):
    def __init__(self, shape=None, do_not_broadcast=None, **variables):
        """Container of tensor variables.

        It includes functionality to broadcast variables to a common shape. This common shape can
        be inferred from the container's tensor and AFGEN variables, or it can be set as an input
        argument.

        Args:
            shape (tuple | torch.Size, optional): Shape to which the variables in the container
                are broadcasted. If given, it should match the shape of all the input variables that
                already have dimensions. Defaults to None.
            do_not_broadcast (list, optional): Name of the variables that are not broadcasted
                to the container shape. Defaults to None, which means that all variables are
                broadcasted.
            variables (dict): Collection of variables to initialize the container, as key-value
                pairs.
        """
        self._shape = ()
        self._do_not_broadcast = [] if do_not_broadcast is None else do_not_broadcast
        HasTraits.__init__(self, **variables)
        self._broadcast(shape)

    def _broadcast(self, shape=None):
        # Identify which variables should be broadcasted. Also check that the input shape is
        # compatible with the existing variable shapes
        vars_to_broadcast = self._get_vars_to_broadcast()
        vars_shape = self._get_vars_shape()
        if shape and vars_shape and vars_shape != shape:
            raise ValueError(f"Input shape {shape} does not match variable shape {vars_shape}")
        shape = tuple(shape or vars_shape)

        # Broadcast all required variables to the identified shape.
        for varname, var in vars_to_broadcast.items():
            try:
                broadcasted = var.expand(shape)
            except RuntimeError as error:
                raise ValueError(f"Cannot broadcast {varname} to shape {shape}") from error
            setattr(self, varname, broadcasted)

        # Finally, update the shape of the container
        self.shape = shape

    def _get_vars_to_broadcast(self):
        vars = {}
        for varname, trait in self.traits().items():
            if varname not in self._do_not_broadcast:
                if isinstance(trait, Tensor):
                    vars[varname] = getattr(self, varname)
        return vars

    def _get_vars_shape(self):
        shape = ()
        for varname, trait in self.traits().items():
            if varname not in self._do_not_broadcast:
                if isinstance(trait, Tensor) or isinstance(trait, AfgenTrait):
                    var = getattr(self, varname)
                    if not var.shape or shape == var.shape:
                        continue
                    elif var.shape and not shape:
                        shape = tuple(var.shape)
                    else:
                        raise ValueError(
                            f"Incompatible shapes within variables: {shape} and {var.shape}"
                        )
        return shape

    @property
    def shape(self):
        """Base shape of the variables in the container."""
        return self._shape

    @shape.setter
    def shape(self, shape):
        if self.shape and self.shape != shape:
            raise ValueError(f"Container shape already set to {self.shape}")
        self._shape = shape


class TensorParamTemplate(TensorContainer, ParamTemplate):
    """Template for storing parameter values as tensors.

    It includes functionality to broadcast parameters to a common shape. See
    `diffwofost.base.states_rates.TensorContainer` and
    `pcse.base.states_rates.ParamTemplate` for details.
    """

    def __init__(self, parvalues, shape=None, do_not_broadcast=None):
        self._shape = ()
        self._do_not_broadcast = [] if do_not_broadcast is None else do_not_broadcast
        ParamTemplate.__init__(self, parvalues=parvalues)
        self._broadcast(shape)


class TensorStatesTemplate(TensorContainer, StatesTemplate):
    """Template for storing state variable values as tensors.

    It includes functionality to broadcast state variables to a common shape. See
    `diffwofost.base.states_rates.TensorContainer` and
    `pcse.base.states_rates.StatesTemplate` for details.
    """

    def __init__(self, kiosk=None, publish=None, shape=None, do_not_broadcast=None, **kwargs):
        self._shape = ()
        self._do_not_broadcast = [] if do_not_broadcast is None else do_not_broadcast
        StatesTemplate.__init__(self, kiosk=kiosk, publish=publish, **kwargs)
        self._broadcast(shape)


class TensorRatesTemplate(TensorContainer, RatesTemplate):
    """Template for storing rate variable values as tensors.

    It includes functionality to broadcast rate variables to a common shape. See
    `diffwofost.base.states_rates.TensorContainer` and
    `pcse.base.states_rates.RatesTemplate` for details.
    """

    def __init__(self, kiosk=None, publish=None, shape=None, do_not_broadcast=None):
        self._shape = ()
        self._do_not_broadcast = [] if do_not_broadcast is None else do_not_broadcast
        RatesTemplate.__init__(self, kiosk=kiosk, publish=publish)
        self._broadcast(shape)
