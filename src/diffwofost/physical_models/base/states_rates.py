from pcse.base import ParamTemplate
from pcse.base import RatesTemplate
from pcse.base import StatesTemplate
from pcse.traitlets import HasTraits
from ..traitlets import Tensor


class TensorContainer(HasTraits):
    def __init__(self, shape=None, do_not_broadcast=None, **variables):
        """Container of tensor variables.

        It includes functionality to broadcast variables to a common shape. This common shape can
        be inferred from the container's variables, or it can be set as an input argument.

        Args:
            shape (tuple | torch.Size, optional): Shape to which the variables in the container
                should be broadcasted to. Only variables listed as `do_not_broadcast` will be
                skipped. Defaults to None.
            do_not_broadcast (list, optional): Name of the variables that will not be broadcasted
                to the container shape. Defaults to None, which means that all variables will be
                broadcasted.
            variables (dict): Collection of variables to initialize the container, as key-value
                pairs.
        """
        self._shape = () if shape is None else tuple(shape)
        self._do_not_broadcast = [] if do_not_broadcast is None else do_not_broadcast
        HasTraits.__init__(self, **variables)
        self._broadcast_to_common_shape()

    @property
    def shape(self):
        """Base shape of the variables in the container."""
        return self._shape

    @shape.setter
    def shape(self, shape):
        if self.shape and self.shape != shape:
            raise ValueError(f"Container shape already set to {self.shape}")
        self._shape = shape

    @property
    def _tensor_variables(self):
        return [key for key, trait in self.traits().items() if isinstance(trait, Tensor)]

    def _broadcast_to_common_shape(self):
        # Identify variables to broadcast and extract common variables' shape.
        vars_to_broadcast = {
            varname: getattr(self, varname)
            for varname in self._tensor_variables
            if varname not in self._do_not_broadcast
        }
        vars_shape = _get_common_vars_shape(vars_to_broadcast)

        # Check whether the shape extracted from the variables is consistent with the one set to the
        # container.
        if not vars_shape and not self.shape:
            return
        if not vars_shape or not self.shape:
            shape = vars_shape or self.shape
        elif vars_shape == self.shape:
            shape = vars_shape
        else:
            raise ValueError(
                f"Container shape {self.shape} does not match variable shape {vars_shape}"
            )

        # Broadcast all required variables to the common shape.
        for varname, var in vars_to_broadcast.items():
            try:
                broadcasted = var.expand(shape)
            except RuntimeError as error:
                raise ValueError(f"Cannot broadcast tensor {varname} to shape {shape}") from error
            setattr(self, varname, broadcasted)

        # Update the container shape
        self.shape = shape


def _get_common_vars_shape(vars):
    shape = ()
    for var in vars.values():
        if not var.shape or shape == var.shape:
            continue
        elif var.shape and not shape:
            shape = tuple(var.shape)
        else:
            raise ValueError(f"Incompatible shapes within variables: {shape} and {var.shape}")
    return shape


class TensorParamTemplate(TensorContainer, ParamTemplate):
    def __init__(self, parvalues, shape=None, do_not_broadcast=None):
        self._shape = () if shape is None else tuple(shape)
        self._do_not_broadcast = [] if do_not_broadcast is None else do_not_broadcast
        ParamTemplate.__init__(self, parvalues=parvalues)
        self._broadcast_to_common_shape()


class TensorStatesTemplate(TensorContainer, StatesTemplate):
    def __init__(self, kiosk=None, publish=None, shape=None, do_not_broadcast=None, **kwargs):
        self._shape = () if shape is None else tuple(shape)
        self._do_not_broadcast = [] if do_not_broadcast is None else do_not_broadcast
        StatesTemplate.__init__(self, kiosk=kiosk, publish=publish, **kwargs)
        self._broadcast_to_common_shape()


class TensorRatesTemplate(TensorContainer, RatesTemplate):
    def __init__(self, kiosk=None, publish=None, shape=None, do_not_broadcast=None):
        self._shape = () if shape is None else tuple(shape)
        self._do_not_broadcast = [] if do_not_broadcast is None else do_not_broadcast
        RatesTemplate.__init__(self, kiosk=kiosk, publish=publish)
        self._broadcast_to_common_shape()
