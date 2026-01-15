from pcse.base import ParamTemplate
from ..traitlets import Tensor


class TensorParamTemplate(ParamTemplate):
    def __init__(self, parvalues, shape=None):
        self._shape = None
        super().__init__(parvalues)
        self._broadcast_to_common_shape(shape)

    @property
    def shape(self):
        """Base shape of the model parameters."""
        return self._shape

    @shape.setter
    def shape(self, shape):
        # Don't need to update shape if we are adding a scalar
        if not shape:
            return
        # Set new shape if not yet initialized or all parameters are scalar
        if self._shape is None or not self._shape:
            self._shape = shape
        elif self._shape == shape:
            pass
        else:
            raise ValueError(f"Trying to set shape {shape} incompatible with shape {self.shape}")

    @property
    def _tensor_params(self):
        return [key for key, trait in self.traits().items() if isinstance(trait, Tensor)]

    def _broadcast_to_common_shape(self, shape=None):
        """Broadcast all parameters to a common shape.

        Args:
            shape (tuple | torch.Size): Target shape of the parameters.
        """
        shape = self.shape if shape is None else shape
        if shape is None:
            return
        for parname in self._tensor_params:
            tensor = getattr(self, parname)
            try:
                tensor_broadcasted = tensor.expand(shape)
            except RuntimeError as error:
                raise ValueError(f"Cannot broadcast tensor {parname} to shape {shape}") from error
            setattr(self, parname, tensor_broadcasted)

    def __setattr__(self, attr, value):
        super().__setattr__(attr, value)
        if value in self._tensor_params:
            self.shape = value.shape
