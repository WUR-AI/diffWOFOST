import torch
from traitlets_pcse import TraitType
from traitlets_pcse import Undefined
from .config import ComputeConfig


class Tensor(TraitType):
    def __init__(
        self,
        default_value=Undefined,
        allow_none=False,
        read_only=None,
        help=None,
        config=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(
            default_value=default_value,
            allow_none=allow_none,
            read_only=read_only,
            help=help,
            config=config,
            **kwargs,
        )
        self.dtype = dtype

    def validate(self, obj, value):
        """Validate input object, recasting it into a tensor if possible."""
        device = ComputeConfig.get_device()
        dtype = ComputeConfig.get_dtype() if self.dtype is None else self.dtype
        if isinstance(value, torch.Tensor):
            casted = value.to(dtype=dtype, device=device)
            return casted
        try:
            # Try casting value into a tensor, raise validation error if it fails
            return torch.tensor(value, dtype=dtype, device=device)
        except:  # noqa: E722
            self.error(obj, value)

    def from_string(self, s):
        """Casting tensor from string is not supported for now."""
        raise NotImplementedError
