"""Optional torch.compile for the numerical kernels of the model components.

The components keep their bookkeeping (parameters, kiosk, traitlets) in Python and
pass tensors to pure functions decorated with :func:`kernel`. When compilation is
switched on (``ComputeConfig.set_compile(True)``) these functions are compiled
with ``torch.compile``, which fuses their element-wise operations into a few
device kernels; otherwise they run eagerly.

Compiled GPU kernels are generated without contracting multiplications and
additions into fused multiply-adds (Triton's ``default_fp_fusion`` is switched
off when the first kernel is compiled). Fused multiply-adds round differently,
and the model has discrete switches on computed values (e.g. roots stop growing
when the interpolated partitioning fraction FR is not > 0), so they would make
compiled and eager runs diverge. Without them, both give the same results.
"""

import functools
import torch
from diffwofost.physical_models.config import ComputeConfig


def kernel(fn=None, *, dynamic=False):
    """Decorate a pure tensor function so it is compiled when compilation is on.

    Args:
        fn: The function.
        dynamic: Passed to torch.compile. The default (False) compiles for fixed
            tensor shapes; None lets shapes that change between calls become dynamic.
    """
    if fn is None:
        return functools.partial(kernel, dynamic=dynamic)
    compiled = None

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        nonlocal compiled
        if not ComputeConfig.get_compile():
            return fn(*args, **kwargs)
        if compiled is None:
            _disable_fp_fusion()
            compiled = torch.compile(fn, fullgraph=True, dynamic=dynamic)
        return compiled(*args, **kwargs)

    wrapper.eager = fn
    return wrapper


def _disable_fp_fusion():
    try:
        import triton
    except ImportError:  # no Triton (e.g. CPU-only builds): nothing to configure
        return
    triton.knobs.language.default_fp_fusion = False
