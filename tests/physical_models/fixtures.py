import pytest
import torch


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    """Parametrize tests over CPU and GPU devices.

    Skips CUDA runs when CUDA isn't available.
    """

    device_name = request.param
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return device_name
