from pathlib import Path
import pytest
import requests
import torch
from diffwofost.physical_models.config import ComputeConfig

LOCAL_TEST_DIR = Path(__file__).parent / "test_data"
BASE_PCSE_URL = "https://raw.githubusercontent.com/ajwdewit/pcse/refs/heads/master/tests/test_data"

model_names = [
    "leafdynamics",
    "rootdynamics",
    "potentialproduction",
    "phenology",
    "partitioning",
    "assimilation",
    "transpiration",
]
FILE_NAMES = [
    f"test_{model_name}_wofost72_{i:02d}.yaml" for model_name in model_names for i in range(1, 45)
]


def download_file(file_name, local_test_dir=LOCAL_TEST_DIR, base_url=BASE_PCSE_URL):
    """Download a single file from GitHub raw URL to local test_data folder."""

    url = f"{base_url}/{file_name}"
    local_test_dir.mkdir(exist_ok=True)
    local_path = local_test_dir / file_name

    if local_path.exists():
        return  # Already downloaded

    print(f"Downloading {file_name} from {url}...")
    response = requests.get(url)
    response.raise_for_status()  # Raise exception on HTTP error
    local_path.write_bytes(response.content)


@pytest.fixture(scope="session", autouse=True)
def download_test_files():
    """Download all required test files before running tests."""
    for file_name in FILE_NAMES:
        download_file(file_name)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    """Parametrize tests over CPU and GPU devices.

    Sets the global ComputeConfig to use the specified device.
    Skips CUDA runs when CUDA isn't available.
    """

    device_name = request.param
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Set the global ComputeConfig to use the specified device
    ComputeConfig.set_device(device_name)

    yield device_name

    # Reset to defaults after the test
    ComputeConfig.reset_to_defaults()
