import re
from pathlib import Path
import pytest
import requests
import torch
from diffwofost.physical_models.config import ComputeConfig


def pytest_addoption(parser):
    """Register custom command-line options."""
    parser.addoption(
        "--full_wofost72_test",
        action="store_true",
        default=False,
        help="Run the full Wofost72 gradient test suite (all parameter-output combos). "
        "By default only parameters directly manipulated in wofost72.py are tested.",
    )
    parser.addoption(
        "--fast",
        action="store_true",
        default=False,
        help="Run a reduced test suite: only data files 05, 06, 10, 17, 20, 30, 44 per model, "
        "skip CUDA tests, and skip tensor-mode gradient tests.",
    )


LOCAL_TEST_DIR = Path(__file__).parent / "physical_models" / "test_data"
BASE_PCSE_URL = "https://raw.githubusercontent.com/ajwdewit/pcse/refs/heads/master/tests/test_data"

model_names = [
    "leafdynamics",
    "rootdynamics",
    "potentialproduction",
    "phenology",
    "partitioning",
    "assimilation",
    "transpiration",
    "respiration",
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
def download_test_files(request):
    """Download all required test files before running tests.

    When --fast is active only the files for indices 05, 06, 10, 17, 20, 30, 44
    per model are downloaded.
    """
    fast = request.config.getoption("--fast", default=False)
    fast_indices = [5, 6, 10, 17, 20, 30, 44]
    all_indices = range(1, 45)
    indices = fast_indices if fast else all_indices
    file_names = [
        f"test_{model_name}_wofost72_{i:02d}.yaml" for model_name in model_names for i in indices
    ]
    for file_name in file_names:
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


@pytest.fixture(autouse=True)
def configure_compute_config_dtype():
    """Ensure all tests run with float64 precision."""
    ComputeConfig.set_dtype(torch.float64)


@pytest.fixture
def fast_mode(request):
    """Restrict test scope when ``--fast`` is passed on the CLI.

    When active this fixture skips the current test if any of the following
    applies to its parametrize arguments:

    1. ``test_data_url`` refers to a data-file index not in {05, 06, 10, 17, 20, 30, 44}
       (those indices cover different crops and agro-management configurations).
    2. ``device`` is ``"cuda"`` (GPU tests are skipped even when a GPU is
       available).
    3. ``config_type`` is ``"tensor"`` (tensor-mode gradient tests are
       skipped).
    """
    if not request.config.getoption("--fast", default=False):
        return

    if not hasattr(request.node, "callspec"):
        return

    params = request.node.callspec.params

    # 2. Skip CUDA tests.
    if params.get("device") == "cuda":
        pytest.skip("--fast: skipping CUDA tests")

    # 3. Skip tensor-mode gradient tests.
    if params.get("config_type") == "tensor":
        pytest.skip("--fast: skipping tensor-mode gradient tests")

    # 1. Limit data files to indices 05, 06, 10, 20, 30, 44.
    fast_indices = {5, 6, 10, 17, 20, 30, 44}
    test_data_url = params.get("test_data_url")
    if test_data_url is not None:
        match = re.search(r"_(\d+)\.yaml$", str(test_data_url))
        if match and int(match.group(1)) not in fast_indices:
            pytest.skip(f"--fast: skipping data file index {match.group(1)}")
