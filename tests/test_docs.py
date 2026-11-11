import json
from pathlib import Path
import pytest

NOTEBOOKS = sorted((Path(__file__).parents[1] / "docs" / "notebooks").glob("*.ipynb"))


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.stem)
def test_diffwofost_imports_in_notebooks_resolve(notebook):
    """Docs are not executed in CI, so at least their diffwofost imports must resolve."""
    code = "\n".join(
        "".join(cell["source"])
        for cell in json.loads(notebook.read_text(encoding="utf-8"))["cells"]
        if cell["cell_type"] == "code"
    )
    for line in code.splitlines():
        if line.startswith(("import diffwofost", "from diffwofost")):
            exec(line, {})  # noqa: S102 - an ImportError here is the failure
