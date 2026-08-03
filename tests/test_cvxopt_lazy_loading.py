import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "app"


def run_import_probe(source):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(APP_DIR)
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_sequential_helper_import_does_not_load_cvxopt():
    output = run_import_probe(
        "import sys; import zSim_Helper; "
        "print(any(k == 'cvxopt' or k.startswith('cvxopt.') for k in sys.modules))"
    )
    assert output == "False"


def test_legacy_matrix_call_loads_cvxopt_lazily():
    output = run_import_probe(
        "import sys; import zSim_Helper as z; "
        "z.matrix([1.0], tc='d'); "
        "print(any(k == 'cvxopt' or k.startswith('cvxopt.') for k in sys.modules))"
    )
    assert output == "True"
