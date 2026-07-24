from types import SimpleNamespace
from unittest.mock import patch

import run


def test_xtts_runtime_check_imports_model_in_clean_interpreter():
    completed = SimpleNamespace(returncode=0, stdout="", stderr="")

    with patch.object(run.subprocess, "run", return_value=completed) as subprocess_run:
        assert run._check_xtts_runtime()

    command = subprocess_run.call_args.args[0]
    assert command[0] == run.sys.executable
    assert command[1] == "-c"
    assert "import_xtts_runtime" in command[2]


def test_xtts_runtime_check_rejects_broken_transitive_import():
    completed = SimpleNamespace(
        returncode=1,
        stdout="",
        stderr="ImportError: Numba needs NumPy 2.4 or less",
    )

    with patch.object(run.subprocess, "run", return_value=completed):
        assert not run._check_xtts_runtime()


def test_ensure_coqui_repairs_dependencies_before_rechecking_runtime():
    with patch.object(run, "_check_xtts_runtime", side_effect=[False, True]), patch.object(
        run,
        "_pip_install",
        return_value=True,
    ) as pip_install:
        assert run.ensure_coqui_tts()

    pip_install.assert_called_once_with(
        "coqui-tts",
        "transformers>=4.0,<5",
        "numpy>=2.0,<2.5",
    )
