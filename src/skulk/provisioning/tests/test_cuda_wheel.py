"""Validate CUDA wheel dependency and loader contracts without a GPU."""

import os
import subprocess
import sys
import tomllib
from pathlib import Path

from pydantic import JsonValue, TypeAdapter


def test_cuda_wheel_declares_nccl_and_exposes_its_runtime(tmp_path: Path) -> None:
    """A clean environment resolves NCCL from the wheel dependencies, not the host."""
    root = Path(__file__).resolve().parents[4]
    package = root / "packaging" / "skulk-llama-server-cuda"
    metadata = TypeAdapter(dict[str, JsonValue]).validate_python(
        tomllib.loads((package / "pyproject.toml").read_text())
    )
    project = metadata["project"]
    assert isinstance(project, dict)
    dependencies = project["dependencies"]
    assert isinstance(dependencies, list)
    assert any(
        isinstance(item, str) and item.startswith("nvidia-nccl-cu12>=")
        for item in dependencies
    )
    namespace = tmp_path / "nvidia"
    namespace.mkdir()
    (namespace / "__init__.py").write_text("")
    for name in ("cuda_runtime", "cublas", "nccl"):
        module = namespace / name
        (module / "lib").mkdir(parents=True)
        (module / "__init__.py").write_text("")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(tmp_path), str(package / "src")))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json; from skulk_llama_server_cuda import "
            "launch_environment; print(json.dumps(launch_environment({'LD_LIBRARY_PATH':"
            "'/existing/runtime'})['LD_LIBRARY_PATH'].split(':')))",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    libraries = TypeAdapter(list[str]).validate_json(result.stdout)
    assert str(namespace / "nccl" / "lib") in libraries
    assert libraries[-1] == "/existing/runtime"
