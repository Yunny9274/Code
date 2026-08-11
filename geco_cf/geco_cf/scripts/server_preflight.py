"""Collect server prerequisites without requiring PyTorch or PyG."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.metadata
import io
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _module_status(module_name: str) -> dict[str, Any]:
    captured = io.StringIO()
    with contextlib.redirect_stderr(captured), contextlib.redirect_stdout(captured):
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # report import failures instead of hiding them
            result: dict[str, Any] = {"installed": False, "error": f"{type(exc).__name__}: {exc}"}
        else:
            result = {"installed": True, "version": getattr(module, "__version__", None)}
    diagnostic = captured.getvalue().strip()
    if diagnostic:
        result["import_diagnostic"] = diagnostic[-4000:]
    return result


def _nvidia_smi() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "error": "nvidia-smi not found"}
    try:
        completed = subprocess.run(
            [executable, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"available": True, "gpus": [line.strip() for line in completed.stdout.splitlines() if line.strip()]}


def collect() -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "working_directory": str(Path.cwd()),
        "packages": {
            package: {"distribution_version": _version(package), **_module_status(module)}
            for package, module in (
                ("geco-cf", "geco_cf"),
                ("networkx", "networkx"),
                ("numpy", "numpy"),
                ("torch", "torch"),
                ("torch-geometric", "torch_geometric"),
            )
        },
        "nvidia_smi": _nvidia_smi(),
    }
    try:
        torch = importlib.import_module("torch")
    except Exception:
        report["torch_runtime"] = {"available": False}
    else:
        cuda = getattr(torch, "cuda", None)
        report["torch_runtime"] = {
            "available": True,
            "torch_version": getattr(torch, "__version__", None),
            "cuda_build": getattr(torch.version, "cuda", None),
            "cuda_available": bool(cuda and cuda.is_available()),
            "device_count": int(cuda.device_count()) if cuda else 0,
            "device_names": [cuda.get_device_name(i) for i in range(cuda.device_count())]
            if cuda and cuda.is_available()
            else [],
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Check GECo-CF server prerequisites")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()
    report = collect()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"Python: {report['python'].split()[0]}")
        print(f"Working directory: {report['working_directory']}")
        print(f"nvidia-smi: {report['nvidia_smi']}")
        print(f"Torch runtime: {report['torch_runtime']}")
        for name, status in report["packages"].items():
            print(f"{name}: {status}")


if __name__ == "__main__":
    main()
