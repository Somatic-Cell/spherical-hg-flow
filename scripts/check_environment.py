"""Check the shared Windows launcher environment; only setup installs packages."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import struct
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TORCH_VERSION = "2.14.1"
TORCH_BUILD = f"{TORCH_VERSION}+cu130"
CUDA_VERSION = "13.0"
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"


class LauncherError(RuntimeError):
    """An actionable launcher error rather than a CDF or training error."""


def check_python() -> None:
    print(f"Python executable: {sys.executable}", flush=True)
    print(f"Python version: {sys.version}", flush=True)
    if sys.version_info[:2] != (3, 14) or struct.calcsize("P") != 8:
        raise LauncherError(
            "These launchers require Python 3.14, 64-bit. Select that interpreter "
            "in environment.bat; setup and training use the same selection."
        )


def check_phaseflow() -> None:
    try:
        module = importlib.import_module("phaseflow")
    except ImportError as exc:
        raise LauncherError(
            f"phaseflow cannot be imported by {sys.executable}. "
            "Run execute.bat setup with the same environment.bat selection. "
            f"Import error: {exc}"
        ) from exc
    actual = getattr(module, "__file__", None)
    print(f"phaseflow module: {actual}", flush=True)
    expected = PROJECT_ROOT / "src" / "phaseflow" / "__init__.py"
    if actual is None or Path(actual).resolve() != expected.resolve():
        raise LauncherError(
            "The selected Python imports phaseflow from a different location. "
            f"Expected {expected}. Run execute.bat setup in this checkout."
        )
    try:
        importlib.import_module("phaseflow.cli")
    except Exception as exc:
        raise LauncherError(
            "phaseflow is present, but its command-line dependencies cannot be loaded. "
            "Run execute.bat setup with the same environment.bat selection. "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def check_torch(device: str) -> str:
    if importlib.util.find_spec("torch") is None:
        raise LauncherError(
            f"PyTorch is not installed in {sys.executable}. Run execute.bat setup. "
            "Packages installed for another Python version are not shared."
        )
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        raise LauncherError(
            "PyTorch was found but could not be imported; it has not been replaced. "
            "Check the Python environment selected in environment.bat. "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    installed = importlib.metadata.version("torch")
    print(f"PyTorch version: {torch.__version__} (package: {installed})", flush=True)
    print(f"PyTorch CUDA runtime: {torch.version.cuda}", flush=True)
    if str(torch.__version__).split("+", 1)[0] != TORCH_VERSION:
        raise LauncherError(
            f"Expected existing PyTorch {TORCH_VERSION}; found {torch.__version__}. "
            "Select the intended interpreter in environment.bat. "
            "The existing PyTorch installation has not been replaced."
        )
    if torch.version.cuda != CUDA_VERSION:
        raise LauncherError(
            f"Expected a PyTorch build for CUDA {CUDA_VERSION}; found {torch.version.cuda}. "
            "The system CUDA Toolkit version does not determine the PyTorch build. "
            "The existing installation has not been replaced."
        )
    try:
        selected = torch.device(device)
        if selected.type != "cuda":
            raise ValueError("These launchers require a CUDA device; no CPU fallback is used.")
        print(f"GPU: {selected}: {torch.cuda.get_device_name(selected)}", flush=True)
        x = torch.ones(1, device=selected, requires_grad=True)
        x.square().sum().backward()
        if x.grad.item() != 2.0:
            raise RuntimeError("The GPU gradient check did not return 2.0.")
        print("GPU forward/backward check: passed", flush=True)
    except Exception as exc:
        raise LauncherError(f"CUDA check failed for {device}: {exc}") from exc
    return installed


def check_tensorboard() -> None:
    try:
        module = importlib.import_module("tensorboard.main")
    except Exception as exc:
        raise LauncherError(
            "TensorBoard cannot be loaded in the selected Python. "
            "Run execute.bat setup to install the monitor extra. "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    print(f"TensorBoard module: {module.__file__}", flush=True)


def setup(device: str) -> int:
    # No torch import is attempted before the missing-package installation.
    if importlib.util.find_spec("torch") is None:
        print(f"Installing {TORCH_BUILD} into {sys.executable}", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                f"torch=={TORCH_BUILD}",
                "--index-url",
                TORCH_INDEX,
            ],
            check=True,
        )
    installed = check_torch(device)
    print(f"Installing this checkout; keeping torch=={installed}", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", ".[dev,monitor]", f"torch=={installed}"],
        cwd=PROJECT_ROOT,
        check=True,
    )
    if importlib.metadata.version("torch") != installed:
        raise LauncherError("The installed torch package version changed during setup.")
    # Check imports in a fresh process, after pip has finished installing dependencies.
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "check", "--device", device],
        check=True,
    )
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "monitor"],
        check=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("setup", "check", "monitor"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    try:
        check_python()
        if args.mode == "setup":
            return setup(args.device)
        if args.mode == "monitor":
            # TensorBoard reads event files; CUDA and torch are not required here.
            check_tensorboard()
        else:
            check_phaseflow()
            check_torch(args.device)
        return 0
    except LauncherError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        return 1
    except subprocess.CalledProcessError as exc:
        print(
            f"[ERROR] Setup command failed with exit code {exc.returncode}. "
            "See the output above; no alternate Python/PyTorch version was selected.",
            file=sys.stderr,
            flush=True,
        )
        return exc.returncode
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
