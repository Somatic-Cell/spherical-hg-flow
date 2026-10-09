"""Launcher integrity plus real CMD routing with stubbed Python workloads.

The CMD tests require Windows. They do not train a model or validate CUDA.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parents[1] / "sweep_architecture.bat"
WINDOWS_ONLY = pytest.mark.skipif(os.name != "nt", reason="Requires real Windows CMD")


def test_launcher_encoding_continuations_and_label_targets():
    """Catch CMD-sensitive source damage without pretending to execute CMD."""
    raw = LAUNCHER.read_bytes()
    script = raw.decode("ascii")
    assert b"\n" not in raw.replace(b"\r\n", b"")
    lines = script.splitlines()
    assert all(line == line.rstrip() for line in lines)
    labels = [line[1:].lower() for line in lines if line.startswith(":")]
    assert len(labels) == len(set(labels))
    targets = re.findall(r"\bgoto\s+:([A-Za-z_][A-Za-z_0-9]*)", script, re.I)
    assert set(target.lower() for target in targets) <= set(labels)
    for index, line in enumerate(lines):
        if line.endswith("^"):
            assert index + 1 < len(lines) and lines[index + 1].strip()


@pytest.fixture
def launcher_workload(tmp_path):
    root = tmp_path / "checkout with spaces"
    (root / "scripts").mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "phaseflow").mkdir()
    (root / "configs" / "rainbow_architecture.json").write_text("{}")
    (root / "phaseflow" / "__init__.py").write_text("")
    record = tmp_path / "rainbow" / "datasets" / "drop_a1_i20_700nm_q1800x3600_c90x1800"
    record.mkdir(parents=True)
    (record / "metadata.json").write_text("{}")
    shutil.copyfile(LAUNCHER, root / LAUNCHER.name)

    # Emulate the existing environment.bat's interpreter choice and pause default.
    python = str(Path(sys.executable)).replace("%", "%%")
    environment = (
        "@echo off\n"
        f'set "PHASEFLOW_PYTHON={python}"\n'
        'set "PHASEFLOW_PYTHON_ARGS=-X utf8"\n'
        'if not defined PHASEFLOW_PAUSE set "PHASEFLOW_PAUSE=1"\n'
        "exit /b 0\n"
    )
    (root / "environment.bat").write_bytes(environment.replace("\n", "\r\n").encode())
    stub = """import json
import os
import sys
from pathlib import Path

stage = 'preflight' if Path(__file__).name == 'check_environment.py' else 'module'
entry = dict(stage=stage, argv=sys.argv[1:], executable=sys.executable,
             cwd=str(Path.cwd()), pause=os.environ.get('PHASEFLOW_PAUSE'),
             unbuffered=os.environ.get('PYTHONUNBUFFERED'), xoptions=sys._xoptions)
with open(os.environ['PHASEFLOW_TEST_CALLS'], 'a', encoding='utf-8') as stream:
    stream.write(json.dumps(entry) + '\\n')
raise SystemExit(int(os.environ.get('PHASEFLOW_TEST_' + stage.upper() + '_EXIT', '0')))
"""
    (root / "scripts" / "check_environment.py").write_text(stub, encoding="utf-8")
    (root / "phaseflow" / "architecture_sweep.py").write_text(stub, encoding="utf-8")
    calls_path = tmp_path / "calls.jsonl"
    env = os.environ.copy()
    env.pop("PHASEFLOW_PAUSE", None)
    env.pop("PYTHONUNBUFFERED", None)
    env["PHASEFLOW_TEST_CALLS"] = str(calls_path)
    env["PYTHONPATH"] = str(root)

    def invoke(arguments=(), **overrides):
        selected_env = env | {key: str(value) for key, value in overrides.items()}
        command = f'call "{root / LAUNCHER.name}"' + "".join(f" {arg}" for arg in arguments)
        # CMD parses its /c command itself; CRT-style backslash escaping from a
        # Python argv list would corrupt the nested quoted path on Windows.
        cmd_executable = os.environ.get("COMSPEC", "cmd.exe")
        command_line = f'"{cmd_executable}" /d /s /c "{command}"'
        result = subprocess.run(
            command_line,
            cwd=tmp_path,
            env=selected_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            timeout=30,
            check=False,
        )
        calls = [json.loads(line) for line in calls_path.read_text().splitlines()] \
            if calls_path.exists() else []
        return root, result, calls

    return invoke


@WINDOWS_ONLY
@pytest.mark.parametrize("mode, flag", [((), None), (("run",), None),
                                        (("check",), "--dry-run"),
                                        (("report",), "--report-only")])
def test_cmd_routes_modes_through_shared_interpreter_without_pause(launcher_workload, mode, flag):
    root, result, calls = launcher_workload(mode)
    assert result.returncode == 0, result.stdout
    assert [entry["stage"] for entry in calls] == ["preflight", "module"]
    assert calls[0]["argv"] == ["check", "--device", "cuda:0"]
    for entry in calls:
        assert Path(entry["executable"]).samefile(sys.executable)
        assert Path(entry["cwd"]) == root
        assert entry["pause"] == "0"
        assert entry["unbuffered"] == "1"
        assert entry["xoptions"]["utf8"] is True
    arguments = calls[1]["argv"]
    if flag is not None:
        assert arguments.pop() == flag
    values = dict(zip(arguments[::2], arguments[1::2], strict=True))
    assert set(values) == {"--record", "--config", "--output", "--device"}
    assert (root / values["--record"] / "metadata.json").is_file()
    assert (root / values["--config"]).is_file()
    assert values["--device"] == "cuda:0"


@WINDOWS_ONLY
@pytest.mark.parametrize("stage, code, call_count", [("PREFLIGHT", 7, 1), ("MODULE", 23, 2)])
def test_cmd_preserves_failure_and_does_not_continue(launcher_workload, stage, code, call_count):
    _, result, calls = launcher_workload(**{f"PHASEFLOW_TEST_{stage}_EXIT": code})
    assert result.returncode == code, result.stdout
    assert len(calls) == call_count
    assert f"exit code {code}" in result.stdout
    assert "[DONE]" not in result.stdout


@WINDOWS_ONLY
@pytest.mark.parametrize("arguments", [("unknown",), ("check", "unexpected")])
def test_cmd_rejects_unrecognized_arguments_before_python(launcher_workload, arguments):
    _, result, calls = launcher_workload(arguments)
    assert result.returncode != 0
    assert calls == []
    assert "Usage:" in result.stdout
