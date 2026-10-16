"""CPU-only execution metadata, captured at launch rather than retroactively."""
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from urllib.parse import urlsplit, urlunsplit


def _safe_url(value):
    parsed = urlsplit(value)
    host = parsed.netloc.rsplit("@", 1)[-1]
    path = re.sub(r"/t/[^/]+/", "/t/REDACTED/", parsed.path)
    return urlunsplit((parsed.scheme, host, path, "", ""))


def _command(argv, root):
    try:
        p = subprocess.run(argv, cwd=root, capture_output=True, text=True, timeout=20)
        return dict(argv=argv, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    except (OSError, subprocess.SubprocessError) as error:
        return dict(argv=argv, unavailable=type(error).__name__)


def capture_at_launch(output, root, *, command=None):
    """Preserve the first snapshot on resume; never import torch or allocate CUDA."""
    output, root = Path(output), Path(root)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "runtime_snapshot.json"
    packages = []
    for distribution in importlib.metadata.distributions():
        row = dict(name=distribution.metadata.get("Name"), version=distribution.version)
        direct = distribution.read_text("direct_url.json")
        if direct:
            row["direct_url"] = json.loads(direct)
            if "url" in row["direct_url"]:
                row["direct_url"]["url"] = _safe_url(row["direct_url"]["url"])
        packages.append(row)
    conda = []
    for path in sorted((Path(sys.prefix) / "conda-meta").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        row = {k: data.get(k) for k in ("name", "version", "build", "channel", "sha256")}
        if isinstance(row["channel"], str):
            row["channel"] = _safe_url(row["channel"])
        conda.append(row)
    record = dict(
        capture_phase="actual_process_start", recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        python=sys.version, executable=sys.executable, platform=platform.platform(),
        actual_python_argv=list(sys.argv if command is None else command),
        command_scope="actual Python process; shell launcher is separately versioned, not reconstructed here",
        git_head=_command(["git", "rev-parse", "HEAD"], root),
        git_tracked_status=_command(["git", "status", "--porcelain", "--untracked-files=no"], root),
        device_query=_command(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"], root),
        selected_environment={k: os.environ[k] for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG") if k in os.environ},
        python_packages=sorted(packages, key=lambda x: (x["name"] or "").lower()), conda_packages=conda,
        note="Runtime-only private metadata; not a container image or a bitwise-reproducibility guarantee")
    if target.exists():
        original = json.loads(target.read_text(encoding="utf-8"))
        for key in ("python", "python_packages", "conda_packages", "selected_environment"):
            if original.get(key) != record[key]:
                raise ValueError(f"Runtime environment changed on resume: {key}")
        return original
    # Exclusive creation: never overwrite an earlier process's launch identity.
    try:
        with target.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, ensure_ascii=False)
    except FileExistsError:
        return json.loads(target.read_text(encoding="utf-8"))
    return record
