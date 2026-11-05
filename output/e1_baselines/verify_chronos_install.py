"""Installation/weight check on synthetic data only; never accesses wind data."""
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "180")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "180")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    status = {"python": sys.executable, "real_data_access": False,
              "packages": {k: md.version(k) for k in (
                  "chronos-forecasting", "torch", "transformers", "accelerate", "numpy", "pandas")}}
    try:
        import numpy as np
        import pandas as pd
        import torch
        from chronos import Chronos2Pipeline
        from huggingface_hub import HfApi, snapshot_download
        torch.set_num_threads(2)
        status["import_ok"] = True
        save(out / "chronos_install.json", status)
        pin = out / "chronos_revision.json"
        if pin.exists():
            revision = json.loads(pin.read_text())["revision"]
        else:
            revision = HfApi().model_info("amazon/chronos-2").sha
            save(pin, {"model": "amazon/chronos-2", "revision": revision})
        status["revision"] = revision
        folder = Path(snapshot_download("amazon/chronos-2", revision=revision,
                      allow_patterns=["*.json", "*.safetensors", "LICENSE*", "README.md"], max_workers=2))
        status["model_files"] = {}
        for file in sorted(folder.rglob("*")):
            if file.is_file():
                h = hashlib.sha256()
                with file.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024*1024), b""):
                        h.update(chunk)
                status["model_files"][str(file.relative_to(folder))] = {"bytes": file.stat().st_size, "sha256": h.hexdigest()}
        pipeline = Chronos2Pipeline.from_pretrained(str(folder), device_map="cpu", local_files_only=True)
        context = pd.DataFrame({"id": ["synthetic"]*48,
                     "timestamp": pd.date_range("2000-01-01", periods=48, freq="h"),
                     "target": np.sin(np.arange(48)/6), "past_covariate": np.cos(np.arange(48)/6)})
        prediction = pipeline.predict_df(context, prediction_length=4, quantile_levels=[.1, .5, .9],
                                         id_column="id", timestamp_column="timestamp", target="target")
        numeric = prediction.select_dtypes(include="number").to_numpy()
        assert len(prediction) == 4 and np.isfinite(numeric).all()
        status.update(status="SYNTHETIC_PASSED", rows=len(prediction), columns=list(prediction.columns),
                      parameters=sum(p.numel() for p in pipeline.model.parameters()))
    except Exception as exc:
        status.update(status="INCOMPLETE", error_type=type(exc).__name__)
        save(out / "chronos_install.json", status)
        print(json.dumps(status), flush=True)
        return 1
    save(out / "chronos_install.json", status)
    print(json.dumps(status), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
