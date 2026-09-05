"""Deterministic, searchable log directories for SDWPF shell entrypoints.

The directory allocator deliberately uses an atomic ``mkdir`` loop instead of
an incrementing counter file.  That keeps daily sequence numbers unique when
multiple folds are launched concurrently and leaves an explicit directory for
an interrupted allocation rather than silently reusing its number.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import re
import tempfile
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


DEFAULT_LOG_ROOT = Path("outputs/logs/SDWPF")
INDEX_HEADER = "sequence\tstarted_at\ttask\trun_id\tparameters\tpath\n"
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9._+-]+")
_LATEST_WRITE_LOCK = threading.Lock()
_INDEX_WRITE_LOCK = threading.Lock()


def sanitize_component(value: object, max_bytes: int = 180) -> str:
    """Return a portable, bounded directory-name component."""

    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    text = _SAFE_COMPONENT.sub("_", str(value).strip()).strip("._-")
    text = re.sub(r"_+", "_", text) or "run"
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text

    digest = hashlib.sha256(encoded).hexdigest()
    if max_bytes <= 2:
        return digest[:max_bytes]
    suffix = "_h" + digest[: min(12, max_bytes - 2)]
    budget = max_bytes - len(suffix.encode("ascii"))
    if budget == 0:
        return suffix
    prefix = text
    while len(prefix.encode("utf-8")) > budget:
        prefix = prefix[:-1]
    prefix = prefix.rstrip("._-")
    return (prefix + suffix) if prefix else suffix


def _validate_day(day: str) -> None:
    if not re.fullmatch(r"\d{8}", day):
        raise ValueError("SDWPF log date must use YYYYMMDD")
    try:
        parsed_day = datetime.strptime(day, "%Y%m%d")
    except ValueError as exc:
        raise ValueError("SDWPF log date must be a real calendar date") from exc
    if parsed_day.strftime("%Y%m%d") != day:
        raise ValueError("SDWPF log date must use canonical YYYYMMDD")


def tensorboard_log_directory(
    model: object,
    data: object,
    task: object,
    run_id: object = "",
    default_root: Union[str, Path] = "./outputs/logs",
) -> str:
    """Keep SDWPF TensorBoard events beside the owning structured log."""

    managed_root = os.environ.get("SDWPF_LOG_DIR")
    if str(data).upper() == "SDWPF" and managed_root:
        # The owning run directory already contains the readable parameters.
        # Keep this child component deliberately short: TensorBoard appends a
        # long event filename, and the old task+run_id form crossed MAX_PATH on
        # ordinary Windows installations.
        task_text = str(task).lower()
        task_alias = {"pretrain": "pt", "finetune": "ft", "test": "ev"}.get(
            task_text,
            sanitize_component(task, max_bytes=8),
        )
        identity = f"{model}|{data}|{task}|{run_id}".encode("utf-8")
        stage = f"{task_alias}_{hashlib.sha256(identity).hexdigest()[:10]}"
        return str(Path(managed_root) / "tb" / stage)
    return str(Path(default_root) / str(model) / str(data))


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(20):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.01 * (attempt + 1))
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _atomic_write_json(path: Path, payload: dict) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


@contextmanager
def _file_lock(lock_path: Path, thread_lock: threading.Lock):
    """Hold an OS-managed lock that is automatically released on process exit."""

    with thread_lock:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+b")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()

        def try_lock() -> bool:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError as exc:
                    if exc.errno in {errno.EACCES, errno.EDEADLK, errno.EAGAIN}:
                        return False
                    raise
            else:
                import fcntl

                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return False
            return True

        deadline = time.monotonic() + 30.0
        try:
            while True:
                if try_lock():
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for {lock_path}")
                time.sleep(0.01)
            try:
                yield
            finally:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _append_index_row(index_path: Path, row: str) -> None:
    """Append one row under a short cross-process directory lock."""

    lock_path = index_path.parent / ".index.lockfile"
    with _file_lock(lock_path, _INDEX_WRITE_LOCK):
        write_header = not index_path.exists() or index_path.stat().st_size == 0
        with index_path.open("a", encoding="utf-8", newline="") as handle:
            if write_header:
                handle.write(INDEX_HEADER)
            handle.write(row)
            handle.flush()
            os.fsync(handle.fileno())


def _field(value: object) -> str:
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")


def allocate_log_directory(
    task: str,
    parameters: str = "",
    run_id: str = "",
    root: Union[str, Path] = DEFAULT_LOG_ROOT,
    date: Optional[str] = None,
    now: Optional[datetime] = None,
) -> Path:
    """Atomically allocate ``YYYYMMDD/NNN_task_parameters`` and index it."""

    timestamp = now or datetime.now().astimezone()
    day = date or os.environ.get("SDWPF_LOG_DATE") or timestamp.strftime("%Y%m%d")
    _validate_day(day)

    # The shell helper invokes the allocator from the repository root, but the
    # returned path can later be consumed after a cwd change. Always return an
    # absolute directory so log writes and final status target the same run.
    root_path = Path(root).expanduser().resolve()
    day_path = root_path / day
    day_path.mkdir(parents=True, exist_ok=True)
    reservation_path = day_path / ".sequence"
    reservation_path.mkdir(exist_ok=True)

    safe_task = sanitize_component(task, max_bytes=24)
    safe_parameters = sanitize_component(parameters, max_bytes=100) if parameters else ""
    stem = safe_task if not safe_parameters else f"{safe_task}_{safe_parameters}"
    # The full, unabridged parameter string remains in index.tsv/log_meta.json.
    # A shorter visible component leaves room for TensorBoard's hostname-heavy
    # event filename on Windows systems where MAX_PATH is still active.
    stem = sanitize_component(stem, max_bytes=72)

    run_path = None
    sequence = None
    for candidate_sequence in range(1, 100000):
        reservation = reservation_path / f"{candidate_sequence:05d}"
        try:
            reservation.mkdir()
        except FileExistsError:
            continue

        # Coexist with directories produced before the reservation mechanism
        # was introduced. Once found, reserve their number permanently.
        existing = list(day_path.glob(f"{candidate_sequence:03d}_*"))
        if existing:
            continue

        candidate = day_path / f"{candidate_sequence:03d}_{stem}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        run_path = candidate
        sequence = candidate_sequence
        break
    if run_path is None or sequence is None:
        raise RuntimeError(f"No free SDWPF log sequence under {day_path}")

    started_at = timestamp.isoformat(timespec="seconds")
    try:
        relative_path = run_path.relative_to(root_path).as_posix()
    except ValueError:
        relative_path = run_path.as_posix()

    metadata = {
        "schema_version": 2,
        "sequence": sequence,
        "date": day,
        "started_at": started_at,
        "task": str(task),
        "parameters": str(parameters),
        "safe_parameters": safe_parameters,
        "run_id": str(run_id),
        "path": run_path.as_posix(),
        "status_file": "status.env",
    }
    _atomic_write_json(run_path / "log_meta.json", metadata)
    _atomic_write_text(
        run_path / "status.env",
        f"STATUS=RUNNING\nSTARTED_AT={started_at}\n",
    )

    row = "\t".join(
        [
            f"{sequence:03d}",
            _field(started_at),
            _field(task),
            _field(run_id),
            _field(parameters),
            _field(relative_path),
        ]
    ) + "\n"
    _append_index_row(day_path / "index.tsv", row)
    # Both pointers share a root-level process lock. Without it, independent
    # Python allocators can collide in os.replace on Windows. Rescan while the
    # lock is held so a slower, lower sequence cannot move latest backward.
    with _file_lock(root_path / ".latest.lockfile", _LATEST_WRITE_LOCK):
        numbered_directories = []
        for child in day_path.iterdir():
            if not child.is_dir() or child.name.startswith("."):
                continue
            prefix = child.name.split("_", 1)[0]
            if prefix.isdigit():
                numbered_directories.append((int(prefix), child))
        latest_path = max(numbered_directories, key=lambda item: item[0])[1]
        _atomic_write_text(day_path / "latest.txt", latest_path.name + "\n")

        root_candidates = []
        for candidate_day in root_path.iterdir():
            if not candidate_day.is_dir():
                continue
            try:
                _validate_day(candidate_day.name)
            except ValueError:
                continue
            for child in candidate_day.iterdir():
                if not child.is_dir() or child.name.startswith("."):
                    continue
                prefix = child.name.split("_", 1)[0]
                if prefix.isdigit():
                    root_candidates.append((candidate_day.name, int(prefix), child))
        root_latest = max(root_candidates, key=lambda item: (item[0], item[1]))[2]
        try:
            latest_relative = root_latest.relative_to(root_path).as_posix()
        except ValueError:
            latest_relative = root_latest.as_posix()
        _atomic_write_text(root_path / "latest.txt", latest_relative + "\n")
    return run_path


def finish_log_directory(
    log_directory: Union[str, Path],
    exit_code: int,
    now: Optional[datetime] = None,
) -> None:
    """Atomically record whether the owning shell entrypoint completed."""

    run_path = Path(log_directory)
    metadata_path = run_path / "log_meta.json"
    if not run_path.is_dir() or not metadata_path.is_file():
        raise FileNotFoundError(f"Not an allocated SDWPF log directory: {run_path}")
    # Metadata is intentionally immutable. ``status.env`` is the sole status
    # authority, avoiding a non-atomic two-file status transaction.
    json.loads(metadata_path.read_text(encoding="utf-8"))
    timestamp = (now or datetime.now().astimezone()).isoformat(timespec="seconds")
    status = "COMPLETED" if int(exit_code) == 0 else "FAILED"
    _atomic_write_text(
        run_path / "status.env",
        f"STATUS={status}\nEXIT_CODE={int(exit_code)}\nCOMPLETED_AT={timestamp}\n",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    allocate = subparsers.add_parser("allocate")
    allocate.add_argument("--task", required=True)
    allocate.add_argument("--parameters", default="")
    allocate.add_argument("--run-id", default="")
    allocate.add_argument(
        "--root",
        default=os.environ.get("SDWPF_LOG_ROOT", str(DEFAULT_LOG_ROOT)),
    )
    allocate.add_argument("--date", default=None)

    finish = subparsers.add_parser("finish")
    finish.add_argument("--log-dir", required=True)
    finish.add_argument("--exit-code", required=True, type=int)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "allocate":
        path = allocate_log_directory(
            task=args.task,
            parameters=args.parameters,
            run_id=args.run_id,
            root=args.root,
            date=args.date,
        )
        print(path.as_posix())
    else:
        finish_log_directory(args.log_dir, args.exit_code)


if __name__ == "__main__":
    main()
