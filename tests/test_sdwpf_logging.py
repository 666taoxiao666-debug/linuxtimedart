import json
import os
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from utils.sdwpf_logging import (
    _atomic_write_text,
    allocate_log_directory,
    finish_log_directory,
    sanitize_component,
    tensorboard_log_directory,
)


class SDWPFLoggingTests(unittest.TestCase):
    def test_daily_sequence_and_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "logs with spaces"
            now = datetime(2026, 9, 4, 8, 30, tzinfo=timezone.utc)
            first = allocate_log_directory(
                "finetune",
                "h12 f0 seed2024 lr=1e-5",
                "run-a",
                root,
                date="20260904",
                now=now,
            )
            second = allocate_log_directory(
                "baseline",
                "h12/f0",
                "run-b",
                root,
                date="20260904",
                now=now,
            )

            self.assertTrue(first.name.startswith("001_finetune_"))
            self.assertTrue(second.name.startswith("002_baseline_"))
            self.assertNotIn(" ", first.name)
            self.assertNotIn("/", second.name)
            self.assertLessEqual(len(first.name.encode("utf-8")), 76)
            rows = (root / "20260904" / "index.tsv").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(rows), 3)
            self.assertEqual(
                (root / "latest.txt").read_text(encoding="utf-8").strip(),
                f"20260904/{second.name}",
            )

    def test_relative_log_root_returns_an_absolute_run_path(self):
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            try:
                os.chdir(temporary)
                run = allocate_log_directory(
                    "finetune", "h12", root="relative-logs", date="20260904"
                )
                self.assertTrue(run.is_absolute())
                self.assertEqual(run.parent, Path(temporary) / "relative-logs/20260904")
            finally:
                os.chdir(original_cwd)

    def test_atomic_directory_allocation_is_unique_under_concurrency(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "logs"

            def allocate(index):
                return allocate_log_directory(
                    "finetune",
                    f"h12_f{index}",
                    f"run-{index}",
                    root,
                    date="20260904",
                )

            with ThreadPoolExecutor(max_workers=8) as executor:
                paths = list(executor.map(allocate, range(20)))

            self.assertEqual(len({path.name for path in paths}), 20)
            sequences = sorted(int(path.name.split("_", 1)[0]) for path in paths)
            self.assertEqual(sequences, list(range(1, 21)))
            rows = (root / "20260904" / "index.tsv").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(rows), 21)
            latest = (root / "latest.txt").read_text(encoding="utf-8").strip()
            self.assertTrue(Path(latest).name.startswith("020_"))

    def test_independent_allocators_do_not_race_on_latest(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "multi process logs"
            processes = [
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "utils.sdwpf_logging",
                        "allocate",
                        "--task",
                        "finetune",
                        "--parameters",
                        f"h12_f{index}",
                        "--run-id",
                        f"run-{index}",
                        "--root",
                        str(root),
                        "--date",
                        "20260904",
                    ],
                    cwd=repository,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                for index in range(24)
            ]
            failures = []
            paths = []
            for process in processes:
                stdout, stderr = process.communicate(timeout=30)
                if process.returncode:
                    failures.append((process.returncode, stderr))
                elif stdout.strip():
                    paths.append(Path(stdout.strip()))

            self.assertEqual(failures, [])
            self.assertEqual(len(paths), 24)
            self.assertEqual(len({path.name for path in paths}), 24)
            sequences = sorted(int(path.name.split("_", 1)[0]) for path in paths)
            self.assertEqual(sequences, list(range(1, 25)))
            rows = (root / "20260904" / "index.tsv").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(rows), 25)
            self.assertEqual(len({row.split("\t", 1)[0] for row in rows[1:]}), 24)
            latest = (root / "latest.txt").read_text(encoding="utf-8").strip()
            self.assertTrue(Path(latest).name.startswith("024_"))

    def test_long_and_unsafe_component_is_portable_and_bounded(self):
        component = sanitize_component("../ bad/value " + "x" * 400)
        self.assertNotIn("/", component)
        self.assertNotIn(" ", component)
        self.assertLessEqual(len(component.encode("utf-8")), 180)
        self.assertRegex(component, r"_h[0-9a-f]{12}$")
        for budget in (1, 2, 3, 8, 14):
            with self.subTest(budget=budget):
                shortened = sanitize_component("unexpected-long-task", budget)
                self.assertLessEqual(len(shortened.encode("utf-8")), budget)
        with self.assertRaises(ValueError):
            sanitize_component("value", 0)

    def test_finish_updates_status_and_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "logs"
            run = allocate_log_directory(
                "pipeline", "h12", "run-a", root, date="20260904"
            )
            metadata_before = (run / "log_meta.json").read_text(encoding="utf-8")
            finish_log_directory(run, 3)
            status = (run / "status.env").read_text(encoding="utf-8")
            metadata = json.loads((run / "log_meta.json").read_text(encoding="utf-8"))
            self.assertIn("STATUS=FAILED", status)
            self.assertIn("EXIT_CODE=3", status)
            self.assertEqual(metadata["schema_version"], 2)
            self.assertEqual(metadata["status_file"], "status.env")
            self.assertNotIn("status", metadata)
            self.assertEqual(
                (run / "log_meta.json").read_text(encoding="utf-8"),
                metadata_before,
            )

    def test_invalid_date_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            for invalid_date in ("2026-09-04", "20261399", "20260230"):
                with self.subTest(date=invalid_date), self.assertRaises(ValueError):
                    allocate_log_directory(
                        "finetune", root=temporary, date=invalid_date
                    )

    def test_corrupt_metadata_cannot_create_a_false_completed_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "logs"
            run = allocate_log_directory(
                "pipeline", "h12", "run-a", root, date="20260904"
            )
            (run / "log_meta.json").write_text("{broken", encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                finish_log_directory(run, 0)
            status = (run / "status.env").read_text(encoding="utf-8")
            self.assertIn("STATUS=RUNNING", status)
            self.assertNotIn("STATUS=COMPLETED", status)

    def test_finish_rejects_an_unallocated_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "typo"
            with self.assertRaises(FileNotFoundError):
                finish_log_directory(missing, 0)
            self.assertFalse(missing.exists())

    def test_atomic_write_cleans_temporary_file_after_fsync_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "latest.txt"
            with patch("utils.sdwpf_logging.os.fsync", side_effect=OSError("disk")):
                with self.assertRaises(OSError):
                    _atomic_write_text(destination, "value\n")
            self.assertEqual(list(Path(temporary).glob(".latest.txt.*.tmp")), [])
            self.assertFalse(destination.exists())

    def test_legacy_numbered_directory_is_not_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "logs"
            day = root / "20260904"
            (day / "001_legacy_run").mkdir(parents=True)
            allocated = allocate_log_directory(
                "finetune", "h12", root=root, date="20260904"
            )
            self.assertTrue(allocated.name.startswith("002_finetune_"))

    def test_invalid_legacy_date_cannot_poison_root_latest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "logs"
            (root / "20261399" / "999_legacy-invalid-date").mkdir(parents=True)
            allocated = allocate_log_directory(
                "finetune", "h12", root=root, date="20260905"
            )
            latest = (root / "latest.txt").read_text(encoding="utf-8").strip()
            self.assertEqual(latest, f"20260905/{allocated.name}")

    def test_all_log_owning_entrypoints_use_the_shared_helper(self):
        repository = Path(__file__).resolve().parents[1]
        entrypoints = [
            "scripts/pretrain/SDWPF.sh",
            "scripts/finetune/SDWPF.sh",
            "scripts/finetune/SDWPF_ablation_prompt.sh",
            "scripts/finetune/SDWPF_folds.sh",
            "scripts/finetune/SDWPF_ablation_suite.sh",
            "scripts/eval/SDWPF_baselines.sh",
            "scripts/eval/SDWPF_logged_final_eval.sh",
            "scripts/eval/SDWPF_logged_validation_plot.sh",
            "scripts/train/SDWPF_full_pipeline.sh",
            "scripts/train/SDWPF_paper_cv.sh",
            "scripts/train/SDWPF_paper_final.sh",
        ]
        for relative_path in entrypoints:
            source = (repository / relative_path).read_text(encoding="utf-8")
            with self.subTest(script=relative_path):
                self.assertIn("lib/sdwpf_log.sh", source)
                self.assertIn("sdwpf_log_init", source)

    def test_sdwpf_tensorboard_events_follow_managed_run_directory(self):
        with patch.dict("os.environ", {"SDWPF_LOG_DIR": "logs/run-001"}):
            result = tensorboard_log_directory(
                "PromptTimeDART", "SDWPF", "finetune", "run/with spaces"
            )
        result_path = Path(result)
        self.assertEqual(result_path.parent, Path("logs/run-001/tb"))
        self.assertRegex(result_path.name, r"^ft_[0-9a-f]{10}$")
        self.assertLessEqual(len(result_path.name), 13)

        with patch.dict("os.environ", {"SDWPF_LOG_DIR": "logs/run-001"}):
            fallback = tensorboard_log_directory(
                "PromptTimeDART", "SDWPF", "unexpected-long-task", "run-a"
            )
        self.assertLessEqual(len(Path(fallback).name.encode("utf-8")), 19)

    def test_other_datasets_keep_legacy_tensorboard_directory(self):
        with patch.dict("os.environ", {"SDWPF_LOG_DIR": "logs/run-001"}):
            result = tensorboard_log_directory(
                "TimeDART", "ETTh1", "finetune", "run-a"
            )
        self.assertEqual(Path(result), Path("outputs/logs/TimeDART/ETTh1"))

    @unittest.skipIf(os.name == "nt", "Bash syntax is checked on the Linux server")
    def test_sdwpf_shell_entrypoints_have_valid_bash_syntax(self):
        repository = Path(__file__).resolve().parents[1]
        scripts = [
            repository / "scripts/lib/sdwpf_log.sh",
            *sorted((repository / "scripts").glob("*/SDWPF*.sh")),
        ]
        subprocess.run(
            ["bash", "-n", *(str(path) for path in scripts)],
            cwd=repository,
            check=True,
        )

    @unittest.skipIf(os.name == "nt", "Bash behavior is checked on the Linux server")
    def test_shell_sidecars_ignore_dots_in_parent_directories(self):
        repository = Path(__file__).resolve().parents[1]
        helper = repository / "scripts/lib/sdwpf_log.sh"
        command = (
            f'source "{helper}"; '
            'SDWPF_LOG_FILE="/tmp/a.b/runfile"; sdwpf_log_sidecar env; '
            'SDWPF_LOG_FILE="./logs/runfile"; sdwpf_log_sidecar summary.txt; '
            'SDWPF_LOG_FILE="/tmp/run.env"; sdwpf_log_sidecar env; '
            'SDWPF_LOG_FILE="/tmp/run.log"; sdwpf_log_sidecar env'
        )
        result = subprocess.run(
            ["bash", "-c", command],
            cwd=repository,
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertEqual(
            result.stdout.splitlines(),
            [
                "/tmp/a.b/runfile.env",
                "./logs/runfile.summary.txt",
                "/tmp/run.env.env",
                "/tmp/run.env",
            ],
        )

    @unittest.skipIf(os.name == "nt", "Bash behavior is checked on the Linux server")
    def test_shell_capture_propagates_tee_failure(self):
        repository = Path(__file__).resolve().parents[1]
        helper = repository / "scripts/lib/sdwpf_log.sh"
        command = (
            f'source "{helper}"; '
            'SDWPF_LOG_FILE=/dev/full; SDWPF_LOG_DIR=/dev; '
            'sdwpf_log_init test h12 run.log run-a; '
            'sdwpf_log_install_exit_trap; sdwpf_log_capture; echo payload'
        )
        result = subprocess.run(
            ["bash", "-c", command],
            cwd=repository,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("tee failed", result.stderr)

    @unittest.skipIf(os.name == "nt", "Bash behavior is checked on the Linux server")
    def test_shell_finalize_failure_changes_a_successful_exit_code(self):
        repository = Path(__file__).resolve().parents[1]
        helper = repository / "scripts/lib/sdwpf_log.sh"
        command = (
            f'source "{helper}"; '
            'SDWPF_LOG_DIR="/tmp/codex-missing-${BASHPID}"; '
            '_SDWPF_LOG_OWNS_DIR=1; sdwpf_log_install_exit_trap; true'
        )
        result = subprocess.run(
            ["bash", "-c", command],
            cwd=repository,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("could not finalize status", result.stderr)


if __name__ == "__main__":
    unittest.main()
