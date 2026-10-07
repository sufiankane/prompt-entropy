"""Integration tests for gemini_uncached_loop.py CLI.

Dry-run and validation paths must work on any interpreter, with or
without gemini-webapi installed; only live paths need the package.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from prompt_entropy import FILLER_MESSAGES

SCRIPT = Path(__file__).parent / "gemini_uncached_loop.py"
HAS_GEMINI = importlib.util.find_spec("gemini_webapi") is not None


def run_loop(*args: str, clean_env: bool = False) -> subprocess.CompletedProcess[str]:
    env = None
    if clean_env:
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in ("GEMINI_SECURE_1PSID", "GEMINI_SECURE_1PSIDTS")
        }
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        env=env,
    )


class CustomPromptTests(unittest.TestCase):
    def test_dry_run_uses_custom_prompt_with_entropy(self) -> None:
        prompt = "What is the airspeed velocity of an unladen swallow?"
        proc = run_loop(
            "--dry-run",
            "--count",
            "2",
            "--interval",
            "0",
            "--prompt",
            prompt,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.count(prompt), 2)
        self.assertRegex(proc.stdout, r"[0-9a-fA-F]{8}")

    def test_dry_run_reads_prompts_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.txt"
            path.write_text(
                "First line prompt\n\n# comment line\nSecond line prompt\n",
                encoding="utf-8",
            )
            proc = run_loop(
                "--dry-run",
                "--count",
                "30",
                "--interval",
                "0",
                "--prompt-file",
                str(path),
            )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("First line prompt", proc.stdout)
        self.assertIn("Second line prompt", proc.stdout)
        self.assertNotIn("# comment line", proc.stdout)

    def test_fillers_flag_is_honoured(self) -> None:
        proc = run_loop(
            "--dry-run",
            "--count",
            "1",
            "--interval",
            "0",
            "--fillers",
            "0",
            "--prompt",
            "No filler please",
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        for note in FILLER_MESSAGES:
            self.assertNotIn(note, proc.stdout)

    def test_level_flag_light_disables_fillers_and_ref(self) -> None:
        proc = run_loop(
            "--dry-run",
            "--count",
            "1",
            "--interval",
            "0",
            "--level",
            "light",
            "--prompt",
            "Quiet prompt",
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotIn("ref:", proc.stdout)
        for note in FILLER_MESSAGES:
            self.assertNotIn(note, proc.stdout)


class GuardTests(unittest.TestCase):
    def test_check_mode_requires_cookies(self) -> None:
        proc = run_loop("--check", clean_env=True)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("Missing the __Secure-1PSID cookie", proc.stderr)

    def test_max_per_day_must_be_positive(self) -> None:
        proc = run_loop("--dry-run", "--max-per-day", "0", "--count", "1", "--interval", "0")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("must be >= 1", proc.stderr)

    def test_dry_run_writes_no_log_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "requests.jsonl"
            proc = run_loop(
                "--dry-run",
                "--count",
                "1",
                "--interval",
                "0",
                "--log-file",
                str(log),
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertFalse(log.exists())
            self.assertFalse(Path(str(log) + ".count.json").exists())


@unittest.skipIf(HAS_GEMINI, "requires an interpreter without gemini-webapi")
class MissingDependencyTests(unittest.TestCase):
    def test_live_mode_without_package_explains_how_to_install(self) -> None:
        proc = run_loop("--check", "--psid", "fake-psid", clean_env=True)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("pip install -U gemini_webapi", proc.stderr)


if __name__ == "__main__":
    unittest.main()
