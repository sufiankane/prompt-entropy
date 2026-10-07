"""Tests for loop_runtime helpers.

Written before the implementation (TDD RED phase): these must fail until
``loop_runtime.py`` exists and behaves as specified.
"""

from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from loop_runtime import DailyCounter, RotatingJsonlWriter, retry_delay


class RetryDelayTests(unittest.TestCase):
    def test_delay_grows_exponentially_up_to_cap(self) -> None:
        rng = random.Random(0)
        base = 10.0
        max_delay = 300.0
        for failures in range(6):
            expected = min(base * (2**failures), max_delay)
            delay = retry_delay(base, failures, max_delay=max_delay, rng=rng)
            self.assertGreaterEqual(delay, expected * 0.5)
            self.assertLessEqual(delay, expected * 1.5)

    def test_cap_applies_after_many_failures(self) -> None:
        delay = retry_delay(10.0, 20, max_delay=60.0, rng=random.Random(1))
        self.assertGreaterEqual(delay, 30.0)
        self.assertLessEqual(delay, 90.0)

    def test_negative_failures_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            retry_delay(10.0, -1)

    def test_zero_base_interval_stays_zero(self) -> None:
        self.assertEqual(retry_delay(0.0, 3, rng=random.Random(2)), 0.0)


class RotatingJsonlWriterTests(unittest.TestCase):
    def test_appends_json_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "logs" / "requests.jsonl"
            writer = RotatingJsonlWriter(path, max_bytes=1000)
            writer.write({"n": 1})
            writer.write({"n": 2})
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual([json.loads(line) for line in lines], [{"n": 1}, {"n": 2}])

    def test_rotates_when_max_bytes_exceeded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "requests.jsonl"
            writer = RotatingJsonlWriter(path, max_bytes=25)
            writer.write({"n": 1})
            writer.write({"n": 2})
            writer.write({"n": 3})
            current = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            backup_path = Path(str(path) + ".1")
            backup = [
                json.loads(line) for line in backup_path.read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual(current, [{"n": 3}])
        self.assertEqual(backup, [{"n": 1}, {"n": 2}])

    def test_invalid_max_bytes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                RotatingJsonlWriter(Path(tmp) / "x.jsonl", max_bytes=0)


class DailyCounterTests(unittest.TestCase):
    def test_records_up_to_limit_and_persists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            counter = DailyCounter(path, limit=2, today="2026-10-07")
            self.assertTrue(counter.allow())
            counter.record()
            self.assertTrue(counter.allow())
            counter.record()
            self.assertFalse(counter.allow())
            reloaded = DailyCounter(path, limit=2, today="2026-10-07")
        self.assertFalse(reloaded.allow())

    def test_resets_on_a_new_day(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            counter = DailyCounter(path, limit=1, today="2026-10-06")
            counter.record()
            self.assertFalse(counter.allow())
            next_day = DailyCounter(path, limit=1, today="2026-10-07")
        self.assertTrue(next_day.allow())

    def test_corrupt_state_starts_fresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            path.write_text("not json at all", encoding="utf-8")
            counter = DailyCounter(path, limit=1, today="2026-10-07")
        self.assertTrue(counter.allow())

    def test_invalid_limit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                DailyCounter(Path(tmp) / "state.json", limit=0)


if __name__ == "__main__":
    unittest.main()
