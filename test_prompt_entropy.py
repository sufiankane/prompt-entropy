"""Tests for the prompt_entropy program.

Written before the implementation (TDD RED phase): these must fail until
``prompt_entropy.py`` exists and behaves as specified.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

from prompt_entropy import (
    ENTROPY_LEVELS,
    FILLER_MESSAGES,
    inject_entropy,
    inject_messages,
)

SCRIPT = Path(__file__).parent / "prompt_entropy.py"


def run_cli(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        input=stdin,
        timeout=60,
    )


def count_filler_mentions(text: str) -> int:
    return sum(text.count(note) for note in FILLER_MESSAGES)


class InjectEntropyTests(unittest.TestCase):
    def test_prompt_text_is_preserved_verbatim(self) -> None:
        prompt = "Explain how a mechanical watch escapement works."
        result = inject_entropy(prompt, seed=1)
        self.assertIn(prompt, result)

    def test_repeated_calls_are_unique(self) -> None:
        results = {inject_entropy("How does binary search achieve O(log n)?") for _ in range(50)}
        self.assertEqual(len(results), 50)

    def test_same_seed_is_reproducible(self) -> None:
        prompt = "Same input, same seed."
        self.assertEqual(inject_entropy(prompt, seed=7), inject_entropy(prompt, seed=7))

    def test_different_seeds_differ(self) -> None:
        prompt = "Same input, different seeds."
        self.assertNotEqual(inject_entropy(prompt, seed=7), inject_entropy(prompt, seed=8))

    def test_salt_appears_when_enabled(self) -> None:
        result = inject_entropy("Hello", seed=3, salt_length=12, fillers=0, suffix=False)
        self.assertRegex(result, r"[0-9a-fA-F]{12}")

    def test_all_entropy_layers_can_be_disabled(self) -> None:
        result = inject_entropy(
            "Hello",
            seed=3,
            salt_length=0,
            fillers=0,
            suffix=False,
            whitespace=False,
        )
        self.assertEqual(result, "Hello")

    def test_filler_notes_are_absent_when_disabled(self) -> None:
        result = inject_entropy("Hello there", seed=5, fillers=0)
        for note in FILLER_MESSAGES:
            self.assertNotIn(note, result)

    def test_filler_note_is_included_when_enabled(self) -> None:
        result = inject_entropy("Hello there", seed=5, salt_length=0)
        self.assertTrue(any(note in result for note in FILLER_MESSAGES))

    def test_suffix_reference_appears_when_enabled(self) -> None:
        result = inject_entropy("Hello", seed=4, fillers=0)
        self.assertIn("ref:", result)

    def test_negative_parameters_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            inject_entropy("Hello", salt_length=-1)
        with self.assertRaises(ValueError):
            inject_entropy("Hello", fillers=-1)


class EntropyLevelTests(unittest.TestCase):
    def test_level_presets_scale_entropy(self) -> None:
        prompt = "Scale the entropy."
        light = inject_entropy(prompt, seed=1, level="light")
        standard = inject_entropy(prompt, seed=1, level="standard")
        paranoid = inject_entropy(prompt, seed=1, level="paranoid")

        self.assertEqual(count_filler_mentions(light), 0)
        self.assertNotIn("ref:", light)
        self.assertEqual(count_filler_mentions(standard), 1)
        self.assertIn("ref:", standard)
        self.assertEqual(count_filler_mentions(paranoid), 3)
        self.assertRegex(paranoid, r"[0-9a-fA-F]{24}")

    def test_explicit_overrides_beat_level(self) -> None:
        result = inject_entropy("Override me", seed=1, level="paranoid", fillers=0)
        self.assertEqual(count_filler_mentions(result), 0)

    def test_unknown_level_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            inject_entropy("Hello", level="turbo")

    def test_levels_are_advertised(self) -> None:
        self.assertEqual(set(ENTROPY_LEVELS), {"light", "standard", "paranoid"})


class SentenceShuffleTests(unittest.TestCase):
    PROMPT = "First sentence. Second sentence. Third sentence. Fourth sentence."

    @staticmethod
    def strip_marker(text: str) -> str:
        return re.sub(r"^[\[\(\<\{][0-9a-fA-F]+[\]\)\>\}]\s*", "", text).strip()

    @staticmethod
    def normalize(text: str) -> list[str]:
        # Splitting on ". " leaves the prompt's final period attached to
        # whichever sentence lands last, so ignore trailing periods.
        return sorted(part.rstrip(".") for part in text.split(". "))

    def test_sentences_are_preserved_in_any_order(self) -> None:
        outputs = {
            self.strip_marker(
                inject_entropy(self.PROMPT, seed=seed, level="light", shuffle_sentences=True)
            )
            for seed in range(20)
        }
        expected = self.normalize(self.PROMPT)
        for text in outputs:
            self.assertEqual(self.normalize(text), expected)
        self.assertGreater(len(outputs), 1)

    def test_single_sentence_is_unchanged(self) -> None:
        prompt = "Just one sentence."
        result = inject_entropy(prompt, seed=1, level="light", shuffle_sentences=True)
        self.assertIn(prompt, result)


class MessageInjectionTests(unittest.TestCase):
    def test_does_not_mutate_input(self) -> None:
        messages = [{"role": "user", "content": "hello"}]
        result = inject_messages(messages, seed=1)
        self.assertEqual(messages, [{"role": "user", "content": "hello"}])
        self.assertIsNot(result, messages)

    def test_inserts_filler_message_at_start(self) -> None:
        result = inject_messages([{"role": "user", "content": "hello"}], seed=1)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["role"], "system")
        self.assertTrue(any(note in result[0]["content"] for note in FILLER_MESSAGES))
        self.assertIn("ref:", result[0]["content"])

    def test_can_append_at_end(self) -> None:
        result = inject_messages(
            [{"role": "user", "content": "hello"}], seed=1, role="user", position="end"
        )
        self.assertEqual(result[-1]["role"], "user")

    def test_custom_note_is_used(self) -> None:
        result = inject_messages([], seed=1, note="just noise")
        self.assertIn("just noise", result[0]["content"])

    def test_results_are_unique_across_calls(self) -> None:
        contents = {
            inject_messages([{"role": "user", "content": "hi"}])[0]["content"]
            for _ in range(20)
        }
        self.assertEqual(len(contents), 20)

    def test_invalid_position_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            inject_messages([], position="middle")


class CliTests(unittest.TestCase):
    def test_prints_entropy_injected_prompt(self) -> None:
        proc = run_cli("Hello from the CLI", "--seed", "11")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Hello from the CLI", proc.stdout)

    def test_count_emits_that_many_variations(self) -> None:
        proc = run_cli("Count me", "--count", "3", "--seed", "2")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.count("Count me"), 3)
        self.assertEqual(proc.stdout.count("---"), 2)

    def test_json_mode_returns_all_variations(self) -> None:
        proc = run_cli("Json prompt", "--count", "2", "--json")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(len(payload["prompts"]), 2)
        for variation in payload["prompts"]:
            self.assertIn("Json prompt", variation)

    def test_reads_prompt_from_stdin(self) -> None:
        proc = run_cli("--seed", "9", stdin="Piped prompt text")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Piped prompt text", proc.stdout)

    def test_missing_prompt_is_an_error(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(SCRIPT)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            stdin=subprocess.DEVNULL,
            timeout=60,
        )
        self.assertEqual(proc.returncode, 2)

    def test_level_flag_light_omits_fillers_and_suffix(self) -> None:
        proc = run_cli("Level flag", "--seed", "3", "--level", "light")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotIn("ref:", proc.stdout)
        for note in FILLER_MESSAGES:
            self.assertNotIn(note, proc.stdout)

    def test_unknown_level_is_rejected_by_cli(self) -> None:
        proc = run_cli("Bogus", "--level", "turbo")
        self.assertEqual(proc.returncode, 2)


if __name__ == "__main__":
    unittest.main()
