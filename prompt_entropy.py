"""Inject random messages and entropy into prompts to minimise cache hits.

Every call to :func:`inject_entropy` returns the original prompt text
verbatim, wrapped in fresh randomness:

* a random hex salt in randomly chosen brackets,
* zero or more random filler messages (unrelated notes) placed before or
  after the prompt,
* an optional ``ref:<hex>`` suffix marker,
* optional leading/trailing whitespace jitter,
* optional sentence shuffling for prompts whose order does not matter.

Entropy presets select how much noise is added:

* ``light``    - salt only,
* ``standard`` - salt, one filler message, ref marker, whitespace jitter,
* ``paranoid`` - 24-hex salt, three filler messages, ref marker, jitter.

:func:`inject_messages` is an adapter for OpenAI/Anthropic-style chat
payloads: it inserts one random filler message into a list of
``{"role", "content"}`` dictionaries.

Library use::

    from prompt_entropy import inject_entropy, inject_messages

    unique_prompt = inject_entropy("Summarise this article.", level="paranoid")
    payload = inject_messages([{"role": "user", "content": "Hello"}])

CLI use::

    python prompt_entropy.py "Summarise this article."
    echo "Summarise this article." | python prompt_entropy.py --count 5
    python prompt_entropy.py --prompt-file prompt.txt --level paranoid --json

All emitted text is ASCII so it can be piped between tools on Windows.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

FILLER_MESSAGES: tuple[str, ...] = (
    "the average depth of the North Sea is about 95 metres",
    "Iceland generates almost all of its electricity from renewable sources",
    "the first vending machines dispensed postage stamps",
    "sea otters hold hands while sleeping so they do not drift apart",
    "bamboo can grow nearly a metre in a single day",
    "the word quarantine comes from the Venetian dialect for forty days",
    "Saturn is less dense than water",
    "the oldest known beer recipe is about four thousand years old",
    "a group of flamingos is called a flamboyance",
    "the Eiffel Tower can stand about 15 centimetres taller in summer",
    "honey does not spoil when it is stored properly",
    "the shortest recorded war lasted about 38 minutes",
    "a single strand of spider silk is thinner than a human hair but stronger than steel of the same weight",
    "Venus rotates so slowly that a day there lasts longer than its year",
    "the first computer programmer was Ada Lovelace in the 1840s",
    "wombat droppings are cube-shaped",
    "the Great Barrier Reef is visible from space",
    "a teaspoon of neutron star material would weigh about a billion tonnes",
    "the Sahara was green about six thousand years ago",
    "lobsters can regenerate lost limbs",
    "the Olympic Games once included tug of war",
    "sound travels about four times faster in water than in air",
    "a day on Mars is about 24 hours and 40 minutes",
    "the Dutch East India Company was the first publicly traded company",
    "cats cannot taste sweetness",
    "the Moon drifts about 3.8 centimetres farther from Earth every year",
    "butterflies taste with their feet",
    "a bolt of lightning is about five times hotter than the surface of the sun",
    "a group of crows is called a murder",
    "the first alarm clock could only ring at 4 a.m.",
)

ENTROPY_LEVELS: dict[str, dict[str, int | bool]] = {
    "light": {"salt_length": 6, "fillers": 0, "suffix": False, "whitespace": False},
    "standard": {"salt_length": 8, "fillers": 1, "suffix": True, "whitespace": True},
    "paranoid": {"salt_length": 24, "fillers": 3, "suffix": True, "whitespace": True},
}

_FILLER_TEMPLATES: tuple[str, ...] = (
    "Unrelated note (please ignore): {note}.",
    "[Off-topic context, safe to ignore: {note}]",
    "(Aside, not related to the request: {note})",
    "P.S. {note} - unrelated to everything else here.",
    "Background noise: {note}.",
)

_BRACKETS: tuple[tuple[str, str], ...] = (("(", ")"), ("[", "]"), ("<", ">"), ("{", "}"))
_SEPARATORS: tuple[str, ...] = ("\n", "\n\n", " ", "\n \n")
_LEADING: tuple[str, ...] = ("", "\n", "\n\n", "  ")
_TRAILING: tuple[str, ...] = ("", "\n", "\n\n", "  ")
_HEX_DIGITS = "0123456789abcdef"
_REF_LENGTH = 8
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _random_hex(rng: random.Random, length: int) -> str:
    return "".join(rng.choice(_HEX_DIGITS) for _ in range(length))


def _salt_marker(rng: random.Random, length: int) -> str:
    left, right = rng.choice(_BRACKETS)
    return f"{left}{_random_hex(rng, length)}{right}"


def _filler_message(rng: random.Random) -> str:
    return rng.choice(_FILLER_TEMPLATES).format(note=rng.choice(FILLER_MESSAGES))


def _shuffle_sentences(text: str, rng: random.Random) -> str:
    parts = _SENTENCE_SPLIT.split(text.strip())
    if len(parts) < 2:
        return text
    rng.shuffle(parts)
    return " ".join(parts)


def inject_entropy(
    prompt: str,
    *,
    seed: int | None = None,
    level: str = "standard",
    salt_length: int | None = None,
    fillers: int | None = None,
    suffix: bool | None = None,
    whitespace: bool | None = None,
    shuffle_sentences: bool = False,
) -> str:
    """Return *prompt* wrapped in fresh random entropy.

    ``level`` chooses an ENTROPY_LEVELS preset; any of ``salt_length``,
    ``fillers``, ``suffix`` and ``whitespace`` may be passed to override
    the preset. ``seed`` makes the output reproducible (handy for tests);
    leave it as ``None`` for real use. ``shuffle_sentences`` randomly
    reorders the prompt's own sentences, which changes word order - use
    it only for prompts whose sentence order does not matter.
    """
    preset = ENTROPY_LEVELS.get(level)
    if preset is None:
        raise ValueError(
            f"unknown level {level!r}; choose from {', '.join(sorted(ENTROPY_LEVELS))}"
        )
    if salt_length is None:
        salt_length = int(preset["salt_length"])
    if fillers is None:
        fillers = int(preset["fillers"])
    if suffix is None:
        suffix = bool(preset["suffix"])
    if whitespace is None:
        whitespace = bool(preset["whitespace"])

    if salt_length < 0:
        raise ValueError("salt_length must be >= 0")
    if fillers < 0:
        raise ValueError("fillers must be >= 0")

    rng = random.Random(seed)

    body = _shuffle_sentences(prompt, rng) if shuffle_sentences else prompt

    before: list[str] = []
    after: list[str] = []
    if salt_length > 0:
        before.append(_salt_marker(rng, salt_length))
    for _ in range(fillers):
        (before if rng.random() < 0.5 else after).append(_filler_message(rng))
    if suffix:
        after.append(f"ref:{_random_hex(rng, _REF_LENGTH)}")

    text = rng.choice(_SEPARATORS).join(before + [body] + after)
    if whitespace:
        text = rng.choice(_LEADING) + text + rng.choice(_TRAILING)
    return text


def inject_messages(
    messages: list[dict[str, object]],
    *,
    seed: int | None = None,
    role: str = "system",
    position: str = "start",
    note: str | None = None,
) -> list[dict[str, object]]:
    """Return a copy of *messages* with one random filler message inserted.

    Adapter for OpenAI/Anthropic-style chat payloads (lists of
    ``{"role", "content"}`` dictionaries). The original list is never
    mutated. ``role`` sets the inserted message's role, ``position`` is
    ``"start"`` or ``"end"``, and ``note`` replaces the random filler
    text with your own message.
    """
    if position not in ("start", "end"):
        raise ValueError("position must be 'start' or 'end'")

    rng = random.Random(seed)
    base = note if note is not None else rng.choice(FILLER_MESSAGES)
    content = (
        f"{rng.choice(_FILLER_TEMPLATES).format(note=base)}"
        f" ref:{_random_hex(rng, _REF_LENGTH)}"
    )
    filler = {"role": role, "content": content}

    result = [dict(message) for message in messages]
    if position == "start":
        result.insert(0, filler)
    else:
        result.append(filler)
    return result


def _read_prompt(args: argparse.Namespace) -> str | None:
    if args.text:
        return " ".join(args.text)
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8").strip()
    if not sys.stdin.isatty():
        data = sys.stdin.read().strip()
        if data:
            return data
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prompt_entropy",
        description=(
            "Inject random messages/entropy into a prompt so repeated "
            "requests differ and prompt caches cannot hit."
        ),
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="Prompt text. If omitted, --prompt-file or stdin is used.",
    )
    parser.add_argument("--prompt-file", help="Read the prompt from this file instead.")
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of variations to emit (default: 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducible output (default: random).",
    )
    parser.add_argument(
        "--level",
        choices=sorted(ENTROPY_LEVELS),
        default="standard",
        help="Entropy preset: light (salt only), standard, or paranoid (default: standard).",
    )
    parser.add_argument(
        "--salt-length",
        type=int,
        default=None,
        help="Hex length of the random salt marker; overrides the level preset; 0 disables it.",
    )
    parser.add_argument(
        "--fillers",
        type=int,
        default=None,
        help="Number of random filler messages; overrides the level preset; 0 disables them.",
    )
    parser.add_argument(
        "--no-suffix",
        action="store_true",
        help="Do not append the ref:<hex> marker.",
    )
    parser.add_argument(
        "--no-whitespace",
        action="store_true",
        help="Do not add leading/trailing whitespace jitter.",
    )
    parser.add_argument(
        "--shuffle-sentences",
        action="store_true",
        help="Randomly reorder the prompt's sentences (changes word order; "
        "use only when sentence order does not matter).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON object with all variations.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.count < 1:
        parser.error("--count must be >= 1")
    if args.salt_length is not None and args.salt_length < 0:
        parser.error("--salt-length must be >= 0")
    if args.fillers is not None and args.fillers < 0:
        parser.error("--fillers must be >= 0")

    try:
        prompt = _read_prompt(args)
    except OSError as exc:
        parser.error(f"could not read prompt file: {exc}")
    if not prompt:
        parser.error("no prompt provided: pass text, use --prompt-file, or pipe it on stdin")

    base_rng = random.Random(args.seed)
    variations = [
        inject_entropy(
            prompt,
            seed=None if args.seed is None else base_rng.getrandbits(32),
            level=args.level,
            salt_length=args.salt_length,
            fillers=args.fillers,
            suffix=False if args.no_suffix else None,
            whitespace=False if args.no_whitespace else None,
            shuffle_sentences=args.shuffle_sentences,
        )
        for _ in range(args.count)
    ]

    if args.json:
        print(
            json.dumps(
                {
                    "seed": args.seed,
                    "level": args.level,
                    "count": args.count,
                    "prompts": variations,
                },
                indent=2,
            )
        )
    else:
        for index, variation in enumerate(variations):
            if index:
                print("---")
            print(variation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
