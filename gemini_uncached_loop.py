#!/usr/bin/env python3
"""Keep asking Google Gemini isolated, mutually unrelated questions.

The goal is to avoid any reuse of context or exact-match prompt caching:

* Every call goes through ``GeminiClient.generate_content``, which is a
  stateless single-turn request. No conversation metadata (c_id / r_id /
  rc_id) is sent, so each network turn starts with a zero-length window.
* Every prompt is wrapped in fresh entropy by ``prompt_entropy`` (random
  salt, optional unrelated filler messages, ref marker, whitespace
  jitter), so two runs never send the same prompt text.
* By default the base prompts are drawn at random from a pool of
  unrelated subjects; pass ``--prompt``/``--prompt-file`` to send your
  own prompts with entropy injected around them.
* Pass ``--temporary`` to also keep the requests out of Gemini history.

The client talks to gemini.google.com over plain HTTP with your account
cookies (via the ``gemini_webapi`` package); no browser or Chromium
process is involved. The package is imported lazily, so --dry-run works
on an interpreter without it.

Setup:
    pip install -U gemini_webapi

    Log in at https://gemini.google.com, open DevTools -> Application
    (Chrome/Brave) or Storage (Firefox) -> Cookies -> gemini.google.com,
    and copy the values of ``__Secure-1PSID`` and ``__Secure-1PSIDTS``.

    PowerShell:
        $env:GEMINI_SECURE_1PSID = "..."
        $env:GEMINI_SECURE_1PSIDTS = "..."
        python gemini_uncached_loop.py --interval 6

    Or pass the cookies directly:
        python gemini_uncached_loop.py --psid "..." --psidts "..."

    Validate the cookies with a single live request:
        python gemini_uncached_loop.py --check

    Send your own prompts with entropy injected around them:
        python gemini_uncached_loop.py --prompt "Summarise this article."
        python gemini_uncached_loop.py --prompt-file prompts.txt --fillers 2

    Test the loop without cookies or network access:
        python gemini_uncached_loop.py --dry-run --count 3

Unattended runs:
    * ``--log-file requests.jsonl`` records every request (rotating to
      ``requests.jsonl.1`` past ``--log-max-bytes``).
    * ``--max-per-day N`` enforces a persistent daily cap that survives
      restarts (state next to the log file, or at ``--state-file``).
    * Failures use exponential backoff with jitter. ``run_scheduled.ps1``
      wraps this for Windows Task Scheduler.

Notes:
    * ``--level`` picks the entropy preset (light, standard, paranoid);
      ``--salt-length`` and ``--fillers`` override it, and 0 disables
      either layer if you want less noise around the prompt.
    * Cookies are read locally only; this script stores nothing. Keep
      the values private and never commit them.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from loop_runtime import DailyCounter, RotatingJsonlWriter, retry_delay
from prompt_entropy import ENTROPY_LEVELS, inject_entropy

if TYPE_CHECKING:
    from gemini_webapi import GeminiClient

UNRELATED_PROMPTS = [
    "What causes the color changes in autumn foliage at a cellular level?",
    "Explain the scoring rules in Olympic curling.",
    "How does a mechanical escapement mechanism work in a wrist watch?",
    "What are the historical origins of sourdough bread fermentation?",
    "Describe the process of deep-sea hydrothermal vent formation.",
    "How do aircraft fly-by-wire flight control systems handle sensor disagreement?",
    "Summarize the dietary habits of the platypus in freshwater habitats.",
    "How does the binary search algorithm achieve O(log n) time complexity?",
]


def preview(text: str, limit: int = 150) -> str:
    """Collapse whitespace and truncate response text for console output."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


async def run_loop(client: GeminiClient | None, args: argparse.Namespace) -> tuple[int, int]:
    ok = 0
    failed = 0
    attempt = 0
    consecutive_failures = 0

    while args.count == 0 or attempt < args.count:
        if args.daily_counter is not None and not args.daily_counter.allow():
            print(
                f"Daily cap of {args.daily_counter.limit} request(s) reached; stopping.",
                flush=True,
            )
            break

        attempt += 1
        topic = random.choice(args.prompt_pool)
        prompt = inject_entropy(
            topic,
            level=args.level,
            salt_length=args.salt_length,
            fillers=args.fillers,
        )
        print(f"[{attempt}] Sending: {prompt}", flush=True)

        next_delay = args.interval
        aborted = False
        record: dict[str, object] = {
            "time": datetime.now(timezone.utc).isoformat(),
            "attempt": attempt,
            "prompt": prompt,
        }

        if args.dry_run:
            ok += 1
            record["ok"] = True
            record["dry_run"] = True
            print("(dry-run: request not sent)")
        else:
            try:
                response = await client.generate_content(prompt, temporary=args.temporary)
                ok += 1
                consecutive_failures = 0
                record["ok"] = True
                record["response"] = preview(response.text or "")
                print("Response preview:")
                print(preview(response.text or ""))
            except Exception as exc:  # keep the loop alive on transient errors
                failed += 1
                consecutive_failures += 1
                next_delay = retry_delay(args.interval, consecutive_failures)
                record["ok"] = False
                record["error"] = f"{type(exc).__name__}: {exc}"
                print(
                    f"Request failed ({consecutive_failures} in a row): {exc}",
                    file=sys.stderr,
                )
                if consecutive_failures >= args.max_failures:
                    print(
                        f"Aborting after {consecutive_failures} consecutive failures. "
                        "Check your cookies and network connection.",
                        file=sys.stderr,
                    )
                    aborted = True

        if not args.dry_run:
            if args.log_writer is not None:
                args.log_writer.write(record)
            if args.daily_counter is not None:
                args.daily_counter.record()

        if aborted:
            break

        print("=" * 60, flush=True)
        if args.count == 0 or attempt < args.count:
            await asyncio.sleep(next_delay)

    return ok, failed


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Send isolated, unrelated, always-unique single-turn prompts to "
            "Google Gemini through the web endpoint."
        )
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=6.0,
        help="Seconds between requests (default: 6).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Number of requests to send; 0 means run until interrupted (default: 0).",
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
        help="Random filler messages injected into each request; overrides the level "
        "preset; 0 disables them.",
    )
    parser.add_argument(
        "--temporary",
        action="store_true",
        help="Send each prompt as a temporary chat so it is not saved to Gemini history.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the cookies with one live request, print the response, and exit.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Client initialization timeout in seconds (default: 30).",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=5,
        help="Abort after this many consecutive request failures (default: 5).",
    )
    parser.add_argument(
        "--max-per-day",
        type=int,
        default=None,
        help="Persistent cap on requests per calendar day (default: unlimited).",
    )
    parser.add_argument(
        "--state-file",
        metavar="PATH",
        help="Where to keep the daily counter state (default: next to "
        "--log-file, else gemini_loop.state.json).",
    )
    parser.add_argument(
        "--log-file",
        metavar="PATH",
        help="Append one JSON record per request to this file (rotates at --log-max-bytes).",
    )
    parser.add_argument(
        "--log-max-bytes",
        type=int,
        default=5_000_000,
        help="Rotate the log once it exceeds this many bytes (default: 5 MB).",
    )
    parser.add_argument(
        "--psid",
        default=os.environ.get("GEMINI_SECURE_1PSID", ""),
        help="Value of the __Secure-1PSID cookie (default: $env:GEMINI_SECURE_1PSID).",
    )
    parser.add_argument(
        "--psidts",
        default=os.environ.get("GEMINI_SECURE_1PSIDTS", ""),
        help="Value of the __Secure-1PSIDTS cookie (default: $env:GEMINI_SECURE_1PSIDTS).",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        metavar="TEXT",
        help="Base prompt to send (repeatable); entropy is injected around it. "
        "Default: a pool of unrelated topics.",
    )
    parser.add_argument(
        "--prompt-file",
        metavar="PATH",
        help="Read base prompts from this file, one per line "
        "(blank lines and '#' comments are skipped).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the prompts that would be sent without contacting Gemini.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging from the gemini_webapi library.",
    )
    args = parser.parse_args(argv)

    if args.count < 0:
        parser.error("--count must be >= 0")
    if args.interval < 0:
        parser.error("--interval must be >= 0")
    if args.salt_length is not None and args.salt_length < 0:
        parser.error("--salt-length must be >= 0")
    if args.fillers is not None and args.fillers < 0:
        parser.error("--fillers must be >= 0")
    if args.max_failures < 1:
        parser.error("--max-failures must be >= 1")
    if args.max_per_day is not None and args.max_per_day < 1:
        parser.error("--max-per-day must be >= 1")
    if args.log_max_bytes < 1:
        parser.error("--log-max-bytes must be >= 1")

    prompt_pool = list(args.prompt)
    if args.prompt_file:
        try:
            lines = Path(args.prompt_file).read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            parser.error(f"could not read --prompt-file: {exc}")
        prompt_pool.extend(
            line.strip()
            for line in lines
            if line.strip() and not line.strip().startswith("#")
        )
    args.prompt_pool = prompt_pool or list(UNRELATED_PROMPTS)

    if args.dry_run:
        print("Dry run: no cookies needed, nothing will be sent.", flush=True)
        args.log_writer = None
        args.daily_counter = None
        ok, _ = await run_loop(None, args)
        print(f"Dry run complete: {ok} prompt(s) generated.")
        return 0

    if not args.psid:
        print(
            "Missing the __Secure-1PSID cookie.\n"
            "Set GEMINI_SECURE_1PSID in your environment or pass --psid.\n"
            "See the module docstring for how to copy it from your browser.",
            file=sys.stderr,
        )
        return 2

    try:
        from gemini_webapi import GeminiClient, set_log_level
    except ImportError:
        print(
            "The gemini-webapi package is required for live modes.\n"
            "Install it with: pip install -U gemini_webapi",
            file=sys.stderr,
        )
        return 2

    if args.verbose:
        set_log_level("DEBUG")

    log_writer = None
    daily_counter = None
    if not args.check:
        if args.log_file:
            log_writer = RotatingJsonlWriter(args.log_file, max_bytes=args.log_max_bytes)
        if args.max_per_day is not None:
            if args.state_file:
                state_path = args.state_file
            elif args.log_file:
                state_path = str(args.log_file) + ".count.json"
            else:
                state_path = "gemini_loop.state.json"
            daily_counter = DailyCounter(state_path, args.max_per_day)
            if not daily_counter.allow():
                print(
                    f"Daily cap of {daily_counter.limit} request(s) already reached; "
                    "nothing to do."
                )
                return 0

    args.log_writer = log_writer
    args.daily_counter = daily_counter

    client = GeminiClient(args.psid, args.psidts)
    await client.init(timeout=args.timeout, auto_refresh=True)
    try:
        if args.check:
            try:
                response = await client.generate_content("Reply with the single word: ok")
            except Exception as exc:
                print(f"Live check failed: {exc}", file=sys.stderr)
                return 1
            print("Live check OK. Response preview:")
            print(preview(response.text or ""))
            return 0

        print(
            "Gemini web client initialized. Starting isolated request loop "
            "(Ctrl+C to stop)...",
            flush=True,
        )
        ok, failed = await run_loop(client, args)
    finally:
        await client.close()

    print(f"Done. Sent: {ok}  Failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        exit_code = 130
    raise SystemExit(exit_code)
