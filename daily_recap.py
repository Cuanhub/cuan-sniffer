# daily_recap.py

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone

from notifier import send_telegram_message


def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def run_analyzer(python_bin: str, extra_args: list[str]) -> tuple[int, str, str]:
    cmd = [python_bin, "analyze_winrate.py", *extra_args]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def extract_summary(stdout: str) -> str:
    if not stdout:
        return "No analyzer output."

    lines = [line.rstrip() for line in stdout.splitlines()]
    picked: list[str] = []

    keep_prefixes = (
        "Fetched:",
        "Using ",
        "FILTER:",
        "  Signals",
        "  Resolved",
        "  Timeouts",
        "  Win rate",
        "  Total R",
        "  Mean R",
        "  Sharpe",
        "  Sortino",
        "  Max DD",
        "  Best streak",
        "  Worst streak",
    )

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(keep_prefixes):
            picked.append(line)

    # keep top rows from By Coin if present
    sections_to_keep = {"By Coin", "By Setup Family", "By Side"}
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped in sections_to_keep:
            picked.append("")
            picked.append(stripped)
            j = i + 1
            kept_rows = 0
            while j < len(lines):
                s = lines[j].strip()
                if not s:
                    j += 1
                    continue
                if s.startswith("By ") and s not in sections_to_keep:
                    break
                if s.startswith("─"):
                    j += 1
                    continue
                if not s.startswith("FILTER:"):
                    picked.append(lines[j])
                    kept_rows += 1
                if kept_rows >= 5:
                    break
                j += 1
            i = j
            continue
        i += 1

    # fallback
    if not picked:
        picked = lines[:30]

    # dedupe while preserving order
    seen = set()
    final_lines = []
    for line in picked:
        key = line.rstrip()
        if key not in seen:
            seen.add(key)
            final_lines.append(line)

    text = "\n".join(final_lines).strip()
    return text[:3500]


def build_message(summary_text: str, chart_name: str | None, mode_label: str) -> str:
    extra = f"\nChart: `{chart_name}`" if chart_name else ""
    return (
        "📊 *Cuan Sniffer Daily Recap*\n\n"
        f"Mode: `{mode_label}`\n"
        f"Time: `{utc_now_str()}`{extra}\n\n"
        f"```text\n{summary_text}\n```"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run daily analyzer recap and send to Telegram.")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print recap instead of sending to Telegram",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Pass --no-fetch to analyzer",
    )
    parser.add_argument(
        "--coin",
        help="Optional coin filter forwarded to analyzer",
    )
    parser.add_argument(
        "--side",
        help="Optional side filter forwarded to analyzer",
    )
    parser.add_argument(
        "--htf",
        help="Optional HTF filter forwarded to analyzer",
    )
    parser.add_argument(
        "--macro",
        help="Optional macro filter forwarded to analyzer",
    )
    parser.add_argument(
        "--regime",
        help="Optional regime filter forwarded to analyzer",
    )
    parser.add_argument(
        "--session",
        help="Optional session filter forwarded to analyzer",
    )
    parser.add_argument(
        "--chart",
        default="winrate_report.png",
        help="Chart filename to pass to analyzer",
    )
    args = parser.parse_args()

    analyzer_args: list[str] = ["--chart", args.chart]

    if args.no_fetch:
        analyzer_args.append("--no-fetch")
    if args.coin:
        analyzer_args.extend(["--coin", args.coin])
    if args.side:
        analyzer_args.extend(["--side", args.side])
    if args.htf:
        analyzer_args.extend(["--htf", args.htf])
    if args.macro:
        analyzer_args.extend(["--macro", args.macro])
    if args.regime:
        analyzer_args.extend(["--regime", args.regime])
    if args.session:
        analyzer_args.extend(["--session", args.session])

    mode_parts = []
    if args.no_fetch:
        mode_parts.append("cache-only")
    if args.coin:
        mode_parts.append(f"coin={args.coin}")
    if args.side:
        mode_parts.append(f"side={args.side}")
    if args.htf:
        mode_parts.append(f"htf={args.htf}")
    if args.macro:
        mode_parts.append(f"macro={args.macro}")
    if args.regime:
        mode_parts.append(f"regime={args.regime}")
    if args.session:
        mode_parts.append(f"session={args.session}")
    mode_label = ", ".join(mode_parts) if mode_parts else "full"

    code, stdout, stderr = run_analyzer(args.python, analyzer_args)

    if code != 0:
        msg = (
            "❌ *Cuan Sniffer Daily Recap Failed*\n\n"
            f"Time: `{utc_now_str()}`\n"
            f"Exit code: `{code}`\n\n"
            f"```text\n{(stderr or stdout or 'Unknown error')[:3000]}\n```"
        )
        if args.dry_run:
            print(msg)
        else:
            send_telegram_message(msg)
        raise SystemExit(code)

    summary = extract_summary(stdout)
    msg = build_message(summary, args.chart, mode_label)

    if args.dry_run:
        print(msg)
    else:
        send_telegram_message(msg)
        print("[DAILY_RECAP] Recap sent.")


if __name__ == "__main__":
    main()