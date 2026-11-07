"""Settings that look live but are read by nothing.

On 2026-10-27 the live .env held 172 variables and the code read 6 of them.
The other 166 came from earlier systems, and several described a bot that
does not exist: RESEARCH_ONLY_MODE=true and PAUSE_NEW_SIGNALS=true while it
traded live, HARD_BLOCKED_COINS=ARB,... while nothing blocked ARB (it traded
ARB on 2026-10-17), MAX_OPEN_POSITIONS=3 against a live cap of 4, a 15%
drawdown halt against a live 10%. A switch that looks real and does nothing
is worse than no switch: someone sets TRADING_DISABLED=true to stop the bot,
and it keeps trading.

So the armed trader compares what .env SETS with what the live code READS,
and refuses to arm while they disagree. "Read" is taken from the source
itself -- every getenv / environ.get / config helper call naming a variable
-- so the list cannot drift from the code the way a hand-kept list does.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Set

REPO = Path(__file__).resolve().parents[1]
# Every module a live service runs. If a service starts reading a setting from
# a file not listed here, test_config_audit fails on the missing key.
SOURCES = ("notify.py", "core/*.py", "tools/live_levels_trader.py",
           "tools/protect_loop.py", "tools/telegram_control.py", "tools/retention.py",
           "tools/liveness.py",
           "research/levels/*.py")
_READ = re.compile(r"""(?:\bgetenv|\benviron\.get|\b_f|\b_i|\b_b)\(\s*["']([A-Z][A-Z0-9_]*)["']""")
_SET = re.compile(r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]*)\s*=")


def keys_read(repo: Path = REPO) -> Set[str]:
    out: Set[str] = set()
    for pattern in SOURCES:
        for p in sorted(repo.glob(pattern)):
            out.update(_READ.findall(p.read_text()))
    return out


def keys_set(env_file: Path) -> List[str]:
    if not env_file.exists():
        return []
    return [m.group(1) for line in env_file.read_text().splitlines()
            if (m := _SET.match(line))]


def dead_keys(env_file: Path = REPO / ".env", repo: Path = REPO) -> List[str]:
    """Variables the .env sets that no live code reads, sorted."""
    read = keys_read(repo)
    return sorted({k for k in keys_set(env_file) if k not in read})


def refusal(dead: List[str]) -> str:
    """Why an armed trader will not arm, or "" when it may."""
    if not dead:
        return ""
    shown = ", ".join(dead[:12]) + (f" and {len(dead) - 12} more" if len(dead) > 12 else "")
    return (f".env sets {len(dead)} variable(s) that NOTHING reads, so they configure "
            f"nothing: {shown}. Remove them (.env.example lists what is real), then "
            f"restart.")
