"""Trade metadata, and nothing else.

What this file CANNOT express is the point of it. There is no size, no "owned"
set, no position record -- only why we entered and where we planned the stop.
The venue answers what we hold.

That constraint is the whole of REVIEW.md findings 5, 6, 8, 9 and 11. All five
were the same shape: a state file that claimed to know the positions, and
decisions made from it. Losing this file now costs a trade's planned stop, which
`core.protect` replaces with one derived from MAX_RISK. It cannot cost us a
forgotten position.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

DATA = pathlib.Path(__file__).resolve().parents[1] / "research/levels/data"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: pathlib.Path, default: Any) -> Any:
    """Read, or return `default` if the file is absent.

    A file that exists but will not parse RAISES. Falling back to the default
    there is how a corrupt file used to become a cleared halt: an unparseable
    file is a problem to look at, not a fresh start.
    """
    if not path.exists():
        return default
    return json.loads(path.read_text())


def write_json(path: pathlib.Path, obj: Any) -> None:
    """Atomic: tmp then rename, so a kill mid-write cannot leave a fragment.
    This process is SIGKILLed routinely by launchd and by deploys.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


@dataclass
class TradeMeta:
    """Why we took this trade, and what we planned. Never what we hold."""
    reason: str = ""
    planned_stop: float = 0.0
    planned_tp: float = 0.0
    opened_at: str = ""


class MetaStore:
    def __init__(self, path: Optional[pathlib.Path] = None) -> None:
        self.path = path or (DATA / "_trade_meta.json")

    def _all(self) -> Dict[str, Dict[str, Any]]:
        raw = read_json(self.path, {})
        return raw if isinstance(raw, dict) else {}

    def get(self, coin: str) -> Optional[TradeMeta]:
        """Read one coin's metadata, coercing each field.

        Explicit rather than a `**row` splat: this file is hand-editable and a
        missing or junk field must degrade to the default, not raise inside a
        safety path. A planned_stop of 0.0 is then simply "no plan recorded",
        which `core.protect` already handles by deriving one.
        """
        row = self._all().get(coin.upper())
        if not isinstance(row, dict):
            return None
        return TradeMeta(reason=str(row.get("reason") or ""),
                         planned_stop=_f(row.get("planned_stop")),
                         planned_tp=_f(row.get("planned_tp")),
                         opened_at=str(row.get("opened_at") or ""))

    def put(self, coin: str, meta: TradeMeta) -> None:
        rows = self._all()
        rows[coin.upper()] = asdict(meta)
        write_json(self.path, rows)

    def drop(self, coin: str) -> None:
        rows = self._all()
        if rows.pop(coin.upper(), None) is not None:
            write_json(self.path, rows)

    def coins(self) -> set[str]:
        return set(self._all())


def _f(v: Any) -> float:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
