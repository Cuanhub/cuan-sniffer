#!/usr/bin/env bash
# One-time setup on the machine OpenClaw runs on. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt

# The lab commits every pre-registration before it runs, so it needs its own
# git history. This is a SEPARATE repository from the live trading repo.
if [ ! -d .git ]; then
  git init -q
  git -c user.name="cuan-lab" -c user.email="cuan-lab@localhost" add -A
  git -c user.name="cuan-lab" -c user.email="cuan-lab@localhost" commit -qm "lab: installed"
fi
git config user.name >/dev/null || git config user.name "cuan-lab"
git config user.email >/dev/null || git config user.email "cuan-lab@localhost"

echo "== selftest"
.venv/bin/python lab.py selftest
echo "== first data refresh (real Hyperliquid candles, read-only)"
.venv/bin/python lab.py refresh || echo "refresh failed -- check network, then: .venv/bin/python lab.py refresh"
.venv/bin/python lab.py status
