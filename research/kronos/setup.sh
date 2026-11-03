#!/bin/sh
# Build research/kronos/.venv and fetch the pinned Kronos weights.
#
# Its own environment on purpose: PyTorch is ~1 GB, and keeping the live
# bot's environment free of it -- and this one free of the Hyperliquid SDK
# and signing libraries -- means nothing run from here can place an order.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="${PYTHON:-python3}"

"$PY" -m venv "$HERE/.venv"
"$HERE/.venv/bin/python" -m pip install -q --upgrade pip
"$HERE/.venv/bin/python" -m pip install -q -r "$HERE/requirements.txt"

cd "$ROOT"
HF_HOME="$HERE/hf" HF_HUB_DISABLE_TELEMETRY=1 "$HERE/.venv/bin/python" - <<'EOF'
import hashlib
from huggingface_hub import hf_hub_download
from research.kronos.forecast import REVISIONS

WANT = {
    ("NeoQuasar/Kronos-small", "model.safetensors"):
        "b082dfcbd8e8c142a725c8bbb99781802f38fec81210e13479effb32b3c3e020",
    ("NeoQuasar/Kronos-Tokenizer-base", "model.safetensors"):
        "59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee",
}
for (repo, name), sha in WANT.items():
    path = hf_hub_download(repo, name, revision=REVISIONS[repo])
    hf_hub_download(repo, "config.json", revision=REVISIONS[repo])
    got = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if got != sha:
        raise SystemExit(f"{repo}/{name}: sha256 {got}, expected {sha} -- refusing")
    print(f"ok  {repo}@{REVISIONS[repo][:10]}  {name}")
EOF
"$HERE/.venv/bin/python" -m pytest -q "$HERE"
