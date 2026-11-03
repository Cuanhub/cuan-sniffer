# Kronos: where this code and these weights came from

## Code (vendored, unmodified)

`model/__init__.py`, `model/kronos.py` and `model/module.py` are copied byte for
byte from https://github.com/shiyu-coder/Kronos at commit `67b630e`
(2026-04-13, "Merge pull request #243"). MIT licence, `LICENSE` beside this
file. Nothing here is edited: the cache and the input handling live in
`research/kronos/`, outside `vendor/`, and `test_kronos_isolation` checks
these hashes.

| file | SHA-256 |
|---|---|
| model/__init__.py | f8f856ca3fedadcaac97e196be23d1aeda1c3c9ffe8903d66d43ea3bcac6240c |
| model/kronos.py | 0a5f90282e2039c2de0771473419715c845def154896dbd0f5747837e6241032 |
| model/module.py | a07edbadc0e96804c8158c021bbc6063bb7cc43b34d7fc470d5c8ff2005a409f |
| LICENSE | acb2d194d378204e5f2be4dcd24d39ecac437903620c790c3315a96dab388fdc |

Reviewed before first use on 2026-11-03: PyTorch model code, no `torch.load`
or pickle, no network access of its own. Weights load through
`huggingface_hub.PyTorchModelHubMixin` from safetensors files only.

## Weights (downloaded by setup.sh into research/kronos/hf, not committed)

| repository | revision | file | SHA-256 |
|---|---|---|---|
| NeoQuasar/Kronos-small | 901c26c1332695a2a8f243eb2f37243a37bea320 | model.safetensors | b082dfcbd8e8c142a725c8bbb99781802f38fec81210e13479effb32b3c3e020 |
| | | config.json | 5e0f6a605d5f81b5c9b559fe5cf716a1acb041c744e6f41bd05b097b7a685396 |
| NeoQuasar/Kronos-Tokenizer-base | 0e0117387f39004a9016484a186a908917e22426 | model.safetensors | 59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee |
| | | config.json | 2366e7ccfec76cbc19cf3c4c1b9c5d901be336ca1e83f2d2292c9bff381b77a2 |

Hugging Face reports both repositories created 2025-06-30 and last modified
2025-09-09. The paper (arXiv 2508.02739) states its pre-training data extends
to June 2024. Every bar this project tests on is from 2026, so none of it can
be in the weights.
