# Vendored DyLAN sources (unmodified)

Upstream: https://github.com/SALT-NLP/DyLAN, commit `006e440a519f7cf21e2826f3b8033d84ae9bf07c`
(`LICENSE`, `code/MMLU/*`), MIT License (see `LICENSE`, Copyright (c) 2023 SALT).

These files are byte-identical copies. `../../dylan.py` imports the prompt constants from
`MMLU/prompt_lib.py` and `test_dylan.py` runs `MMLU/LLMLP.py` with a stubbed LLM to check that the
adapter reproduces the native control flow. Do not edit; the adapter manifest records their hashes.
