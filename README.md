# Santa 2026 – The Elf Calculator

Tools for the Kaggle competition [Santa 2026 – The Elf Calculator](https://www.kaggle.com/competitions/santa-2026):
write, for each of 1,000 puzzles, the shortest arithmetic program (`+ - * /` over 48 registers) that produces every
target number. Score = total instructions; lower is better.

The competition data is not included. Download it into `data/` with
`python -m kaggle competitions download -c santa-2026 -p data` (after accepting the rules) and unzip.

| File | What it does |
|---|---|
| `score.py` | Strict validator and scorer: replays every program against all rules. `python score.py sub.csv` |
| `solve.py` | Greedy constructive solver (one-step reuse, shared helpers, meet-in-the-middle via `mitm.py`, chain fallback) |
| `mitm.py` | Numba meet-in-the-middle search over values 1–3 instructions from the register pool |
| `exact.py` | Optimal programs for tiny puzzles by iterative-deepening search |
| `merge.py` | Keeps the shortest valid program per puzzle across several submission files |
| `refresh.py` | Pulls public notebook outputs, merges, optionally submits (`--no-submit`, `--ids lo-hi`) |
| `submit_api.py` | Submits over IPv4 with a timeout (the CLI can hang on a broken IPv6 route) |
| `lns_driver.py` | Runs a C++ large-neighbourhood-search engine on a submission (`--best`, `--ids`) |
| `polish.py`, `resynth.py` | Local post-optimisers (delete-and-reroute, window re-synthesis) |
| `finalize_range.py` | Builds an updated file for a puzzle range, keeping all other puzzles unchanged |

`lns_driver.py` expects `bin/elfopt.exe`, built from the public engine in the Kaggle dataset
`goodpjw2008/santa-2026-dag-lns-programs` (`g++ -std=c++17 -O3 -march=native -o bin/elfopt.exe elfopt.cpp`).
That code is not part of this repository.
