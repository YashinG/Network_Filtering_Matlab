# CLAUDE.md

This repo holds MATLAB code for network filtering (MST/PMFG), DBHT clustering, and QIS/EWMA correlation estimation of stock returns, plus a verified Python port in `python/netfilter`. The full history of how the port and the multiplex extension were built, and the decisions behind them, is in **`docs/HANDOFF.md`**. Read it before making non-trivial changes.

## Layout
- `Functions/`, `Scripts/`, `External Functions/`: the original MATLAB code. Run the scripts from `Scripts/`, because they use relative paths.
- `python/netfilter/`: the Python package.
- `python/scripts/run_*.py`: one per MATLAB script, plus `run_multiplex.py`.
- `python/tests/`: pytest. `tests/data/*.json` holds the MATLAB R2025b reference outputs.
- `Data/Data.mat`: 20 stocks, 199 weekly returns. `netfilter.load_data()` reads it.

## Commands
```bash
cd python
pip install -r requirements.txt
pytest -W ignore::DeprecationWarning          # 60 tests; 2 skip without the template data
MULTIPLEX_TEMPLATE_DATA=/path/to/ngozzi/multiplex/data pytest   # also runs the template checks
python scripts/run_multiplex.py               # figures and tables go to python/results/ (gitignored)
```

## Rules for changes
- **Keep MATLAB parity.** `tests/test_matlab_reference.py` must keep passing. Function names and positional argument order mirror the MATLAB functions, and result dictionaries use the MATLAB struct field names.
- **Known deliberate differences:** ranks treat values equal to 12 significant figures as ties; DBHT linkage rows are sorted by height; the leaf-ordering criterion is `'adjacent'`. These are documented in `python/README.md` and `docs/HANDOFF.md`. Don't "fix" them back.
- **Match MATLAB's numerics, not textbook ones.** Closeness is `(A/(N-1))^2 / C`. Betweenness is unnormalised. Eigenvector centrality is scaled to sum to 1. PageRank uses power iteration with a stopping tolerance of 1e-4.
- **Multiplex layers:** new built-in measures go in `BUILTIN_MEASURES` in `multiplex.py` as a `(function, kind)` pair. They must accept time weights, and they need a test against an independent reference.
- **Template data:** don't copy data from `ngozzi/multiplex`, which has no licence.
- **Git:** commit messages end with the session's attribution lines.
