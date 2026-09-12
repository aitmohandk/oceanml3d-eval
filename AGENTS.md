# oceanml3d-eval — Project Guidelines

Workflow: read `PLAN.md` → implement → `pytest -q` + `ruff check .` → dated entry in `CHANGELOG.md`.

## Layout
- `oceanml3d_eval/product.py`, `regions.py` — inputs. `reference/` — truth data readers.
- `oceanml3d_eval/metrics/` — one file per metric family, registered by name.
- `oceanml3d_eval/benchmarks/` — YAML spec loader + runner (cache in `results/`).
- `oceanml3d_eval/report.py`, `cli.py` — leaderboards.
- `benchmarks/`, `products/`, `regions/` — data-free declarative files (`${OCEANML3D_DATA}`).
- `tests/` — synthetic products/drifters built in `conftest.py`; no real data needed.

## Rules
- A metric never reads files itself: it receives an opened product and a loaded reference.
- Scores are flat floats (leaderboard-able); anything else goes in `MetricResult.diagnostics`.
- `oceanml3d_eval/product_contract.py` is a byte-for-byte copy of the file in `oceanml3d-core`; never edit one alone
  (see `docs/product_format.md` for the pinned hash and the change procedure).
- This repo must stay installable without torch/lightning/hydra: never import `oceanml3d` outside a test guarded by
  `pytest.importorskip`.
- When a metric changes, re-save the affected `baselines/*.json` in the same commit, with the reason in the changelog.
- No absolute paths in YAML: use `${OCEANML3D_DATA}`.
- New metrics must be validated on the synthetic fixtures (perfect product → ~0 error, noisier product → worse score).
