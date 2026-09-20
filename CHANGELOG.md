# Changelog

## 2026-09-20: `split --depth-indices`, and a truth the OSSE-3D benchmark can actually use

**Summary:** `split` turns a depth-resolved store into one variable per level (`thetao_d00`,
`u_d12`...), so the GLORYS truth of `osse3d_gs21` can be built with one command.
`products/glorys_gs21_truth.yaml` now lists the 64 variables the benchmark asks for instead of
three and a comment.

**Files modified:** `oceanml3d_eval/split.py` -- `depth_indices`, `_flatten_depth`,
`parse_indices`, manifest validated against the contract before it is written, missing variables
and out-of-range levels refused; `oceanml3d_eval/cli.py` -- `--depth-indices '0,2,4' | '0-25' |
'all'`; `products/glorys_gs21_truth.yaml`, `products/README.md`, `README.md`;
`tests/test_split_plots.py`.

**Rationale:** The benchmark could not be run as shipped: its reference manifest was a stub, and
`split` only ever kept one level, under the plain name (`u`, not `u_d05`), so nothing produced the
`<var>_d<ii>` layout that both the benchmark and the products exported by `oceanml3d-core` use.
Levels are selected by position on the depth axis, not by metres: that is what `depth_index` means
in core's `config/data/osse3d_gs21.yaml`, and reading them as metres would silently shift every
level the day the truth is prepared at another vertical sampling.

**Verification:** `pytest` -- 43 passed. End to end on a synthetic GLORYS-like store: `split
--depth-indices 0,2,4,6,8,10-25` wrote a 64-variable truth, and `oceanml3d-eval run -b osse3d_gs21`
produced the leaderboard and the skill-vs-depth table against a model product.


## 2026-09-11: Rename to `oceanml3d-eval`, package included

**Summary:** the repository directory, the distribution, the import package and every reference to
them now read `oceanml3d`. Unlike `oceanml3d-core`, the name here is genuinely *in the code*:
67 `from oceanml_eval …` / `import oceanml_eval` statements, plus packaging metadata.

**Files modified:**
- `oceanml_eval/` → `oceanml3d_eval/` — the package directory, and all its intra-package imports
- `pyproject.toml` — `name = "oceanml3d-eval"`, `include = ["oceanml3d_eval*"]`, the console script
  `oceanml3d-eval = "oceanml3d_eval.cli:main"`, the five plugin entry points, and the
  entry-point **group** `[project.entry-points."oceanml3d_eval.metrics"]` (a published contract:
  any third-party metric plugin must now register under the new group name)
- `tests/test_product_contract.py`, `docs/product_format.md` (and the donor prototype's two
  counterparts) — `PINNED_SHA256` re-baselined from `a55075a5…` to `dd44f0e1…`
- `.github/workflows/{ci,e2e}.yml`, `AGENTS.md`, `PLAN.md`, `README.md`, `docs/`, `products/`,
  `benchmarks/` — the `${OCEANML3D_DATA}` variable and the path references

**Rationale:** requested rename. The distribution name *is* the project name, so leaving
`oceanml_eval` would have made `pyproject.toml` contradict the repository it describes.

**On the re-pinned hash:** `product_contract.py` is shared byte-for-byte with the donor prototype
and sealed by a sha256. Its only change is the module docstring (`oceanml` → `oceanml3d` on three
lines) — no behaviour moved, `PRODUCT_FORMAT_VERSION` stays at 2. The two copies were re-checked
with `cmp` after the rewrite and are still identical, so the cross-repo contract holds; the seal was
re-calibrated in all four places at once, as the test's own docstring instructs.

**Verification:** `pytest -q` — **35 passed, 3 skipped** (the skips are missing optional plotting /
numba dependencies in this venv, unrelated). `ruff check .` clean. `oceanml3d-eval --help` works and
`importlib.metadata.entry_points(group="oceanml3d_eval.metrics")` resolves all five metrics, so the
plugin registry survived the group rename.

**Note on the virtualenv:** `.venv/` held a poetry editable install pinned to the old absolute path
(`.pth`, `dist-info`, and 14 script shebangs). Shebangs and `pyvenv.cfg` were rewritten and the
package reinstalled with `pip install -e . --no-deps` — poetry's own install path needs `setuptools`,
which this venv does not carry. Functionally equivalent; run `poetry install` to restore poetry's
exact layout.

## 2026-09-10: coherence/integrity audit fixes

**Summary:** gridded-family metrics (`gridded_rmse`, `spectral_score`, `ensemble_scores`) now default
`variables` to the product∩reference shared fields instead of the hard `["u", "v"]`, so the
depth-resolved `osse3d_gs21` benchmark scores (it previously crashed in `align`); CI `ruff check .`
made green (`third_party`/`legacy_from_fm` truly excluded via `force-exclude`, lint scoped to
`oceanml3d_eval tests`, remaining `UP038` fixed, hash-pinned `product_contract.py` ignored not edited);
`spectral_score` in `osse3d_gs21` pinned to `[ssh]` (leaderboard only reads `eff_resolution_km_ssh`);
divide-by-zero in `effective_resolution` silenced.
**Verification:** `pytest -q` — 36 passed, 2 skipped; `ruff check oceanml3d_eval tests` clean.

## 2026-09-08: ensemble support (audit follow-up)

**Summary:** product format v2 (`ensemble_size` + `coords.member`), `metrics/ensemble.py`
(`ensemble_scores`: CRPS and energy score in their fair forms, spread, spread-skill ratio with the
`sqrt((m+1)/m)` correction, rank histogram), ported and extended from `4dvarnet-fm-opencode`.
**Rationale:** probabilistic methods (EnKF, flow matching, score-based DA) had no way to be described
or scored; see `docs/feature_inventory.md`.
**Verification:** `pytest -q` — 37 tests.

## 2026-09-08: contract hardening (review follow-up)

**Summary:** `product_contract.py` shared with oceanml3d-core (hash-pinned, validated on load),
`oceanml3d_eval/regression.py` + `oceanml3d-eval baseline --save/--check`, cross-repo contract test,
end-to-end CI workflow.
**Verification:** `pytest -q` — 28 tests.

## 2026-09-08: Phase 2

**Summary:** `split` command, `plots.py` + `plot` command, vendored `velocity_metrics` with a bridge
(`velocity_metrics_backend.py`) selectable through `backend: velocity_metrics`.
**Verification:** `pytest -q` — 10 tests.

## 2026-09-08: OSSE-3D metrics (NOSC branch first_implementation)

**Summary:** isotropic 2D PSD option in `spectral_score`, `report.depth_profile` + `--depth-profile` CLI,
`osse3d_gs21` benchmark, `GulfStream_eval` region.
**Rationale:** per-depth skill curve is the central result of the 3D extension.
**Verification:** `pytest -q` — 8 tests.

## 2026-09-08: Initial skeleton

**Summary:** Generic evaluation framework extracted from NOSC (`run_*_GL.py`, `metric/dictionary`, `env/velocity_metrics`) and the placeholder `evaluation/` of `4dvarnet-ocean-reanalyses`.
**Files modified:** everything (new repo).
**Rationale:** evaluate neural models and reference products with one declarative benchmark definition.
**Verification:** `pytest -q` — 7 tests on synthetic products/drifters.
