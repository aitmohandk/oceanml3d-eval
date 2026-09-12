# oceanml3d-eval — Plan

## 0. Provenance
This repository is new. Its evaluation abstractions come from the oceanml3d prototype; the code it
must replace was copied into `legacy_from_fm/` from `4dvarnet-fm-opencode` (`evaluation/` + the 12
root `eval_*.py` drivers, ~11.8 k lines). The upstream repositories are read-only references and are
never modified.

## 0b. Conversion of legacy_from_fm (strangler fig)
- [ ] pin the numbers each legacy driver currently produces (fixed seed, small case)
- [ ] convert one driver at a time into `benchmarks/*.yaml` + a metric, delete it only once its
      numbers reproduce within tolerance
- [ ] `evaluation/baselines.py` (3 014 lines) moves to `oceanml3d-core` as registered models
      (assimilation family), not here: a filter produces a product like any other method

Open work only; the audit of what the original repos could do is in
[`docs/feature_inventory.md`](docs/feature_inventory.md) (shared with `oceanml3d-core`).

## 1. Needs real data
- [ ] validate the AOML / CMEMS drifter readers on the real yearly files (`columns` mapping)
- [ ] cross-check the built-in RK4 Lagrangian backend against `velocity_metrics` on one region/year
- [ ] reproduce the NOSC paper tables (DUACS / GlobCurrent / NeurOST / `nosc_unet`) and freeze them with `baseline --save`

## 2. Metrics still to port (see inventory §1.4, §2)
- [ ] explained variance and anomaly correlation (fm `evaluation/metrics.py`, reanalyses `training/metrics.py`)
- [ ] SW component metrics (fm) for multi-component states
- [ ] fronts / FSLE diagnostics (from the vendored `velocity_metrics`)
- [ ] observation-space scores (innovation statistics) for filters that report an analysis and a forecast

## 3. References and tasks
- [ ] along-track altimetry reference (`reference/alongtrack.py`) for OSE SSH benchmarks
- [ ] real Argo profiles as a reference for 3D reanalyses (T, S) — the OSSE→OSE step
- [ ] benchmarks for the toy systems (Lorenz-96 with an ensemble truth), so a method is compared on
      toy and ocean tasks with the same leaderboard code

## 4. Reporting
- [ ] HTML report + leaderboard published by CI
- [ ] trajectory plots for the Lagrangian metric (`plots.py`)

## Done (summary)
Phase 1 skeleton (products, regions, references, 4 metric families, benchmarks, runner, leaderboard) ·
Phase 2 NOSC parity (`split`, plots, vendored `velocity_metrics` backend) · Phase 2b contract and
non-regression (`product_contract.py` hash-pinned, `baseline --save/--check`, cross-repo test, e2e CI) ·
Phase 2c ensemble support (product format v2, CRPS, energy score, spread-skill, rank histogram).
