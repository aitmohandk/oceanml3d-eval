# oceanml3d-eval

Benchmarks and metrics for **ocean reconstruction products** — neural models exported by
[`oceanml3d-core`](../oceanml3d-core) and reference products (DUACS, GlobCurrent, NeurOST,
GLORYS…) are evaluated by exactly the same code.

```
product.yaml (+ daily NetCDF) ──► oceanml3d-eval run -b <benchmark> ──► results/<bench>/<product>/*.json
                                                                        └─► oceanml3d-eval report → leaderboard
```

## Install & run

```bash
pip install -e ".[dev]"
oceanml3d-eval list                                   # benchmarks / metrics / regions
export OCEANML3D_DATA=/path/to/data                   # used by benchmarks/*.yaml and products/*.yaml
oceanml3d-eval run -b surface_currents_15m -p outputs/nosc_15m_duacs/<run>/product/product.yaml --baselines
oceanml3d-eval report -b surface_currents_15m --region "Gulf Stream" --markdown leaderboard.md
oceanml3d-eval report -b osse3d_gs21 --depth-profile nrmse            # skill-vs-depth table
oceanml3d-eval plot -b osse3d_gs21 --depth-profile nrmse --figure skill_depth.png
oceanml3d-eval split --input duacs_2019.nc --out products/duacs --name duacs --var u=ugos --var v=vgos --depth-m 15
# a 3D reference: one variable per level (<var>_d<ii>), the naming the OSSE-3D benchmark uses
oceanml3d-eval split --input $OCEANML3D_DATA/glorys/glorys_gs_multidepth_2010-2020.zarr \
    --out $OCEANML3D_DATA/glorys_gs21_truth --name glorys_gs21_truth \
    --var ssh=zos --var thetao=thetao --var u=uo --var v=vo --depth-indices 0,2,4,6,8,10-25
```

## Concepts

| Object | File | Role |
|---|---|---|
| `product_contract.py` | `oceanml3d_eval/` | The format contract, byte-identical with `oceanml3d-core` and hash-pinned; every manifest is validated on load. |
| `ProductSpec` / `open_product` | `oceanml3d_eval/product.py` | Reads any product from a manifest (`product.yaml`, or legacy NOSC `*.json`) into `xr.Dataset(u, v, ssh…)` with `time/lat/lon`. |
| `Region` | `oceanml3d_eval/regions.py` | bbox + polygon from `regions/region_<name>.json` (NOSC / velocity_metrics format). |
| `Reference` | `oceanml3d_eval/reference/` | `drifters_points` (NetCDF/CSV/`.pyo.gz` → DataFrame) or `gridded` (a product used as truth). |
| `Metric` | `oceanml3d_eval/metrics/` | `compute(product, reference, region, first, last) -> MetricResult(scores, diagnostics)`. |
| `BenchmarkSpec` | `benchmarks/*.yaml` | period + references + regions + metrics + baselines + leaderboard columns. |
| `run_benchmark` | `oceanml3d_eval/benchmarks/runner.py` | loops metrics × regions, caches JSON scores and NetCDF diagnostics. |

## Metrics

| name | reference | scores | diagnostics |
|---|---|---|---|
| `eulerian_drifters` | drifter points | `rmse_u/v/vec`, `bias`, `corr`, `var_explained`, `n_obs` | binned RMSE map |
| `lagrangian_drifters` | drifter points (with `id`) | `sep_km_dayN` (RK4 advection; optional `velocity_metrics` backend) | — |
| `gridded_rmse` | gridded truth | `rmse`, `bias`, `nrmse` (÷ anomaly std), `anom_corr`, `var_explained`, and `mu`/`sigma` (SSH data-challenge, ÷ RMS); area-weighted | time-mean RMSE map |
| `ensemble_scores` | gridded truth | `crps`, `energy_score`, `spread`, `rmse_mean`, `spread_skill` | rank histogram |
| `spectral_score` | gridded truth | `eff_resolution_km` (PSD err/ref = 0.5; along-lon or `isotropic: true`) | PSDs |

## Benchmarks

| name | task | references | from |
|---|---|---|---|
| `surface_currents_15m` | 15 m currents 2019 | AOML drogued drifters | NOSC (`run_rmse_GL.py`, `run_lagrangian_GL.py`) |
| `surface_currents_00m` | surface currents 2019 | AOML undrogued drifters | NOSC |
| `ssh_mapping_osse` | SSH vs GLORYS truth | GLORYS | ocean data challenges |
| `ocean_reanalysis_osse` | full surface state OSSE | GLORYS | 4dvarnet-ocean-reanalyses CS1–CS4 |
| `osse3d_gs21` | 3D OSSE Gulf Stream, 21 levels: nRMSE + isotropic effective resolution per level | GLORYS, built with `split --depth-indices` (see `products/glorys_gs21_truth.yaml`) | NOSC `first_implementation` (`depth_profile_metrics.py`) |

Ensemble products (EnKF, flow-matching samplers) declare `ensemble_size` and a `member` coordinate
(product format v2); a deterministic product is scored as a 1-member ensemble, so both appear on the
same leaderboard line. What each original repo could do and where it went:
[`docs/feature_inventory.md`](docs/feature_inventory.md).

## Non-regression gate

```bash
oceanml3d-eval baseline -b surface_currents_15m -p .../product.yaml --save     # freeze reference scores
oceanml3d-eval baseline -b surface_currents_15m -p .../product.yaml --check --rtol 1e-3
```

`--check` recomputes and fails with a per-score diff (`rmse_u: 0.0812 -> 0.0894 (+10.10%)`). Pin the reference
NOSC experiment this way before refactoring anything: a migration that changes the science then fails a test
instead of going unnoticed. `baselines/*.json` belongs in git.

## Adding a metric / benchmark

* Metric: subclass `Metric`, decorate `@register_metric("name")` (or declare an
  `oceanml3d_eval.metrics` entry point), set `needs` to the reference kind. Add a test on the
  synthetic fixtures in `tests/conftest.py`.
* Benchmark: one YAML in `benchmarks/`. Baselines are product manifests in `products/`.
* Region: drop a `region_<name>.json` in `regions/`.

`third_party/velocity_metrics/` is the vendored OceanDataLab package (LGPL) used by NOSC;
`pip install -e third_party/velocity_metrics` then `options: {backend: velocity_metrics, ...}` on
`lagrangian_drifters` reproduces the paper's Lagrangian/SDE diagnostics bit-for-bit
(`oceanml3d_eval/velocity_metrics_backend.py` writes the legacy JSON descriptors for you).
`oceanml3d_eval/plots.py` gives RMSE maps, PSD ratios and skill-vs-depth curves.
