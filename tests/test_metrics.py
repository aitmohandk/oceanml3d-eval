import numpy as np
import xarray as xr

from oceanml3d_eval.metrics.eulerian import EulerianDrifters
from oceanml3d_eval.metrics.gridded import GriddedRMSE
from oceanml3d_eval.metrics.spectral import SpectralScore, effective_resolution
from oceanml3d_eval.product import open_product
from oceanml3d_eval.reference.drifters import read_drifters
from oceanml3d_eval.regions import Region


def test_eulerian_truth_is_perfect_and_ordering(products, drifters):
    obs = read_drifters(str(drifters), columns={"time": "date", "u": "ums", "v": "vms"})
    region = Region.get("GulfStream")
    m = EulerianDrifters(bin_deg=5)
    r_truth = m.compute(open_product(products["truth"]), obs, region, "2019-01-01", "2019-01-12")
    r_noisy = m.compute(open_product(products["noisy"]), obs, region, "2019-01-01", "2019-01-12")
    assert r_truth.scores["n_obs"] > 100
    assert r_truth.scores["rmse_vec"] < 1e-4
    assert 0.05 < r_noisy.scores["rmse_u"] < 0.2 and r_noisy.scores["rmse_vec"] > r_truth.scores["rmse_vec"]
    assert "rmse_map" in r_truth.diagnostics


def test_gridded_and_spectral(products):
    truth = open_product(products["truth"])
    region = Region.get("GulfStream")
    g = GriddedRMSE().compute(open_product(products["noisy"]), truth, region, "2019-01-01", "2019-01-12")
    assert abs(g.scores["rmse_u"] - 0.1) < 0.02 and 0 < g.scores["mu_u"] < 1
    s = SpectralScore().compute(open_product(products["noisy"]), truth, region, "2019-01-01", "2019-01-12")
    assert np.isfinite(s.scores["eff_resolution_km_u"]) and s.scores["eff_resolution_km_u"] > 0


def test_effective_resolution():
    k = np.linspace(0.001, 0.1, 100)
    ref = np.ones_like(k)
    err = np.linspace(0, 1, 100)
    assert abs(effective_resolution(k, err, ref) - 1 / k[np.argmin(np.abs(err - 0.5))]) < 5


def test_isotropic_spectral_and_depth_profile(products, tmp_path):
    import pandas as pd

    from oceanml3d_eval.report import depth_profile

    truth = open_product(products["truth"])
    region = Region.get("GulfStream")
    s = SpectralScore(isotropic=True, time_stride=4).compute(open_product(products["noisy"]), truth, region, "2019-01-01", "2019-01-12")
    assert np.isfinite(s.scores["eff_resolution_km_u"]) and s.scores["eff_resolution_km_u"] > 0
    df = pd.DataFrame([{"product": "m", "region": "GS", "metric": "gridded_rmse", "nrmse_thetao_d00": 0.1, "nrmse_thetao_d12": 0.3, "nrmse_ssh": 0.05}])
    prof = depth_profile(df, "nrmse")
    assert list(prof.depth_index) == [0, 12] and list(prof.variable) == ["thetao", "thetao"]


def test_depth_resolved_variables_scored_without_variables_option(truth_field):
    """osse3d_gs21: metrics get no `variables` option; the depth-suffixed fields must still score."""
    import pandas as pd
    import xarray as xr

    from oceanml3d_eval.report import depth_profile

    base = truth_field.sel(lat=slice(33, 43), lon=slice(-65, -55))
    names = ["ssh", "thetao_d00", "thetao_d12", "u_d00"]
    rng = np.random.default_rng(0)
    truth = xr.Dataset({n: base["u"] if n.startswith(("ssh", "thetao")) else base["v"] for n in names})
    prod = truth + 0.1 * rng.standard_normal((truth.sizes["time"], truth.sizes["lat"], truth.sizes["lon"])).astype("f4")
    region = Region.get("GulfStream_eval")
    g = GriddedRMSE(product_spec=None).compute(prod, truth, region, "2019-01-01", "2019-01-12")
    assert {"nrmse_ssh", "nrmse_thetao_d00", "nrmse_thetao_d12", "nrmse_u_d00"} <= set(g.scores)
    s = SpectralScore(isotropic=True, time_stride=4).compute(prod, truth, region, "2019-01-01", "2019-01-12")
    assert np.isfinite(s.scores["eff_resolution_km_ssh"])
    prof = depth_profile(pd.DataFrame([{"product": "m", "region": "GS", "metric": "gridded_rmse", **g.scores}]), "nrmse")
    assert list(prof[prof.variable == "thetao"].depth_index) == [0, 12]



# --- gridded scores: normalised by the anomaly, weighted by area ----------------------------------

def _pair(mean=0.0, error=0.5, days=6, n=8, lat0=30.0, seed=0):
    """A truth with a given mean and unit-ish variability, and a prediction with a constant error."""
    import pandas as pd
    import xarray as xr

    rng = np.random.default_rng(seed)
    time = pd.date_range("2019-01-01", periods=days)
    lat, lon = np.linspace(lat0, lat0 + 8, n), np.linspace(-64, -56, n)
    anom = rng.normal(size=(days, n, n))
    truth = xr.DataArray(mean + anom, dims=("time", "lat", "lon"),
                         coords={"time": time, "lat": lat, "lon": lon})
    return truth + error, truth


def test_nrmse_does_not_depend_on_the_mean_of_the_field():
    """The bug this replaces: with rms normalisation a 0.5 degC error on an 18 degC field scored
    nrmse 0.028 (mu 0.97) against 0.49 on a zero-mean field -- the same error, 18x flattered."""
    from oceanml3d_eval.metrics.gridded import scores_for

    cold = scores_for(*_pair(mean=0.0, error=0.5))
    warm = scores_for(*_pair(mean=18.0, error=0.5))
    assert abs(cold["rmse"] - warm["rmse"]) < 1e-9
    assert abs(cold["nrmse"] - warm["nrmse"]) < 1e-9          # anomaly-normalised: identical
    assert warm["mu"] > 0.95 and cold["mu"] < 0.6             # the RMS-based pair still drifts
    assert abs(warm["nrmse"] - 0.5) < 0.05                    # 0.5 error over a unit-std anomaly


def test_bias_correlation_and_explained_variance():
    from oceanml3d_eval.metrics.gridded import scores_for

    pred, truth = _pair(mean=18.0, error=0.5)
    s = scores_for(pred, truth)
    assert abs(s["bias"] - 0.5) < 1e-9                        # a constant offset is a bias, not noise
    assert s["anom_corr"] > 0.999                             # the anomaly is reproduced exactly
    assert s["var_explained"] > 0.999                         # ... so the variance is explained

    climatology = truth.mean("time").broadcast_like(truth)     # the reference a model must beat
    c = scores_for(climatology, truth)
    assert abs(c["var_explained"]) < 0.05 and abs(c["nrmse"] - 1.0) < 0.05
    assert abs(c["anom_corr"]) < 0.1 or np.isnan(c["anom_corr"])

    noisy = truth + np.random.default_rng(1).normal(scale=0.5, size=truth.shape)
    n = scores_for(noisy, truth)
    assert abs(n["bias"]) < 0.05 and 0.6 < n["var_explained"] < 0.85


def test_scores_are_weighted_by_cell_area():
    """An error at 60 degN covers half the area of the same error at the equator."""
    import pandas as pd

    from oceanml3d_eval.metrics.gridded import scores_for

    time = pd.date_range("2019-01-01", periods=3)
    lat, lon = np.linspace(0, 60, 7), np.linspace(-64, -56, 7)
    truth = xr.DataArray(np.zeros((3, 7, 7)), dims=("time", "lat", "lon"),
                         coords={"time": time, "lat": lat, "lon": lon})
    north, south = np.zeros((3, 7, 7)), np.zeros((3, 7, 7))
    north[:, -1, :] = 1.0                                      # wrong at 60 degN
    south[:, 0, :] = 1.0                                       # wrong at the equator
    mk = lambda a: truth + xr.DataArray(a, dims=truth.dims, coords=truth.coords)  # noqa: E731
    assert scores_for(mk(south), truth)["rmse"] > scores_for(mk(north), truth)["rmse"] > 0
    plain = [scores_for(mk(a), truth, weighted=False)["rmse"] for a in (north, south)]
    assert abs(plain[0] - plain[1]) < 1e-12                    # unweighted: position does not matter


def test_gridded_metric_exposes_every_score(products):
    from oceanml3d_eval.metrics.gridded import GriddedRMSE

    truth = open_product(products["truth"])
    g = GriddedRMSE().compute(open_product(products["noisy"]), truth, Region.get("GulfStream"),
                              "2019-01-01", "2019-01-12")
    for score in ("rmse", "bias", "nrmse", "anom_corr", "var_explained", "mu", "sigma"):
        assert f"{score}_u" in g.scores and np.isfinite(g.scores[f"{score}_u"]), score
    assert g.meta["area_weighted"] is True


# --- nothing is scored silently -------------------------------------------------------------------

def test_a_benchmark_whose_variables_the_product_lacks_stops_the_run(products, tmp_path):
    """The uo_d00 vs u_d00 case: the run used to score whatever else the product contained."""
    import pytest
    import yaml

    from oceanml3d_eval.benchmarks.runner import matching_variables
    from oceanml3d_eval.benchmarks.spec import BenchmarkSpec
    from oceanml3d_eval.product import ProductSpec, open_product

    spec = ProductSpec.load(products["truth"])                 # declares u and v
    bench_doc = {"name": "b", "period": ["2019-01-01", "2019-01-12"], "variables": ["uo", "vo"],
                 "references": {"truth": {"kind": "gridded", "path": str(products["truth"])}},
                 "regions": ["GulfStream"], "metrics": [{"name": "gridded_rmse", "reference": "truth"}]}
    path = tmp_path / "b.yaml"
    path.write_text(yaml.safe_dump(bench_doc))
    bench = BenchmarkSpec.load(path)
    with pytest.raises(ValueError, match="has none of the 2 variables"):
        matching_variables(bench, spec)

    bench_doc["variables"] = ["u", "vo"]                        # partial: keeps going, says so
    path.write_text(yaml.safe_dump(bench_doc))
    assert matching_variables(BenchmarkSpec.load(path), spec) == ["u"]

    with pytest.raises(ValueError, match="empty variable list"):
        open_product(spec, variables=[])
    with pytest.raises(KeyError, match="no variable"):
        open_product(spec, variables=["u", "nope"])


def test_a_truth_on_another_grid_is_refused_unless_asked_for():
    import pytest

    from oceanml3d_eval.metrics.gridded import align, check_same_grid

    fine, _ = _pair(n=17)                                       # 0.5 deg steps
    coarse = fine.isel(lat=slice(None, None, 4), lon=slice(None, None, 4))    # 2 deg steps
    check_same_grid(fine.to_dataset(name="u"), fine.to_dataset(name="u"))     # same grid: fine
    with pytest.raises(ValueError, match="not on the product's grid"):
        align(fine.to_dataset(name="u"), coarse.to_dataset(name="u"), ["u"])
    p, t = align(fine.to_dataset(name="u"), coarse.to_dataset(name="u"), ["u"], regrid=True)
    assert t.u.shape == p.u.shape                               # opt-in: interpolated on purpose


def test_a_truth_from_another_period_is_refused():
    import pandas as pd
    import pytest

    from oceanml3d_eval.metrics.gridded import align

    pred, truth = _pair()
    other = truth.assign_coords(time=truth.time + pd.Timedelta(days=400))
    with pytest.raises(ValueError, match="do not cover the same period"):
        align(pred.to_dataset(name="u"), other.to_dataset(name="u"), ["u"])


def test_an_effective_resolution_that_cannot_be_computed_says_why():
    from oceanml3d_eval.metrics.spectral import effective_resolution, psd_lon

    k = np.linspace(0.001, 0.1, 50)
    report: dict = {}
    assert np.isnan(effective_resolution(k, 0.01 * np.ones_like(k), np.ones_like(k), report))
    assert "below half the signal" in report["eff_resolution_nan_reason"]

    report = {}
    assert np.isnan(effective_resolution(np.array([]), np.array([]), np.array([]), report))
    assert "no usable row" in report["eff_resolution_nan_reason"]

    da, _ = _pair(days=2, n=6)
    da = da.where(da.lat < da.lat[-1])                           # one latitude entirely masked (land)
    report = {}
    psd_lon(da, report)
    assert report["rows_used"] == 10 and report["rows_total"] == 12
