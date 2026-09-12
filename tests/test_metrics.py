import numpy as np

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

