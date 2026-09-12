import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
R = 6371e3


@pytest.fixture(scope="session")
def truth_field():
    lat = np.arange(20, 40.25, 0.25)
    lon = np.arange(-70, -39.75, 0.25)
    time = pd.date_range("2019-01-01", periods=12)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    t = np.arange(len(time))[:, None, None]
    u = 0.5 * np.sin(np.deg2rad(lo) * 6 + t / 5) * np.ones_like(la)
    v = 0.3 * np.cos(np.deg2rad(la) * 6 - t / 5)
    return xr.Dataset({"u": (("time", "lat", "lon"), u.astype("f4")), "v": (("time", "lat", "lon"), v.astype("f4"))},
                      coords={"time": time, "lat": lat, "lon": lon})


def _write_product(ds, root: Path, name: str, noise: float = 0.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    daily = root / name / "daily"
    daily.mkdir(parents=True)
    for t in ds.time.values:
        d = ds.sel(time=[t]).copy()
        for v in ("u", "v"):
            d[v] = d[v] + noise * rng.standard_normal(d[v].shape).astype("f4")
        d.to_netcdf(daily / f"{name}_{pd.Timestamp(t):%Y-%m-%d}.nc")
    man = {"name": name, "path": str(daily), "pattern": f"{name}_(\\d{{4}})-(\\d{{2}})-(\\d{{2}})\\.nc",
           "variables": {"u": {"standard_name": "eastward_sea_water_velocity"}, "v": {"standard_name": "northward_sea_water_velocity"}},
           "depth_m": 15}
    p = root / name / "product.yaml"
    p.write_text(yaml.safe_dump(man))
    return p


@pytest.fixture(scope="session")
def products(truth_field, tmp_path_factory):
    root = tmp_path_factory.mktemp("products")
    return {"truth": _write_product(truth_field, root, "truth"),
            "noisy": _write_product(truth_field, root, "noisy", noise=0.1),
            "worse": _write_product(truth_field, root, "worse", noise=0.3)}


@pytest.fixture(scope="session")
def drifters(truth_field, tmp_path_factory):
    """Points sampled exactly on the truth field (product 'truth' should score ~0 RMSE)."""
    rng = np.random.default_rng(1)
    n = 400
    lat = rng.uniform(36, 43, n)
    lon = rng.uniform(-66, -46, n)
    time = pd.to_datetime(rng.choice(truth_field.time.values, n))
    pts = dict(lat=xr.DataArray(lat, dims="p"), lon=xr.DataArray(lon, dims="p"), time=xr.DataArray(time.values, dims="p"))
    s = truth_field.interp(lat=pts["lat"], lon=pts["lon"]).sel(time=pts["time"])
    df = pd.DataFrame({"date": time, "lat": lat, "lon": lon, "ums": s.u.values, "vms": s.v.values, "id": np.arange(n) % 50})
    path = tmp_path_factory.mktemp("drifters") / "drifter_2019.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def benchmark_file(products, drifters, tmp_path_factory):
    doc = {
        "name": "test_bench", "period": ["2019-01-01", "2019-01-12"], "variables": ["u", "v"],
        "references": {
            "drifters": {"kind": "drifters_points", "path": str(drifters), "options": {"columns": {"time": "date", "u": "ums", "v": "vms"}}},
            "truth": {"kind": "gridded", "path": str(products["truth"])},
        },
        "regions": ["GulfStream"],
        "metrics": [
            {"name": "eulerian_drifters", "reference": "drifters", "options": {"bin_deg": 5}},
            {"name": "gridded_rmse", "reference": "truth"},
            {"name": "spectral_score", "reference": "truth"},
        ],
        "baselines": {"worse": str(products["worse"])},
        "leaderboard": ["rmse_vec", "rmse_u", "eff_resolution_km_u"],
    }
    p = tmp_path_factory.mktemp("bench") / "test_bench.yaml"
    p.write_text(yaml.safe_dump(doc))
    return p
