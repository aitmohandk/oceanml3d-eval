import numpy as np
import pandas as pd
import pytest
import xarray as xr

from oceanml3d_eval.cli import main
from oceanml3d_eval.product import open_product


def test_split_and_open(truth_field, tmp_path):
    src = tmp_path / "yearly.nc"
    truth_field.rename({"u": "ugos", "v": "vgos", "lat": "latitude", "lon": "longitude"}).to_netcdf(src)
    main(["split", "--input", str(src), "--out", str(tmp_path / "prod"), "--name", "duacs", "--var", "u=ugos", "--var", "v=vgos", "--depth-m", "15"])
    ds = open_product(tmp_path / "prod" / "product.yaml", "2019-01-02", "2019-01-04")
    assert ds.sizes["time"] == 3 and set(ds.data_vars) == {"u", "v"} and "lat" in ds.dims


def test_plots(tmp_path):
    pytest.importorskip("matplotlib")
    from oceanml3d_eval import plots

    diag = xr.Dataset({"rmse_u": (("lat", "lon"), np.random.rand(4, 5))}, coords={"lat": np.arange(4), "lon": np.arange(5)})
    diag.to_netcdf(tmp_path / "d__rmse_map.nc")
    plots.rmse_map(tmp_path / "d__rmse_map.nc", "u", out=tmp_path / "m.png")
    prof = pd.DataFrame({"product": ["a", "a", "b", "b"], "variable": "thetao", "depth_index": [0, 2, 0, 2], "nrmse": [0.1, 0.2, 0.15, 0.3]})
    plots.depth_profile(prof, "nrmse", {0: 0.5, 2: 2.6}, out=tmp_path / "p.png")
    assert (tmp_path / "m.png").exists() and (tmp_path / "p.png").exists()


# --- 3D references: one variable per level -------------------------------------------------------

def _depth_source(tmp_path, n_levels=6, days=3):
    import numpy as np
    import pandas as pd
    import xarray as xr

    time = pd.date_range("2019-01-01", periods=days)
    lat, lon, depth = np.arange(30.0, 34.0), np.arange(-62.0, -58.0), np.array([0.49, 5.0, 15.0, 30.0, 50.0, 100.0])[:n_levels]
    shape = (days, n_levels, lat.size, lon.size)
    field = np.arange(np.prod(shape), dtype="f4").reshape(shape)
    ds = xr.Dataset({"thetao": (("time", "depth", "latitude", "longitude"), field),
                     "uo": (("time", "depth", "latitude", "longitude"), -field),
                     "zos": (("time", "latitude", "longitude"), field[:, 0])},
                    coords={"time": time, "depth": depth, "latitude": lat, "longitude": lon})
    src = tmp_path / "truth.nc"
    ds.to_netcdf(src)
    return src, ds


def test_split_writes_one_variable_per_level(tmp_path):
    import xarray as xr

    from oceanml3d_eval.product import ProductSpec, open_product
    from oceanml3d_eval.split import split_to_product

    src, ref = _depth_source(tmp_path)
    manifest = split_to_product(src, tmp_path / "out", "truth3d",
                                {"ssh": "zos", "thetao": "thetao", "u": "uo"}, depth_indices=[0, 2, 5])
    spec = ProductSpec.load(manifest)
    assert set(spec.variables) == {"ssh", "thetao_d00", "thetao_d02", "thetao_d05",
                                   "u_d00", "u_d02", "u_d05"}
    ds = open_product(spec)
    assert ds.sizes["time"] == 3 and "depth" not in ds.dims
    # the values are the level, not level 0 repeated -- and u comes from uo
    xr.testing.assert_allclose(ds.thetao_d02.transpose("time", "lat", "lon"),
                               ref.thetao.isel(depth=2, drop=True).rename({"latitude": "lat", "longitude": "lon"}))
    xr.testing.assert_allclose(ds.u_d05.transpose("time", "lat", "lon"),
                               ref.uo.isel(depth=5, drop=True).rename({"latitude": "lat", "longitude": "lon"}))
    import yaml
    doc = yaml.safe_load(manifest.read_text())
    assert doc["variables"]["thetao_d05"] == {"name": "thetao_d05", "depth_index": 5, "depth_m": 100.0}
    assert doc["variables"]["ssh"] == {"name": "ssh"}          # no depth axis: no suffix


def test_split_refuses_a_level_that_does_not_exist_and_an_unknown_variable(tmp_path):
    import pytest

    from oceanml3d_eval.split import split_to_product

    src, _ = _depth_source(tmp_path, n_levels=3)
    with pytest.raises(IndexError, match="out of range"):
        split_to_product(src, tmp_path / "a", "x", {"thetao": "thetao"}, depth_indices=[0, 7])
    with pytest.raises(KeyError, match="no variable"):
        split_to_product(src, tmp_path / "b", "x", {"so": "so"}, depth_indices=[0])


def test_single_level_split_is_unchanged(tmp_path):
    from oceanml3d_eval.product import ProductSpec
    from oceanml3d_eval.split import split_to_product

    src, _ = _depth_source(tmp_path)
    spec = ProductSpec.load(split_to_product(src, tmp_path / "s", "surf", {"thetao": "thetao"},
                                             depth_m=0.49, depth_index=0))
    assert set(spec.variables) == {"thetao"} and spec.depth_m == 0.49


def test_parse_indices():
    from oceanml3d_eval.split import parse_indices

    assert parse_indices("0,2,4") == [0, 2, 4]
    assert parse_indices("0,2,4,6,8,10-25") == [0, 2, 4, 6, 8] + list(range(10, 26))
    assert parse_indices("all", 3) == [0, 1, 2]


def test_the_osse3d_benchmark_and_its_truth_manifest_agree():
    """The benchmark asks for 64 variables; the shipped truth manifest must offer exactly those."""
    import os
    from pathlib import Path

    import yaml

    from oceanml3d_eval.benchmarks.spec import BENCHMARKS_DIR
    from oceanml3d_eval.product_contract import validate_manifest

    bench = yaml.safe_load((BENCHMARKS_DIR / "osse3d_gs21.yaml").read_text())
    path = Path(__file__).resolve().parents[1] / "products/glorys_gs21_truth.yaml"
    os.environ.setdefault("OCEANML3D_DATA", "/data")
    truth = yaml.safe_load(os.path.expandvars(path.read_text()))
    assert validate_manifest(truth, strict=True) == []
    assert set(bench["variables"]) == set(truth["variables"]), (
        set(bench["variables"]) ^ set(truth["variables"]))
    for name, spec in truth["variables"].items():
        if "_d" in name:
            assert spec["depth_index"] == int(name.rsplit("_d", 1)[1])
