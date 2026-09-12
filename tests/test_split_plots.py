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
