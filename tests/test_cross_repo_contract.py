"""Contract test: a product written by oceanml3d-core must be readable here, unchanged.

Skipped when oceanml3d-core is not installed (this repo does not depend on it); the end-to-end CI
workflow installs both and runs the real chain.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from oceanml3d_eval.product import open_product


def test_core_written_product_is_readable(tmp_path):
    core_export = pytest.importorskip("oceanml3d.inference.export", reason="oceanml3d-core not installed")
    from oceanml3d.variables import VariableSet

    variables = VariableSet.from_config({
        "ssh_obs": {"source": "p", "role": "input"},
        "u_drifter": {"source": "t", "role": "target"},
        "v_drifter": {"source": "t", "role": "target"},
        "thetao": {"source": "t", "role": "target", "depth_indices": [0, 4], "group": "temperature", "units": "degC"}})
    time = pd.date_range("2019-01-01", periods=3)
    names = [s.name for s in variables.targets]
    field = xr.DataArray(np.random.rand(len(names), 3, 4, 5).astype("f4"), dims=("channel", "time", "lat", "lon"),
                         coords={"channel": names, "time": time, "lat": np.arange(4), "lon": np.arange(5)})
    manifest = core_export.write_product(field, variables, tmp_path, "xp", depth_m=15, attrs={"model": "nosc_unet"})

    ds = open_product(manifest, "2019-01-01", "2019-01-02")
    assert set(ds.data_vars) == {"u", "v", "thetao_d00", "thetao_d04"}
    assert ds.sizes["time"] == 2 and ds.attrs["depth_m"] == 15
    assert ds.u.attrs["standard_name"] == "eastward_sea_water_velocity"


def test_contract_module_matches_core():
    """The two copies of product_contract.py must stay byte-identical."""
    import hashlib
    from pathlib import Path

    core = pytest.importorskip("oceanml3d.inference.product_contract", reason="oceanml3d-core not installed")
    import oceanml3d_eval.product_contract as mine

    h = [hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (core, mine)]
    assert h[0] == h[1], "product_contract.py differs between oceanml3d-core and oceanml3d-eval"
