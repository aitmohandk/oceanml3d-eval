import numpy as np

from oceanml3d_eval.product import ProductSpec, open_product
from oceanml3d_eval.regions import Region, available


def test_open_product(products):
    spec = ProductSpec.load(products["truth"])
    ds = open_product(spec, "2019-01-03", "2019-01-05")
    assert ds.sizes["time"] == 3 and set(ds.data_vars) == {"u", "v"}
    assert ds.attrs["depth_m"] == 15


def test_legacy_nosc_json(tmp_path, products):
    spec = ProductSpec.load(products["truth"])
    j = tmp_path / "legacy.json"
    import json
    j.write_text(json.dumps({"data_type": "x", "label": "x", "path": spec.path, "pattern": "truth",
                             "match": r"truth_(\d{4})-(\d{2})-(\d{2}).nc", "varu": "u", "varv": "v",
                             "nlon": "lon", "nlat": "lat", "time_coverage_hours": 24}))
    s2 = ProductSpec.load(j)
    assert s2.variables == {"u": "u", "v": "v"} and len(s2.files()) == 12


def test_regions():
    assert "GulfStream" in available()
    gs = Region.get("GulfStream")
    m = gs.mask(np.array([40.0, 30.0]), np.array([-60.0, -60.0]))
    assert m.tolist() == [True, False]
