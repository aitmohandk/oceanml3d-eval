import json

import pandas as pd
import pytest

from oceanml3d_eval.regression import compare, save_baseline


def _df(rmse=0.1, extra=None):
    row = {"product": "m", "benchmark": "b", "metric": "gridded_rmse", "region": "GS", "rmse_u": rmse, "n_obs": 100}
    row.update(extra or {})
    return pd.DataFrame([row])


def test_identical_run_reproduces(tmp_path):
    p = save_baseline(_df(), tmp_path / "b.json")
    assert compare(_df(), p) == []
    assert json.loads(p.read_text())["meta"] == {}


def test_small_drift_within_tolerance(tmp_path):
    p = save_baseline(_df(0.1), tmp_path / "b.json")
    assert compare(_df(0.1000001), p, rtol=1e-3) == []
    problems = compare(_df(0.11), p, rtol=1e-3)
    assert len(problems) == 1 and "+10.00%" in problems[0]


def test_missing_and_new_entries_reported(tmp_path):
    p = save_baseline(_df(extra={"corr_u": 0.9}), tmp_path / "b.json")
    assert any("corr_u: missing" in x for x in compare(_df(), p))
    assert any("not in the baseline" in x for x in compare(pd.concat([_df(), _df().assign(region="Agulhas")]), p))


def test_nan_handling(tmp_path):
    p = save_baseline(_df(float("nan")), tmp_path / "b.json")
    assert compare(_df(float("nan")), p) == []
    assert compare(_df(0.1), p) != []


def test_cli_gate(products, benchmark_file, tmp_path, monkeypatch):
    from oceanml3d_eval.cli import main

    monkeypatch.chdir(tmp_path)
    main(["baseline", "-b", str(benchmark_file), "-p", str(products["noisy"]), "--save", "--out", str(tmp_path / "r")])
    assert (tmp_path / "baselines").glob("*.json")
    main(["baseline", "-b", str(benchmark_file), "-p", str(products["noisy"]), "--check", "--out", str(tmp_path / "r")])
    ref = next((tmp_path / "baselines").glob("*.json"))
    doc = json.loads(ref.read_text())
    key = next(iter(doc["scores"]))
    doc["scores"][key]["rmse_u"] = 42.0
    ref.write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match="differ from the baseline"):
        main(["baseline", "-b", str(benchmark_file), "-p", str(products["noisy"]), "--check", "--out", str(tmp_path / "r")])
