"""Leaderboards and comparison tables from cached benchmark results."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from oceanml3d_eval.benchmarks.spec import BenchmarkSpec


def collect(results_dir: str | Path, bench: BenchmarkSpec) -> pd.DataFrame:
    rows = [json.loads(p.read_text()) for p in (Path(results_dir) / bench.name).glob("*/*.json")]
    return pd.DataFrame(rows)


def leaderboard(df: pd.DataFrame, scores: list[str] | None = None, region: str | None = None,
                metric: str | None = None) -> pd.DataFrame:
    sub = df.copy()
    if region:
        sub = sub[sub.region == region]
    if metric:
        sub = sub[sub.metric == metric]
    if sub.empty:
        return sub
    cols = [c for c in (scores or []) if c in sub] or [c for c in sub.columns if c.startswith(("rmse", "var_expl", "mu", "eff_res", "sep_km"))]
    table = sub.pivot_table(index="product", columns="region", values=cols, aggfunc="first")
    return table.sort_values(table.columns[0])


def to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    try:
        return df.to_markdown(floatfmt=floatfmt)
    except ImportError:      # tabulate missing
        return df.to_string(float_format=lambda x: format(x, floatfmt))


_DEPTH = __import__("re").compile(r"^(?P<score>[a-z_]+?)_(?P<var>[A-Za-z]+)_d(?P<idx>\d{2})$")


def depth_profile(df: pd.DataFrame, score: str = "nrmse") -> pd.DataFrame:
    """Long table (product, region, variable, depth_index, value) for scores of multi-level
    variables named ``<score>_<var>_d<ii>`` (the skill-vs-depth curve of the OSSE-3D study)."""
    rows = []
    for _, r in df.iterrows():
        for col, val in r.items():
            m = _DEPTH.match(str(col))
            if m and m.group("score") == score and pd.notna(val):
                rows.append({"product": r["product"], "region": r["region"], "metric": r["metric"],
                             "variable": m.group("var"), "depth_index": int(m.group("idx")), score: float(val)})
    out = pd.DataFrame(rows)
    return out.sort_values(["product", "variable", "depth_index"]).reset_index(drop=True) if not out.empty else out
