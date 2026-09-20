"""Run every (metric, region) of a benchmark on a product and cache the results."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from oceanml3d_eval.benchmarks.spec import BenchmarkSpec
from oceanml3d_eval.metrics import get_metric
from oceanml3d_eval.product import ProductSpec, open_product
from oceanml3d_eval.regions import Region


def matching_variables(bench: BenchmarkSpec, product: ProductSpec) -> list[str]:
    """The benchmark's variables that the product actually has -- refusing the empty intersection.

    A product whose names do not match the benchmark's (``uo_d00`` where the benchmark says
    ``u_d00``) used to be scored anyway, on whatever else it contained, and the leaderboard looked
    normal. Nothing is silently skipped now: an empty intersection stops the run, a partial one is
    reported.
    """
    have = [v for v in bench.variables if v in product.variables]
    missing = [v for v in bench.variables if v not in product.variables]
    if not have:
        raise ValueError(
            f"product '{product.name}' has none of the {len(bench.variables)} variables of benchmark "
            f"'{bench.name}'.\n  benchmark wants: {bench.variables[:6]}...\n  product declares: "
            f"{sorted(product.variables)[:6]}...\n  Check the naming: the product format uses the "
            f"canonical u, v, ssh, sst, thetao, so, mld, with a _d<ii> suffix per level.")
    if missing:
        print(f"[{bench.name}] {product.name}: {len(missing)}/{len(bench.variables)} variables missing "
              f"from the product, scored on the {len(have)} shared ones (missing: {missing[:6]}"
              f"{'...' if len(missing) > 6 else ''})")
    return have


def _cache_key(bench: BenchmarkSpec, product: ProductSpec, metric: str, region: str) -> str:
    h = hashlib.md5(f"{bench.name}|{bench.first}|{bench.last}|{product.path}|{metric}|{region}".encode()).hexdigest()[:10]
    return f"{product.name}__{metric}__{region}__{h}"


def run_benchmark(bench: BenchmarkSpec, product: ProductSpec | str, out_dir: str | Path,
                  regions: list[str] | None = None, metrics: list[str] | None = None,
                  force: bool = False, save_diagnostics: bool = True) -> pd.DataFrame:
    product = product if isinstance(product, ProductSpec) else ProductSpec.load(product)
    out_dir = Path(out_dir) / bench.name / product.name
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = open_product(product, bench.first, bench.last, matching_variables(bench, product))
    refs_cache = {}
    rows = []
    for mspec in bench.metrics:
        if metrics and mspec.name not in metrics:
            continue
        metric = get_metric(mspec.name)(**mspec.options, product_spec=product)
        for rname in regions or bench.regions:
            key = _cache_key(bench, product, mspec.name, rname)
            cached = out_dir / f"{key}.json"
            if cached.exists() and not force:
                rows.append(json.loads(cached.read_text()))
                continue
            if mspec.reference not in refs_cache:
                refs_cache[mspec.reference] = bench.references[mspec.reference].load(bench.first, bench.last)
            region = Region.get(rname)
            res = metric.compute(ds, refs_cache[mspec.reference], region, bench.first, bench.last)
            row = {"product": product.name, "benchmark": bench.name, "metric": mspec.name, "region": region.name,
                   "reference": mspec.reference, **res.scores}
            cached.write_text(json.dumps(row, default=float))
            if save_diagnostics:
                for dname, diag in res.diagnostics.items():
                    p = out_dir / f"{key}__{dname}"
                    diag.to_netcdf(p.with_suffix(".nc")) if hasattr(diag, "to_netcdf") else diag.to_csv(p.with_suffix(".csv"))
            rows.append(row)
            print(f"[{bench.name}] {product.name} {mspec.name} {region.name}: "
                  + ", ".join(f"{k}={v:.4g}" for k, v in res.scores.items() if isinstance(v, float)))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "scores.csv", index=False)
    return df
