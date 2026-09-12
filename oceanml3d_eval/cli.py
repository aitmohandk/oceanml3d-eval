"""``oceanml3d-eval`` CLI.

    oceanml3d-eval list
    oceanml3d-eval run  --benchmark surface_currents_15m --product outputs/.../product/product.yaml
    oceanml3d-eval run  --benchmark surface_currents_15m --baselines           # DUACS, GlobCurrent, ...
    oceanml3d-eval report --benchmark surface_currents_15m [--region GulfStream] [--metric eulerian_drifters]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from oceanml3d_eval.benchmarks import BenchmarkSpec, available_benchmarks, run_benchmark
from oceanml3d_eval.metrics import list_metrics
from oceanml3d_eval.regions import available as available_regions
from oceanml3d_eval.report import collect, depth_profile, leaderboard, to_markdown


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="oceanml3d-eval")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list benchmarks, metrics and regions")

    r = sub.add_parser("run", help="evaluate one product (or the benchmark baselines)")
    r.add_argument("--benchmark", "-b", required=True)
    r.add_argument("--product", "-p", action="append", default=[], help="product.yaml (repeatable)")
    r.add_argument("--baselines", action="store_true", help="also evaluate the benchmark baselines")
    r.add_argument("--region", action="append", help="restrict to region(s)")
    r.add_argument("--metric", action="append", help="restrict to metric(s)")
    r.add_argument("--out", default="results")
    r.add_argument("--force", action="store_true")

    rep = sub.add_parser("report", help="leaderboard from cached results")
    rep.add_argument("--benchmark", "-b", required=True)
    rep.add_argument("--out", default="results")
    rep.add_argument("--region")
    rep.add_argument("--metric")
    rep.add_argument("--markdown", help="write the table to this file")
    rep.add_argument("--depth-profile", metavar="SCORE", help="skill-vs-depth table for multi-level variables (e.g. nrmse)")

    bl = sub.add_parser("baseline", help="freeze / verify reference scores (non-regression gate)")
    bl.add_argument("--benchmark", "-b", required=True)
    bl.add_argument("--product", "-p", required=True)
    bl.add_argument("--save", action="store_true")
    bl.add_argument("--check", action="store_true")
    bl.add_argument("--file", help="baseline JSON (default baselines/<benchmark>__<product>.json)")
    bl.add_argument("--out", default="results")
    bl.add_argument("--rtol", type=float, default=1e-3)

    sp = sub.add_parser("split", help="yearly NetCDF/Zarr -> daily product files + manifest")
    sp.add_argument("--input", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--name", required=True)
    sp.add_argument("--var", action="append", required=True, help="canonical=source, e.g. u=ugos (repeatable)")
    sp.add_argument("--depth-m", type=float)
    sp.add_argument("--depth-index", type=int)

    pl = sub.add_parser("plot", help="figures from cached results")
    pl.add_argument("--benchmark", "-b", required=True)
    pl.add_argument("--out", default="results")
    pl.add_argument("--depth-profile", metavar="SCORE")
    pl.add_argument("--figure", default="figure.png")

    a = p.parse_args(argv)
    if a.cmd == "split":
        from oceanml3d_eval.split import split_to_product

        variables = dict(v.split("=", 1) for v in a.var)
        print(split_to_product(a.input, a.out, a.name, variables, a.depth_m, a.depth_index))
        return
    if a.cmd == "baseline":
        from oceanml3d_eval.product import ProductSpec
        from oceanml3d_eval.regression import BASELINE_DIR, check_baseline, save_baseline

        bench = BenchmarkSpec.load(a.benchmark)
        spec = ProductSpec.load(a.product)
        df = run_benchmark(bench, spec, a.out)
        path = Path(a.file) if a.file else BASELINE_DIR / f"{bench.name}__{spec.name}.json"
        if a.save:
            print(f"baseline written to {save_baseline(df, path, {'benchmark': bench.name, 'product': spec.name, 'period': [bench.first, bench.last]})}")
        if a.check or not a.save:
            check_baseline(df, path, a.rtol)
            print(f"scores reproduce the baseline {path} (rtol={a.rtol})")
        return
    if a.cmd == "list":
        print("benchmarks:", ", ".join(available_benchmarks()))
        print("metrics:   ", ", ".join(list_metrics()))
        print("regions:   ", ", ".join(available_regions()))
        return
    bench = BenchmarkSpec.load(a.benchmark)
    if a.cmd == "run":
        products = list(a.product) + (list(bench.baselines.values()) if a.baselines else [])
        if not products:
            p.error("give --product and/or --baselines")
        for prod in products:
            run_benchmark(bench, prod, a.out, a.region, a.metric, a.force)
        print(to_markdown(leaderboard(collect(a.out, bench), bench.leaderboard)))
    elif a.cmd == "plot":
        from oceanml3d_eval import plots

        df = collect(a.out, bench)
        if a.depth_profile:
            plots.depth_profile(depth_profile(df, a.depth_profile), a.depth_profile, out=a.figure)
        else:
            maps = sorted((Path(a.out) / bench.name).glob("*/*__rmse_map.nc"))
            if maps:
                plots.rmse_map(maps[0], bench.variables[0], out=a.figure)
        print(f"figure written to {a.figure}")
    elif a.cmd == "report":
        df = collect(a.out, bench)
        if a.depth_profile:
            prof = depth_profile(df, a.depth_profile)
            table = prof.pivot_table(index=["variable", "depth_index"], columns="product", values=a.depth_profile) if not prof.empty else prof
        else:
            table = leaderboard(df, bench.leaderboard, a.region, a.metric)
        md = to_markdown(table)
        print(md)
        if a.markdown:
            Path(a.markdown).write_text(md)


if __name__ == "__main__":
    main()
