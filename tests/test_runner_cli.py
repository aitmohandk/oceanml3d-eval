from oceanml3d_eval.benchmarks import BenchmarkSpec, run_benchmark
from oceanml3d_eval.cli import main
from oceanml3d_eval.report import collect, leaderboard


def test_run_benchmark_and_report(benchmark_file, products, tmp_path, capsys):
    bench = BenchmarkSpec.load(benchmark_file)
    df = run_benchmark(bench, products["noisy"], tmp_path)
    assert set(df.metric) == {"eulerian_drifters", "gridded_rmse", "spectral_score"}
    main(["run", "-b", str(benchmark_file), "-p", str(products["truth"]), "--baselines", "--out", str(tmp_path)])
    table = leaderboard(collect(tmp_path, bench), bench.leaderboard, metric="gridded_rmse")
    assert list(table.index) == ["truth", "noisy", "worse"]          # sorted by rmse_u
    main(["report", "-b", str(benchmark_file), "--out", str(tmp_path), "--markdown", str(tmp_path / "lb.md")])
    assert (tmp_path / "lb.md").exists()
    main(["list"])
    assert "surface_currents_15m" in capsys.readouterr().out
