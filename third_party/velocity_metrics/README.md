# velocity_metrics (vendored)

Copyright (C) 2020-2024 OceanDataLab (Lucile Gaultier) — GNU LGPL v3 or later (see the header of
every module). Vendored unchanged from `NOSC/env/velocity_metrics` so that `oceanml3d-eval` can run
the exact Lagrangian (`drifters.run_all_load_once` + `cumulative_distance.run`) and spectral
diagnostics used in the NOSC paper. Install with `pip install -e third_party/velocity_metrics`
and select it with `options: {backend: velocity_metrics}` in a benchmark metric entry.
Add the LGPL-3.0 license text as `LICENSE` before redistributing.
