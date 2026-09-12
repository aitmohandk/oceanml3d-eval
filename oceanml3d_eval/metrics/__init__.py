"""Metric registry. A metric turns (product, reference, region, period) into scalar scores
(+ optional gridded/tabular diagnostics)."""
from oceanml3d_eval.metrics import (  # noqa: F401,E402
    ensemble,
    eulerian,
    gridded,
    lagrangian,
    spectral,
)
from oceanml3d_eval.metrics.base import (  # noqa: F401
    Metric,
    MetricResult,
    get_metric,
    list_metrics,
    register_metric,
)
