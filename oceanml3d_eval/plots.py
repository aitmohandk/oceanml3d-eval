"""Figures from cached benchmark results (matplotlib; cartopy optional)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def rmse_map(diag_nc: str | Path, var: str = "u", out: str | Path | None = None, vmax: float | None = None):
    plt = _plt()
    ds = xr.open_dataset(diag_nc)
    fig, ax = plt.subplots(figsize=(8, 4))
    ds[f"rmse_{var}"].plot(ax=ax, vmax=vmax, cmap="magma_r")
    ax.set_title(f"RMSE {var} — {Path(diag_nc).stem}")
    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight")
    return fig


def psd(diag_ncs: dict[str, str | Path], var: str = "u", out: str | Path | None = None):
    """``diag_ncs``: label -> psd_<var> NetCDF written by spectral_score."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, f in diag_ncs.items():
        d = xr.open_dataset(f)
        ax.loglog(1 / d.k, d.psd_err / d.psd_ref, label=label)
    ax.axhline(0.5, color="k", ls="--", lw=0.8)
    ax.set_xlabel("wavelength [km]")
    ax.set_ylabel("PSD(err) / PSD(ref)")
    ax.invert_xaxis()
    ax.legend()
    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight")
    return fig


def depth_profile(prof: pd.DataFrame, score: str = "nrmse", depth_values: dict[int, float] | None = None,
                  out: str | Path | None = None):
    """Skill-vs-depth curves, one line per product and variable (``prof`` from report.depth_profile)."""
    plt = _plt()
    variables = sorted(prof.variable.unique())
    fig, axes = plt.subplots(1, len(variables), figsize=(4 * len(variables), 5), squeeze=False)
    for ax, var in zip(axes[0], variables, strict=True):
        for product, g in prof[prof.variable == var].groupby("product"):
            y = [depth_values.get(i, i) for i in g.depth_index] if depth_values else g.depth_index
            ax.plot(g[score], y, marker="o", label=product)
        ax.invert_yaxis()
        ax.set_title(var)
        ax.set_xlabel(score)
        ax.set_ylabel("depth [m]" if depth_values else "depth index")
        ax.grid(alpha=0.3)
    axes[0][0].legend()
    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight")
    return fig
