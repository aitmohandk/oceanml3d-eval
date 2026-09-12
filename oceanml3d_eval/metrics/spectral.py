"""Wavenumber spectra along longitude and the effective resolution (scale where the
error PSD reaches half the signal PSD), following the SSH data challenges."""
from __future__ import annotations

import numpy as np
import xarray as xr

from oceanml3d_eval.metrics.base import Metric, MetricResult, register_metric
from oceanml3d_eval.metrics.gridded import align, select_variables
from oceanml3d_eval.regions import Region

R_EARTH_KM = 6371.0


def psd_lon(da: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    """Mean PSD along lon (rows with NaN dropped), wavenumber in cycles/km."""
    arr = da.transpose("time", "lat", "lon").values
    rows = arr.reshape(-1, arr.shape[-1])
    rows = rows[np.isfinite(rows).all(axis=1)]
    if len(rows) == 0:
        return np.array([]), np.array([])
    lat_mean = float(np.abs(da.lat).mean())
    dlon_km = float(np.diff(da.lon).mean()) * np.pi / 180 * R_EARTH_KM * np.cos(np.deg2rad(lat_mean))
    rows = rows - rows.mean(axis=1, keepdims=True)
    rows = rows * np.hanning(rows.shape[1])
    spec = np.abs(np.fft.rfft(rows, axis=1)) ** 2
    k = np.fft.rfftfreq(rows.shape[1], d=dlon_km)
    return k[1:], spec.mean(axis=0)[1:]


def isotropic_psd(field2d: np.ndarray, dx_km: float) -> tuple[np.ndarray, np.ndarray]:
    """Radially averaged PSD of one 2D field (NaN -> 0 after mean removal); k in cycles/km."""
    f = np.nan_to_num(field2d - np.nanmean(field2d))
    ny, nx = f.shape
    spec = np.abs(np.fft.fftshift(np.fft.fft2(f))) ** 2
    ky = np.fft.fftshift(np.fft.fftfreq(ny))
    kx = np.fft.fftshift(np.fft.fftfreq(nx))
    kr = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    bins = np.linspace(0, 0.5, min(ny, nx) // 2)
    which = np.digitize(kr.ravel(), bins)
    psd = np.array([spec.ravel()[which == b].mean() if np.any(which == b) else np.nan for b in range(1, len(bins))])
    k = 0.5 * (bins[1:] + bins[:-1]) / dx_km
    return k, psd


def effective_resolution(k: np.ndarray, psd_err: np.ndarray, psd_ref: np.ndarray) -> float:
    """Wavelength (km) at which err/ref PSD ratio crosses 0.5 (NaN if never)."""
    if len(k) == 0:
        return np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = psd_err / psd_ref
    idx = np.where(ratio >= 0.5)[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    if i == 0:
        return float(1 / k[0])
    k0, k1, r0, r1 = k[i - 1], k[i], ratio[i - 1], ratio[i]
    kc = k0 + (0.5 - r0) * (k1 - k0) / (r1 - r0)
    return float(1 / kc)


@register_metric("spectral_score")
class SpectralScore(Metric):
    needs = "gridded"

    def compute(self, product: xr.Dataset, reference: xr.Dataset, region: Region, first: str, last: str) -> MetricResult:
        variables = select_variables(product, reference, self.options)
        prod = region.bbox_subset(product).sel(time=slice(first, last))  # spectra need full rows: bbox only
        prod, truth = align(prod, reference.sel(time=slice(first, last)), variables)
        res, diag = {}, {}
        isotropic = bool(self.options.get("isotropic", False))
        stride = int(self.options.get("time_stride", 5))
        for v in variables:
            if isotropic:
                p_ref, p_err = _isotropic_mean(prod[v], truth[v], stride)
                k = p_ref[0]
                p_ref, p_err = p_ref[1], p_err[1]
            else:
                k, p_ref = psd_lon(truth[v].load())
                _, p_err = psd_lon((prod[v] - truth[v]).load())
            res[f"eff_resolution_km_{v}"] = effective_resolution(k, p_err, p_ref)
            diag[f"psd_{v}"] = xr.Dataset({"psd_ref": ("k", p_ref), "psd_err": ("k", p_err)}, coords={"k": k})
        return MetricResult(res, diag, {"region": region.name})


def _isotropic_mean(prod: xr.DataArray, truth: xr.DataArray, stride: int):
    lat_mean = float(np.abs(truth.lat).mean())
    dx_km = float(np.diff(truth.lon).mean()) * np.pi / 180 * R_EARTH_KM * np.cos(np.deg2rad(lat_mean))
    err = (prod - truth).load()
    truth = truth.load()
    k = None
    ref_acc, err_acc, n = 0, 0, 0
    for i in range(0, truth.sizes["time"], stride):
        k, pr = isotropic_psd(truth.isel(time=i).values, dx_km)
        _, pe = isotropic_psd(err.isel(time=i).values, dx_km)
        ref_acc, err_acc, n = ref_acc + np.nan_to_num(pr), err_acc + np.nan_to_num(pe), n + 1
    return (k, ref_acc / max(n, 1)), (k, err_acc / max(n, 1))
