"""Evaluation regions (bounding box + optional polygon), loaded from ``regions/*.json``.
The JSON format is the one used by NOSC / velocity_metrics (``lllon, urlon, lllat, urlat, coords``)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

REGIONS_DIR = Path(__file__).resolve().parent.parent / "regions"


@dataclass(frozen=True)
class Region:
    name: str
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    polygon: tuple[tuple[float, float], ...] | None = None

    @classmethod
    def load(cls, path: str | Path) -> Region:
        d = json.loads(Path(path).read_text())
        poly = tuple(tuple(p) for p in d["coords"]) if d.get("coords") else None
        return cls(d["name"].strip(), d["lllon"], d["urlon"], d["lllat"], d["urlat"], poly)

    @classmethod
    def get(cls, name: str, regions_dir: Path = REGIONS_DIR) -> Region:
        path = Path(name) if str(name).endswith(".json") else regions_dir / f"region_{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"region '{name}' not found ({path}); available: {available(regions_dir)}")
        return cls.load(path)

    def bbox_mask(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        return (lat >= self.lat_min) & (lat <= self.lat_max) & (lon >= self.lon_min) & (lon <= self.lon_max)

    def mask(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Boolean mask for broadcastable ``lat``/``lon`` arrays (points or 2D grids)."""
        lat, lon = np.broadcast_arrays(np.asarray(lat), np.asarray(lon))
        m = self.bbox_mask(lat, lon)
        if self.polygon is None:
            return m
        from shapely import contains_xy
        from shapely.geometry import Polygon

        inside = contains_xy(Polygon(self.polygon), lon.ravel(), lat.ravel()).reshape(lat.shape)
        return m & inside

    def bbox_subset(self, ds: xr.Dataset) -> xr.Dataset:
        return ds.sel(lat=slice(self.lat_min, self.lat_max), lon=slice(self.lon_min, self.lon_max))

    def subset(self, ds: xr.Dataset) -> xr.Dataset:
        """Crop a gridded dataset to the bbox and mask cells outside the polygon."""
        sub = self.bbox_subset(ds)
        if self.polygon is None:
            return sub
        lo, la = np.meshgrid(sub.lon.values, sub.lat.values)
        return sub.where(xr.DataArray(self.mask(la, lo), dims=("lat", "lon")))


def available(regions_dir: Path = REGIONS_DIR) -> list[str]:
    return sorted(p.stem.removeprefix("region_") for p in regions_dir.glob("region_*.json"))
