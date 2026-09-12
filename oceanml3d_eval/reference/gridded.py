"""Gridded truth (OSSE: model output regarded as truth, or a reference L4 product)."""
from __future__ import annotations

import xarray as xr

from oceanml3d_eval.product import ProductSpec, open_product
from oceanml3d_eval.reference.base import register_reference


@register_reference("gridded")
def read_gridded(path: str, first: str | None = None, last: str | None = None,
                 variables: list[str] | None = None, **_) -> xr.Dataset:
    """``path`` is a product manifest (yaml/json) so that external products double as references."""
    return open_product(ProductSpec.load(path), first, last, variables)
