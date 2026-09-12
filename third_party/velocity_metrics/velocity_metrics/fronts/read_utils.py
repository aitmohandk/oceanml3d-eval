# vim: ts=4:sts=4:sw=4

# @date 2022-10-06
# @author lucile.gaultier@oceandatalab.com

# Copyright (C) 2020-2023 OceanDataLab
# This file is part of fronts_detection

# fronts_detection is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# fronts_detection is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with fronts_detection.  If not, see <http://www.gnu.org/licenses/>.


import datetime
import cftime
from scipy import ndimage
from typing import Tuple
import velocity_metrics.utils.constant as const
import numpy


def cftime2datetime(d) -> datetime.datetime:

    if isinstance(d, datetime.datetime):
        return d
    if isinstance(d, cftime.DatetimeNoLeap):
        return datetime.datetime(d.year, d.month, d.day, d.hour, d.minute,
                                 d.second)
    elif isinstance(d, cftime.DatetimeGregorian):
        return datetime.datetime(d.year, d.month, d.day, d.hour, d.minute,
                                 d.second)
    else:
        return None


def get_gradient(sst: numpy.ndarray, lon2d: numpy.ndarray,
                 lat2d: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
    # Get x-gradient in "sx"
    gc_row = ndimage.sobel(sst, axis=0, mode='nearest')

    # Get y-gradient in "sy"
    gc_col = ndimage.sobel(sst, axis=1, mode='nearest')
    # Divide by 4 to rescale the Sobel Kernel : | -1 0 1 |
    #                                           | -2 0 2 |
    #                                           | -1 0 1 |
    gc_col = gc_col / 4
    gc_row = gc_row / 4

    # Get square root of sum of squares
    gc_lon, gc_lat = rescale_gradient(gc_row, gc_col, lon2d, lat2d)
    return gc_lon, gc_lat


# @nb.njit(cache=True, nogil=True)
def compute_gradient(image: numpy.ndarray,
                     threshold: float) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Computes the 2D gradient of an image (central difference)"""
    grad_x = numpy.zeros(numpy.shape(image), dtype=numpy.float64)
    grad_y = numpy.zeros(numpy.shape(image), dtype=numpy.float64)
    grad_x[:, 1:-1] = (image[:, 2:] - image[:, :-2]) / 2
    grad_x[:, 0] = image[:, 1] - image[:, 0]
    grad_x[:, -1] = image[:, -1] - image[:, -2]
    grad_y[1: -1, :] = (image[2:, :] - image[:-2, :]) / 2
    grad_y[0, :] = image[1, :] - image[0, :]
    grad_y[-1, :] = image[-1, :] - image[-2, :]

    # Thresholding the gradient to temp
    grad_x[grad_x > threshold] = threshold
    grad_x[grad_x < -threshold] = -threshold
    grad_y[grad_y > threshold] = threshold
    grad_y[grad_y < -threshold] = -threshold
    grad_x = numpy.ma.masked_array(grad_x, image.mask)
    grad_x = numpy.ma.masked_array(grad_x, image.mask)
    return grad_x, grad_y


# @nb.njit(cache=True, nogil=True)
def rescale_gradient(g_row: numpy.ndarray, g_col: numpy.ndarray,
                     lon_img: numpy.ndarray, lat_img: numpy.ndarray
                     ) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Turns (row,col) gradient into degrees/km along lon and lat axes. The
    (row,col) coordinates are calculated thanks to the function
    compute_gradient above or with a sobel filter."""
    # Init
    deg_to_km = const.deg2km
    shape_lon = numpy.shape(lon_img)
    n_g_row = numpy.zeros(shape_lon, dtype=numpy.float64)
    # km scaled gradient
    n_g_col = numpy.zeros(shape_lon, dtype=numpy.float64)
    # n_lon_img = numpy.zeros(shape_lon, dtype=numpy.float64)
    # km scaled gradient
    # n_lat_img = numpy.zeros(shape_lon, dtype=numpy.float64)
    theta = numpy.zeros(shape_lon, dtype=numpy.float64)
    # Angle between (row,col)
    # theta2 = numpy.zeros(shape_lon, dtype=numpy.float64)
    # and (lon,lat) coordinate

    # Compute the gradient along row,col axes.
    coslat = numpy.cos(numpy.deg2rad(lat_img))
    # gradient row
    _dist = ((lat_img[2:, :] - lat_img[:-2, :])**2
             + (coslat[1:-1, :] * (lon_img[2:, :] - lon_img[:-2, :]))**2)
    n_g_row[1:-1, :] = g_row[1:-1, :] / (deg_to_km * numpy.sqrt(_dist))

    # gradient first row
    _dist = ((lat_img[1, :] - lat_img[0, :])**2
             + (coslat[0, :] * (lon_img[1, :] - lon_img[0, :]))**2)
    n_g_row[0, :] = g_row[0, :] / (deg_to_km * numpy.sqrt(_dist))

    # gradient last row
    _dist = ((lat_img[-1, :] - lat_img[-2, :])**2
             + (coslat[-1, :] * (lon_img[-1, :] - lon_img[-2, :]))**2)
    n_g_row[-1, :] = g_row[-1, :] / (deg_to_km * numpy.sqrt(_dist))

    # gradient col
    _dist = ((lat_img[:, 2:] - lat_img[:, :-2])**2
             + (coslat[:, 1:-1] * (lon_img[:, 2:] - lon_img[:, :-2]))**2)
    n_g_col[:, 1:-1] = g_col[:, 1:-1] / (deg_to_km * numpy.sqrt(_dist))

    # gradient first col
    _dist = ((lat_img[:, 1] - lat_img[:, 0])**2
             + (coslat[:, 0] * (lon_img[:, 1] - lon_img[:, 0]))**2)
    n_g_col[:, 0] = g_col[:, 0] / (deg_to_km * numpy.sqrt(_dist))

    # gradient last col
    _dist = ((lat_img[:, -1] - lat_img[:, -2])**2
             + (coslat[:, -1] * (lon_img[:, -1] - lon_img[:, -2]))**2)
    n_g_col[:, -1] = g_col[:, -1] / (deg_to_km * numpy.sqrt(_dist))

    # Theta is the angle between the row axis, and the local lat axis. This
    # angle is not the same everywhere on the map. It must be computed in order
    # to get the gradient along lon and lat axes from the row/col gradient,
    # thanks to a rotation.

    # The precision on theta may be better if distances are computed along cols
    # instead of rows.

    # theta col
    theta[:, 1:-1] = numpy.arctan((lat_img[:, 2:] - lat_img[:, :-2])
                                  / (coslat[:, 1:-1]
                                  * (lon_img[:, 2:] - lon_img[:, :-2])))
    # theta first col
    theta[:, 0] = numpy.arctan((lat_img[:, 1] - lat_img[:, 0]) / (coslat[:, 0]
                               * (lon_img[:, 1] - lon_img[:, 0])))
    # theta last col
    theta[:, -1] = numpy.arctan((lat_img[:,  -1] - lat_img[:, -2])
                                / (coslat[:, -1]
                                * (lon_img[:, -1] - lon_img[:, -2])))
    u = -numpy.sin(theta) * n_g_row - numpy.cos(theta) * n_g_col
    v = numpy.cos(theta) * n_g_row - numpy.sin(theta) * n_g_col
    return u, v
