# vim: ts=4:sts=4:sw=4
#
# @author lucile.gaultier@oceandatalab.com
# @date 2023-10-01
#
# Copyright (C) 2022-2024 OceanDataLab
# This file is part of velocity_metrics

# this program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# this program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
""" Useful tools to convert, make default, load python files """


import sys
from typing import Tuple
import pyproj
import os
from math import sqrt, pi
import datetime
import velocity_metrics.utils.constant as const
import numpy
import logging
logger = logging.getLogger(__name__)


def load_python_file(file_path: str):
    """
    Load a file and parse it as a Python module.
    Args:
        file_path (str): Path to the python module file
    Returns:
        Module object
    """
    if not os.path.exists(file_path):
        raise IOError('File not found: {}'.format(file_path))

    full_path = os.path.abspath(file_path)
    python_filename = os.path.basename(full_path)
    module_name, _ = os.path.splitext(python_filename)
    module_dir = os.path.dirname(full_path)
    if module_dir not in sys.path:
        sys.path.append(module_dir)

    module = __import__(module_name, globals(), locals(), [], 0)
    return module


def make_default_lagrangian(p):
    """
    Fill module object with default parameters for lagrangian functions
    Args:
        p (module): Parameter python module
    Returns
        Updated module
    """
    # Advection grid
    p.make_grid = getattr(p, 'make_grid', True)
    p.output_step = getattr(p, 'output_step', 1.)
    p.box = getattr(p, 'box', None)

    # Collocated or advected tracer
    p.list_tracer = getattr(p, 'list_tracer', None)
    p.list_grid = getattr(p, 'list_grid', None)
    p.list_num = getattr(p, 'list_num', None)
    p.tracer_filter = getattr(p, 'tracer_filter', (0, 0))

    p.vel_format = getattr(p, 'vel_format', 'regular_netcdf')
    if p.vel_format == 'regular_netcdf':
        p.name_lon = getattr(p, 'name_lon', 'lon')
        p.name_lat = getattr(p, 'name_lat', 'lat')
        p.name_u = getattr(p, 'name_u', 'u')
        p.name_v = getattr(p, 'name_v', 'v')
        # p.name_h = getattr(p, 'name_h', 'h')
    elif p.vel_format == 'nemo':
        p.name_lon = getattr(p, 'name_lon', 'nav_lon')
        p.name_lat = getattr(p, 'name_lat', 'nav_lat')
        p.name_u = getattr(p, 'name_u', 'vozocrtx')
        p.name_v = getattr(p, 'name_v', 'vomecrty')
        # p.name_h = getattr(p, 'name_h', 'sossheig')
    p.depth = getattr(p, 'depth', None)
    p.vel_filter = getattr(p, 'vel_filter', None)
    p.output_step = getattr(p, 'output_step', 1.0)
    p.stationary = getattr(p, 'stationary', True)
    p.name_h = getattr(p, 'name_h', None)
    p.subsample = getattr(p, 'subsample', 1)
    p.missing_value = getattr(p, 'missing_value', 0)

    # Advection parameters
    p.K = getattr(p, 'K', 0.)
    p.B = sqrt(2 * float(p.K)) / const.deg2km
    p.scale = getattr(p, 'scale', 1.)
    p.gamma = getattr(p, 'gamma', None)
    p.weight_part = getattr(p, 'weight_part', 1)
    p.radius_part = getattr(p, 'radius_part', 0)

    # outputs
    p.fill_value = getattr(p, 'fill_value', -1e36)
    p.save_U = getattr(p, 'save_U', False)
    p.save_V = getattr(p, 'save_V', False)
    p.save_OW = getattr(p, 'save_OW', False)
    p.save_RV = getattr(p, 'save_RV', False)
    p.save_S = getattr(p, 'save_S', False)
    p.save_traj = getattr(p, 'save_traj', False)
    p.output_dir = getattr(p, 'output_dir', './')

    # misc
    p.parallelisation = getattr(p, 'parallelisation', False)
    p.factor = 180.0 / (pi * const.Rearth)
    p.out_pattern = getattr(p, 'out_pattern', 'Lap_output')
    return None


def update_progress(progress: float, arg1: str, arg2: str) -> None:
    """
    Display progress bar in terminal
    Args:
        progress (float): percentage of progress
        arg1 (str): first message to display
        arg2 (str): second message to display
    """
    barLength = 30  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "Error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength * progress))
    if arg1 and arg2:
        _arg2 = f'{arg1}, {arg2}'
    elif arg1:
        _arg2 = arg1
    else:
        _arg2 = ''
    _arg0 = "#"*block + "-"*(barLength - block)
    _arg1 = "%.2f" % (progress*100)
    text = f"\rPercent: [{_arg0}] {_arg1}%, {_arg2}, {status}"
    sys.stdout.write(text)
    sys.stdout.flush()


def lin_1Dinterp(a: numpy.ndarray, delta: float) -> float:
    """
    Linear 1d interpolation between two points
    Args:
        a (numpy.ndarray): segment to be interpolated
        delta (float): distance for interpolation on segment
    Returns:
       interpolated point
    """
    if len(a) > 1:
        y = a[0]*(1 - delta) + a[1] * delta
    elif len(a) == 1:
        y = a[0]
    else:
        raise Exception('empty array in 1d interpolation.')
    return y


def lin_2Dinterp(a: numpy.ndarray, delta1: float, delta2: float) -> float:
    """
    Linear 2d interpolation between two points
    Args:
        a (numpy.ndarray): segment to be interpolated
        delta1 (float): distance for interpolation in first dimension
        delta2 (float): distance for interpolation in second dimension
    Returns:
        interpolated point
    """
    x = lin_1Dinterp(a[0, :], delta1)
    y = lin_1Dinterp(a[1, :], delta1)
    z = lin_1Dinterp([x, y], delta2)
    return z


def bissextile(year: int) -> int:
    """
    Detect if a year is bissextile
    Args:
       year (int): year
    Returns
       1 if year is bissextile, else 0
    """
    biss = 0
    if numpy.mod(year, 400) == 0:
        biss = 1
    if (numpy.mod(year, 4) == 0 and numpy.mod(year, 100) != 0):
        biss = 1
    return biss


def dinm(year: int, month: int) -> int:
    """
    Compute number of days in a given month
    Args:
        year (int): Year of date
        month (int): Month of date
    Returns
        Number of days in the month
    """
    if month > 12:
        logger.critical("wrong month in dinm")
        sys.exit(1)
    biss = bissextile(year)
    if (month == 4 or month == 6 or month == 9 or month == 11):
        daysinmonth = 30
    elif month == 2:
        if biss:
            daysinmonth = 29
        else:
            daysinmonth = 28
    else:
        daysinmonth = 31
    return daysinmonth


def jj2date(sjday: int) -> Tuple[int, int, int]:
    """
    Convert Julian day to date
    Args:
        sjday (int): julian day
    Returns
        year, mont and day of date
    """
    #  sys.exit("to be written")
    jday = int(sjday)
    year = 1950
    month = 1
    day = 1

    for iday in numpy.arange(1, jday + 1):
        day += 1
        daysinmonth = dinm(year, month)
        if (day > daysinmonth):
            day = 1
            month = month + 1
        if (month > 12):
            month = 1
            year = year + 1
    return year, month, day


def haversine(lon1: numpy.ndarray, lon2: numpy.ndarray, lat1: numpy.ndarray,
              lat2: numpy.ndarray) -> numpy.ndarray:
    """
    Compute distance between two arrays from coordinates using haversine
    function
    Args:
        lon1 (numpy.ndarray): longitude for first point
        lat1 (numpy.ndarray): latitude for first point
        lon2 (numpy.ndarray): longitude for second point
        lat2 (numpy.ndarray): latitude for second point
    Returns:
        Distance between two points (numpy.ndarray)
    """
    lon1 = numpy.deg2rad(lon1)
    lon2 = numpy.deg2rad(lon2)
    lat1 = numpy.deg2rad(lat1)
    lat2 = numpy.deg2rad(lat2)
    havlat = numpy.sin((lat2 - lat1) / 2)**2
    havlon = numpy.cos(lat1) * numpy.cos(lat2)
    havlon = havlon * numpy.sin((lon2 - lon1) / 2)**2
    d = 2 * const.Rearth * numpy.arcsin(numpy.sqrt(havlat + havlon))
    return d


def convert(x: numpy.ndarray, y: numpy.ndarray, u: numpy.ndarray,
            v: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Convert U V components from metric to angular system units
    Args:
        x (numpy.ndarray): longitude coordinate
        y (numpy.ndarray): latitude coordinate
        u (numpy.ndarray): zonal velocity in m/s
        v (numpy.ndarray): meridional velocity in m/s
    Returns:
        u, v converted in deg/s
    """
    assert (u.shape == v.shape)

    # nx = len(x)
    # ny = len(y)

    # if nx == u.shape[0]:
    #     assert(u.shape == (nx, ny))
    #     transpose = False
    # else:
    #     assert(u.shape == (ny, nx))
    #     transpose = True

    # Keep longitude between -180, 180
    # x[numpy.where(x > 180)] -= 360
    x0 = + x
    y0 = + y
    x0[numpy.where(x0 > 180)] -= 360
    # Conversion from spherical to cartesian coordinates and move it
    # for one second using U and V component
    lon = numpy.radians(x0)
    lat = numpy.radians(y0)
    sin_x = numpy.sin(lon)
    cos_x = numpy.cos(lon)
    sin_y = numpy.sin(lat)
    cos_y = numpy.cos(lat)
    xc = -u * sin_x - v * cos_x * sin_y
    yc = u * cos_x - v * sin_y * sin_x
    zc = v * cos_y
    xc = xc + const.Rearth * cos_y * cos_x
    yc = yc + const.Rearth * cos_y * sin_x
    zc = zc + const.Rearth * sin_y

    # Conversion from cartesian to spherical coordinates
    x1 = numpy.degrees(numpy.arctan2(yc, xc))
    y1 = numpy.degrees(numpy.arcsin(
                       zc / numpy.sqrt(xc * xc + yc * yc + zc * zc)))
    dx = x1 - x0
    dy = y1 - y0
    # Return the velocity in degrees/s
    return numpy.mod(dx + 180.,  360.) - 180., dy


def convert1d(x: numpy.ndarray, y: numpy.ndarray, u: numpy.ndarray,
              v: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Convert U V components from metric to angular system units
    Args:
        x (numpy.ndarray): longitude coordinate
        y (numpy.ndarray): latitude coordinate
        u (numpy.ndarray): zonal velocity in m/s
        v (numpy.ndarray): meridional velocity in m/s
    Returns:
        u, v converted in deg/s
    """

    # nx = len(x)
    # ny = len(y)

    # Keep longitude between -180, 180
    x[x > 180] -= 360
    x0 = x
    y0 = y

    # Conversion from spherical to cartesian coordinates and move it
    # for one second using U and V component
    lon = numpy.radians(x0)
    lat = numpy.radians(y0)
    sin_x = numpy.sin(lon)
    cos_x = numpy.cos(lon)
    sin_y = numpy.sin(lat)
    cos_y = numpy.cos(lat)
    xc = -u * sin_x - v * cos_x * sin_y
    yc = u * cos_x - v * sin_y * sin_x
    zc = v * cos_y
    xc += const.Rearth * cos_y * cos_x
    yc += const.Rearth * cos_y * sin_x
    zc += const.Rearth * sin_y

    # Conversion from cartesian to spherical coordinates
    x1 = numpy.degrees(numpy.arctan2(yc, xc))
    y1 = numpy.degrees(numpy.arcsin(
        zc / numpy.sqrt(xc * xc + yc * yc + zc * zc)))
    dx = x1 - x0
    dy = y1 - y0

    # Return the velocity in degrees/s
    return (dx + 180) % 360 - 180, dy


def ms2degd(lon: numpy.ndarray, lat: numpy.ndarray, u: numpy.ndarray,
            v: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """ Conversion from m/s to deg/timestep
    Args:
        x (numpy.ndarray): longitude coordinate
        y (numpy.ndarray): latitude coordinate
        u (numpy.ndarray): zonal velocity in m/s
        v (numpy.ndarray): meridional velocity in m/s
    Returns:
        u, v converted in deg/timestep
    """
    geod = pyproj.Geod(ellps='WGS84')
    #  pyproj.Geod.fwd expects bearings to be clockwise angles from north
    # (in degrees)
    azim = numpy.pi / 2. - numpy.arctan2(v, u)
    dist1s = numpy.sqrt(u**2 + v**2)
    lon180 = numpy.mod(lon + 180, 360) - 180
    lonend, latend, _ = geod.fwd(lon180, lat, numpy.rad2deg(azim), dist1s,
                                 radians=False)
    uout = lonend - lon180
    vout = latend - lat
    return uout, vout


def npdate2datetime(npdate: numpy.datetime64)-> datetime.datetime:
    ts = ((npdate - numpy.datetime64('1970-01-01T00:00:00Z'))
          / numpy.timedelta64(1, 's'))
    ddate = datetime.datetime.utcfromtimestamp(ts)
    return ddate


def datetime2npdate(ddate: datetime.datetime)-> numpy.datetime64:
    npfmt = '%Y-%m-%dT%H:%M:%S'
    return numpy.datetime64(ddate.strftime(npfmt))
