# vim: ts=4:sts=4:sw=4
#
# @author lucile.gaultier@oceandatalab.com
# @date 2024-01-10
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

"""
Build Grid and read velocity
"""
from scipy import interpolate
import numpy
import netCDF4
import logging
import sys
import os
from typing import Optional
import velocity_metrics.writer.write_utils as write
import velocity_metrics.writer.idf as idf
import velocity_metrics.utils.constant as const

logger = logging.getLogger(__name__)


class trajectory():
    def __init__(self, ):
        pass


def read_trajectory(infile: str, list_var: list) -> dict:
    dict_var = {}
    fid = netCDF4.Dataset(infile, 'r')
    for key in list_var:
        if key in fid.variables.keys():
            dict_var[key] = fid.variables[key][:]
        else:
            logger.warn(f'variable {key} not found in file {infile}')
    fid.close()
    return dict_var


def read_points(par: dict):
    # # TODO, lon= ... lat= ...
    dic = read_trajectory(par["filetrajectory"],
                          ('lon_hr', 'lat_hr', 'mask_hr', 'zonal_velocity'))
    traj = trajectory()
    traj.lon = dic['lon_hr'][-1, :]
    traj.lat = dic['lat_hr'][-1, :]
    traj.mask = dic['mask_hr'][-1, :]
    return traj


def make_mask(par: dict, VEL: dict):
    # mask_grid
    # VEL.lon = (VEL.lon + 360.) % 360.
    masku = + VEL['u']['array'][0, :, :]
    masku[abs(masku) > 50.] = numpy.nan
    masku[abs(VEL['v']['array'][0, :, :]) > 50.] = numpy.nan
    masku[VEL['v']['array'][0, :, :] == 0.] = numpy.nan
    masku[VEL['u']['array'][0, :, :] == 0.] = numpy.nan
    vlat = VEL['u']['lat']
    vlon = VEL['u']['lon']
    Teval = interpolate.RectBivariateSpline(vlat, vlon,
                                            numpy.isnan(masku), kx=1, ky=1,
                                            s=0)
    return Teval


def make_grid(par: dict, VEL: dict, mask: Optional[bool] = True):
    _coord = list(par["parameter_grid"])
    if len(_coord) == 6:
        lon0, lon1, dlon, lat0, lat1, dlat = list(par["parameter_grid"])
    else:
        logger.critical('Grid parameters must be specified in parameter_grid '
                     '(lon0, lon1, dlon, lat0, lat1, dlat)')
        sys.exit(1)
    # Build grid
    #lon0 = numpy.mod(lon0 + 360, 360)
    #lon1 = numpy.mod(lon1 + 360, 360)
    traj = trajectory()
    if lon1 > lon0:
        lontmp = numpy.linspace(lon0, lon1, int((lon1 - lon0) / dlon))
    else:
        lontmp = numpy.linspace(lon1, lon0, int((lon0 - lon1) / dlon))
    lattmp = numpy.linspace(lat0, lat1, int((lat1 - lat0) / dlat))
    traj.lon, traj.lat = numpy.meshgrid(lontmp, lattmp)
    # TODO PROVIDE A MASK FILE
    Teval = make_mask(par, VEL)
    masktmp = Teval(lattmp, lontmp)
    shape_tra = numpy.shape(traj.lon)
    if mask is True:
        Teval = make_mask(par, VEL)
        masktmp = Teval(lattmp, lontmp)
        masktmp = masktmp.reshape(shape_tra)
    else:
        masktmp = numpy.zeros(shape_tra)
    # TODO create real mask
    # masktmp = numpy.zeros(shape_tra)
    traj.mask = numpy.ma.getdata((masktmp > 0))
    if numpy.ma.getdata(traj.mask).all():
        logger.info(f'no data in box {lon0}, {lon1}, {lat0}, {lat1}')
        sys.exit(0)
    return traj


def interp_vel(VEL: dict, coord: dict) -> dict:
    interp2d = interpolate.RectBivariateSpline
    _inte = {}
    _inte['time'] = coord['time']
    for key in VEL.keys():
        _inte[key] = None
        VEL[key]['array'][numpy.isnan(VEL[key]['array'])] = 0
    if len(numpy.shape(VEL['u']['array'])) > 2:
        for key in VEL.keys():
            _inte[key] = []
        for t in range(len(coord['time'])):
            for key in VEL.keys():
                nlon = VEL[key]['lon']
                nlat = VEL[key]['lat']
                _tmp = interp2d(nlon, nlat,
                                VEL[key]['array'][t, :, :].T, kx=1, ky=2)
                _inte[key].append(_tmp)
    else:
        for key in VEL.keys():
            nlon = VEL[key]['lon']
            nlat = VEL[key]['lat']
            _tmp = interp2d(nlon, nlat,
                            VEL[key]['array'][0, :, :].T, kx=1, ky=2)
            _inte[key] = list([_tmp, ])
    return _inte


def write_drifter(par: dict, paro: dict, drifter: trajectory,
                  listTr: list,
                  output_dir: Optional[str] = './',
                  file_pattern: Optional[str] = 'Advection',
                  output_file: Optional[str] = None,
                  dic_global_attr: Optional[dict] = {},
                  is_idf: Optional[bool] = False
                  ):
    start = par["first_date"].strftime('%Y%m%d')
    stop = par["last_date"].strftime('%Y%m%d')
    file_def = f'{file_pattern}_{start}_{stop}.nc'
    if output_file is None:
        output_file = os.path.join(output_dir, file_def)
    global_attr = idf.global_idf
    for key,value in dic_global_attr.items():
        global_attr[key] = value
    # if ide is not None:
    #    global_attr['ide'] = ide
    global_attr['time_coverage_start'] = par["first_date"].strftime(const.FMT)
    write.write_listracer_1d(output_file, drifter, par, paro, listTr,
                             global_attribute=global_attr)
