# vim: ts=4:sts=4:sw=4

# @date 2022-10-06
# @author lucile.gaultier@oceandatalab.com

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

import netCDF4
import numpy
import logging
import os
import sys
import datetime
from typing import Optional
import velocity_metrics.utils.load_parameters as load_parameters
from velocity_metrics.reader.seascope_idf import geoloc_from_gcps
logger = logging.getLogger(__name__)


def build_par(region: str, data_type_file: str, depth: Optional[int] = None
              ) -> dict:
    x = load_parameters.sel_region(region)
    lllon, urlon, lllat, urlat, coords, name = x
    logger.info(f'Load velocity parameters {data_type_file}')
    if os.path.exists(data_type_file):
        dic = load_parameters.load_file(data_type_file, section='data')
    else:
        logger.error(f'please provide data_type file {data_type_file}')
        sys.exit(1)
    dic["bbox"] = [lllon, urlon, lllat, urlat]
    if depth is not None and 'depth' in dic.keys():
        dic['depth'] = dic['depth'][depth]
    return dic


def read_velocity(file_vel: str, dic_par: dict) -> dict:
    handler = netCDF4.Dataset(file_vel, 'r')
    dic_vel = {}
    box = None
    if 'bbox' in dic_par.keys():
        box = dic_par['bbox']
    # Extract data from the handler
    if 'gcp' in dic_par['nlon']:
        _slice_lon = slice(0, None)
        _slice_lat = slice(0, None)
        lon_gcp = numpy.array(handler['lon_gcp'][_slice_lon])
        lat_gcp = numpy.array(handler['lat_gcp'][_slice_lat])
        i_gcp = numpy.array(handler['index_lat_gcp'][_slice_lat])
        j_gcp = numpy.array(handler['index_lon_gcp'][_slice_lon])
        # cond = (lon_gcp[-1] - lon_gcp[0]) > 180.0
        # if cond:
        #    print('Difference between first and last longitude exceeds '
        #          '180 degrees, assuming IDL crossing and remapping '
        #          'longitudes in [0, 360]')
        #    lon_gcp = numpy.mod((lon_gcp + 360.0), 360.0)
        j_shaped, i_shaped = numpy.meshgrid(j_gcp, i_gcp)
        lon_shaped, lat_shaped = numpy.meshgrid(lon_gcp, lat_gcp[:])
        shape = numpy.shape(handler[dic_par['u']][0, _slice_lat, _slice_lon])
        dst_lin = numpy.arange(0, shape[0])
        dst_pix = numpy.arange(0, shape[1])
        _dst_lin = numpy.tile(dst_lin[:, numpy.newaxis], (1, shape[1]))
        _dst_pix = numpy.tile(dst_pix[numpy.newaxis, :], (shape[0], 1))

        lon2d, lat2d = geoloc_from_gcps(lon_shaped, lat_shaped, i_shaped,
                                        j_shaped, _dst_lin, _dst_pix)
        dic_vel['lon'] = lon2d[0, :]
        dic_vel['lat'] = lat2d[:, 0]
    else:
        _slice_lat = slice(0, None)
        _slice_lon = slice(0, None)
        dic_vel['lon'] = handler[dic_par['nlon']][:]
        if box is not None:
            ind = numpy.where(dic_vel['lon'] > box[0]
                              & dic_vel['lon'] < box[1])
            _slice_lon = slice(ind[0], ind[-1] + 1)
        dic_vel['lat'] = handler[dic_par['nlat']][:]
        if box is not None:
            ind = numpy.where(dic_vel['lat'] > box[0]
                              & dic_vel['lat'] < box[1])
            _slice_lat = slice(ind[0], ind[-1] + 1)
    dic_vel['u'] = handler[dic_par['varuu']][:]
    dic_vel['v'] = handler[dic_par['varvv']][:]
    _shape = numpy.shape(dic_vel['u'])
    idepth = None
    if set(('depth', 'ndepth')).issubset(dic_par.keys()):
        if (dic_par['depth'] is not None) and (dic_par['ndepth'] is not None):
            depth = handler[dic_par['ndepth']][:]
            idepth = numpy.abs(depth - dic_par['depth']).argmin()
    if len(_shape) == 4:
        dic_vel['u'] = dic_vel['u'][:, idepth, _slice_lat, _slice_lon]
        dic_vel['v'] = dic_vel['v'][:, idepth, _slice_lat, _slice_lon]
    elif len(_shape) == 3:
        if idepth is None:
            dic_vel['u'] = dic_vel['u'][:, _slice_lat, _slice_lon]
            dic_vel['v'] = dic_vel['v'][:, _slice_lat, _slice_lon]
        else:
            dic_vel['u'] = dic_vel['u'][idepth, _slice_lat, _slice_lon]
            dic_vel['v'] = dic_vel['v'][idepth, _slice_lat, _slice_lon]
    if 'time' in handler.variables.keys():
        time = netCDF4.num2date(handler['time'][:],
                                units=handler['time'].units)
        time_delta = int(dic_par['time_coverage_hours'] / 2)
        _stop = []
        _start = []
        for _time in time:
            _stop.append(_time + datetime.timedelta(hours=time_delta))
            _start.append(_time - datetime.timedelta(hours=time_delta))
    elif 'time_coverage_start' in handler.ncattrs():
        _fmt = dic_par['fmt']
        _start = handler.time_coverage_start
        _start = list(datetime.datetime.strptime(_start, _fmt))
        _stop = handler.time_coverage_end
        _stop = list(datetime.datetime.strptime(_stop, _fmt))
    dic_vel['time_coverage_start'] = _start
    dic_vel['time_coverage_end'] = _stop
    handler.close()
    if len(numpy.shape(dic_vel['u'])) == 2:
        dic_vel['u'] = dic_vel['u'][numpy.newaxis, :, :]
        dic_vel['v'] = dic_vel['v'][numpy.newaxis, :, :]
    return dic_vel
