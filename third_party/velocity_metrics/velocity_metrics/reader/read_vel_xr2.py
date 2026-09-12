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


import xarray
import read_utils_xr
import logging
import numpy
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def read_vel(list_vel: list, box: list, depth: int,
             name_lon: Optional[str] = 'longitude',
             name_lat: Optional[str] = 'latitude',
             name_u: Optional[str] = 'uo', name_v: Optional[str] = 'vo'
             ) -> Tuple[dict, dict]:
    ds = xarray.open_mfdataset(list_vel, concat_dim="time", combine="nested")
    IDL = read_utils_xr.check_crossing(box[0], box[1])
    _box1 = box[1]
    _box0 = box[0]
    lon_ctor = getattr(ds, name_lon)
    if (lon_ctor.values > 185).any():
        _box0 = numpy.mod(box[0] + 360, 360)
        _box1 = numpy.mod(box[1] + 360, 360)
        if box[0] > box[1]:
            IDL = True
    logger.debug(f'IDL {IDL}')
    if IDL is True:
        ds = ds.sortby(numpy.mod(lon_ctor + 360, 360))
        if depth is not None:
            if name_lat == 'lat':
                VEL = ds.sel(depth=depth, lat=slice(box[2], box[3]))
            else:
                VEL = ds.sel(depth=depth, latitude=slice(box[2], box[3]))
        else:
            if name_lat == 'lat':
                VEL = ds.sel(lat=slice(box[2], box[3]))
            else:
                VEL = ds.sel(latitude=slice(box[2], box[3]))
    else:
        if depth is not None:
            if name_lon == 'lon':
                VEL = ds.sel(depth=depth, lat=slice(box[2], box[3]),
                             lon=slice(_box0, _box1))

            else:
                VEL = ds.sel(depth=depth, latitude=slice(box[2], box[3]),
                             longitude=slice(_box0, _box1))
        else:
            if name_lon == 'lon':
                VEL = ds.sel(lat=slice(box[2], box[3]),
                             lon=slice(_box0, _box1))
            else:
                VEL = ds.sel(latitude=slice(box[2], box[3]),
                             longitude=slice(_box0, _box1))
    # Intialize empty matrices
    ds.close()
    del ds
    coord = {}
    dic = {}
    lon_ctor = getattr(VEL, name_lon)
    lat_ctor = getattr(VEL, name_lat)

    coord['lonu'] = numpy.mod(lon_ctor[:].values + 360., 360.)
    coord['latu'] = lat_ctor[:].values
    coord['lonv'] = numpy.mod(lon_ctor[:].values + 360., 360.)
    coord['latv'] = lat_ctor[:].values
    coord['time'] = [numpy.datetime64(x) for x in VEL.time.values]
    logger.debug(f'{len(VEL.time.values)} time steps found')
    # stfmt = '%Y-%m-%dT%H:%M:%s.%f'
    # _time = [datetime.datetime.strptime(x, stfmt) for x in VEL.time.values]
    uo_ctor = getattr(VEL, name_u)
    vo_ctor = getattr(VEL, name_v)
    dic['ums'] = {'array': uo_ctor.values, 'lon': coord['lonu'],
                  'lat': coord['latu'], 'time': coord['time']}
    dic['vms'] = {'array': vo_ctor.values, 'lon': coord['lonv'],
                  'lat': coord['latv'], 'time': coord['time']}
    return dic, coord
