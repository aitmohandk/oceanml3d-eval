# vim: ts=4:sts=4:sw=4
#
# @author lucile.gaultier@oceandatalab.com
# @date 2022-06-10
#
# Copyright (C) 2022-2024 OceanDataLab
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

""" Read fsle regular netcdf files """

import netCDF4
import numpy
from .utils import cftime2datetime


def reader(file_path: str, box: list, per: float) -> dict:
    handler = netCDF4.Dataset(file_path, 'r')
    lon = handler['lon'][:]
    lat = handler['lat'][:]
    ind_lon = numpy.where((lon > box[0]) & (lon < box[1]))[0]
    ind_lat = numpy.where((lat > box[2]) & (lat < box[3]))[0]
    lon_reg = lon[ind_lon]
    lat_reg = lat[ind_lat]
    lon2d, lat2d = numpy.meshgrid(lon_reg, lat_reg)
    FSLE = handler['FSLE_bin'][0, ind_lat, ind_lon]
    FSLE[FSLE > 100] = numpy.nan
    tme = netCDF4.num2date(handler['time'][0], handler['time'].units)
    tme = cftime2datetime(tme)
    dico_data = {'lon2d': lon2d, 'lat2d': lat2d, 'time': tme}
    handler.close()

    per = 80
    min_val = numpy.min(FSLE)
    FSLE_flat = FSLE.flatten()
    FSLE_sort = numpy.sort(FSLE_flat[FSLE_flat > min_val])
    threshold_val = FSLE_sort[int((len(FSLE_sort) - 1) * per / 100)]
    FSLE_per = min_val * numpy.ones(numpy.shape(FSLE))
    FSLE_per[FSLE > threshold_val] = FSLE[FSLE > threshold_val]
    # FSLE_per[FSLE>min_val ] = FSLE[FSLE>min_val]
    # print(numpy.max(FSLE_per))
    # FSLE_per[FSLE> 1] = 1
    FSLE2a_per = ((FSLE_per - numpy.min(FSLE_per))
                  / (numpy.max(FSLE_per) - numpy.min(FSLE_per)))
    FSLE2_per = numpy.array(FSLE2a_per)
    FSLE2_per[FSLE2_per < 0.0] = 0
    dico_data['FSLE'] = FSLE2_per
    return dico_data
