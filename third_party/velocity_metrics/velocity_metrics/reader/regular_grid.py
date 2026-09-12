# vim: ts=4:sts=4:sw=4

# @date 2022-10-06
# @author lucile.gaultier@oceandatalab.com

# Copyright (C) 2022-2023 OceanDataLab
# This file is part of fronts_detection

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
import os
import sys
from typing import Optional
from .utils import cftime2datetime, get_gradient
import logging
logger = logging.getLogger(__name__)


def reader(file_path: str, box: list, variable: str,
           longitude: Optional[str] = 'lon',
           latitude: Optional[str] = 'lat') -> dict:
    """
    Considers a file name and an area. Returns coordinates and sst np
    arrays. Returns data time also.

    Input :
        - file_path (str): total file path with name
        - box (list of 4 int): defines the area of the world map to select. The
        area is defined with the list box as [min_lon, max_lon, min_lat,
                                              max_lat].
        - variable (str): name of variable in netcdf
        - longitude (str, default is 'lon'): name of longitude in netcdf
        - latitude (str, default is 'lat'): name of latitude in netcdf

    Output :
        - dico_output (dict): standardized data
            - lon_reg, lat_reg: 1d coordinates (works if modis data are
            regular)
            - lon2d, lat2d: 2d coordinates
            - sst_reg: sst of the selected area
            - time:
    """
    dico_output = {}
    extension = os.path.splitext(file_path)[1]
    if extension != '.nc':
        raise ValueError("Extension should be '.nc'")
        sys.exit(0)
    try:
        # Try to find the netcdf file
        handler = netCDF4.Dataset(file_path, 'r')
    except FileNotFoundError:
        logger.error(f'{file_path} file not found')
        sys.exit(1)
    # handler.set_auto_mask = False

    # Extract data from the handler
    lon = handler[longitude][:]  # 1d
    lat = handler[latitude][:]  # 1d
    sst = handler[variable][0, :, :]  # 2d
    sstg = handler[variable][0, :, :]
    if 'quality_level' in handler.variables.keys():
        qual = handler['quality_level'][0, :, :]
        sst = numpy.ma.masked_where(qual < 3, sst)
        sstg = numpy.ma.masked_where(qual < 4, sstg)
    sst_scale = handler[variable].scale_factor
    sst_offset = handler[variable].add_offset
    fill_value = handler[variable]._FillValue
    time = netCDF4.num2date(handler['time'][:], handler['time'].units)

    # Turning float64 sst data into short integers (int16)
    sst_short = numpy.array((sst - sst_offset) / (sst_scale),
                            dtype=numpy.int16)
    sst_short = numpy.ma.masked_where(sst_short == fill_value, sst_short)

    # Selecting the area thanks to the box
    ind_lon = numpy.where((lon > box[0]) & (lon < box[1]))
    ind_lat = numpy.where((lat > box[2]) & (lat < box[3]))
    if len(ind_lon) == 0 or len(ind_lat) == 0:
        msg_box = ', '.join(box)
        logger.error(f'No data found in box {msg_box}')
        sys.exit(1)
    lon_reg = + lon[ind_lon]  # 1d
    lat_reg = + lat[ind_lat]  # 1d

    # Get 2d coordinates
    lon2d, lat2d = numpy.meshgrid(lon_reg, lat_reg)

    # Get sst data in the area
    _sst_reg_tmp = + sst_short[ind_lat[0], :]
    sst_reg_tmp = _sst_reg_tmp[:, ind_lon[0]]
    sst_reg = sst_reg_tmp.copy()
    sst_reg = numpy.ma.masked_where(sst_reg == fill_value, sst_reg)
    if 'quality_level' in handler.variables.keys():
        _qual_reg = + qual[ind_lat[0], :]
        qual_reg = _qual_reg[:, ind_lon[0]]
    _sstg_reg_tmp = sstg[ind_lat[0], :].copy()
    sstg_reg_tmp = _sstg_reg_tmp[:, ind_lon[0]].copy()
    sstg_reg = sstg_reg_tmp.copy()
    sstg_reg = numpy.ma.masked_where(sstg_reg == fill_value, sstg_reg)
    sst_reg = numpy.ma.masked_where(sst_reg == fill_value, sst_reg)

    # Return standardized dictionnary format
    dico_output['lon_reg'] = lon_reg
    dico_output['lat_reg'] = lat_reg
    dico_output['lon2d'] = lon2d
    dico_output['lat2d'] = lat2d
    dico_output['sst'] = sst_reg
    dico_output['sst_final'] = sstg_reg
    if 'quality_level' in handler.variables.keys():
        dico_output['sst_quality_level'] = qual_reg
    # Get x-gradient in "sx"
    gc_lon, gc_lat = get_gradient(sstg_reg, lon2d, lat2d)

    if 'quality_level' in handler.variables.keys():
        gc_lon = numpy.ma.masked_where(qual_reg < 4, gc_lon)
        gc_lat = numpy.ma.masked_where(qual_reg < 4, gc_lat)
        dico_output['sst_final'] = numpy.ma.masked_where(qual_reg < 4,
                                                         sstg_reg)
        dico_output['sst'] = numpy.ma.masked_where(qual_reg < 4, sst_reg)
        gc_lon[gc_lon.mask] = numpy.nan
        gc_lat[gc_lat.mask] = numpy.nan
        dico_output['sst_final'][dico_output['sst_final'].mask] = numpy.nan

    dico_output['sst_grad_lon'] = gc_lon
    dico_output['sst_grad_lat'] = gc_lat
    dico_output['sst_grad'] = numpy.hypot(dico_output['sst_grad_lon'],
                                          dico_output['sst_grad_lat'])
    dico_output['time'] = time[0]
    dico_output['time'] = cftime2datetime(time[0])

    handler.close()
    return dico_output
