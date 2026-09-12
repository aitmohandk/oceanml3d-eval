# vim: ts=4:sts=4:sw=4
#
# @author lucile.gaultier@oceandatalab.com
# @date 2022-10-10
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

""" Read irregular modis sst data """

import netCDF4
import numpy
import os
from .utils import cftime2datetime
from scipy import ndimage
from fronts_detection import multiimage


def reader(file_path: str, nfile: str, box: list) -> dict:
    """
    Considers a file name and an area. Returns coordinates and sst np
    arrays. Returns data time also.

    Input :
        - file_path (str): total file path
        - nfile (str): name of the file
        - box (list of 4 int): defines the area of the world map to select. The
        area is defined with the list box as [min_lon, max_lon, min_lat,
                                              max_lat].
    Output :
        - dico_output (dict): standardized data
            - lon_reg, lat_reg: 1d coordinates (works if modis data are
            regular)
            - lon2d, lat2d: 2d coordinates
            - sst: sst of the selected area
            - time:
    """
    dico_output = {}
    extension = os.path.splitext(file_path)[1]
    if extension != '.nc':
        raise ValueError("Extension should be '.nc'")
    try:
        # Try to find the netcdf file
        handler = netCDF4.Dataset(file_path, 'r')
    except FileNotFoundError:
        return None
    # Extract data from the handler
    lon = handler['lon'][:]  # 1d
    lat = handler['lat'][:]  # 1d
    sst = handler['sea_surface_temperature'][0, :, :]  # 2d
    # qual = handler['quality_level'][:]
    sst_scale = handler['sea_surface_temperature'].scale_factor
    sst_offset = handler['sea_surface_temperature'].add_offset
    fill_value = handler['sea_surface_temperature']._FillValue
    time = handler['time'][0]
    time_units = handler['time'].units
    handler.close()

    # Turning float64 sst data into short integers (int16)
    # 2*sst_scale may change and become sst_scale with other data
    sst_short = numpy.array((sst - sst_offset) / (2 * sst_scale),
                            dtype=numpy.int16)
    sst_short = numpy.ma.masked_where(sst_short == fill_value, sst_short)

    # selecting the area thanks to the box
    ind_lon = numpy.where((lon > box[0]) & (lon < box[1]))
    ind_lat = numpy.where((lat > box[2]) & (lat < box[3]))
    if numpy.shape(ind_lon)[1] == 0 or numpy.shape(ind_lat)[1] == 0:
        return None
    lon_reg = + lon[ind_lon]  # 1d
    lat_reg = + lat[ind_lat]  # 1d

    # Get 2d coordinates
    lon2d, lat2d = numpy.meshgrid(lon_reg, lat_reg)

    # Get sst data in the area
    _sst_reg_tmp = + sst_short[ind_lat[0], :]
    sst_reg_tmp = _sst_reg_tmp[:, ind_lon[0]]
    sst_reg = sst_reg_tmp
    sst_reg = numpy.ma.masked_where(sst_reg == fill_value, sst_reg)

    # Return standardized dictionnary format
    dico_output['lon_reg'] = lon_reg
    dico_output['lat_reg'] = lat_reg
    dico_output['lon2d'] = lon2d
    dico_output['lat2d'] = lat2d
    dico_output['sst'] = sst_reg
    # Get x-gradient in "sx"
    gc_row = ndimage.sobel(sst_reg, axis=0, mode='nearest')
    # Get y-gradient in "sy"
    gc_col = ndimage.sobel(sst_reg, axis=1, mode='nearest')
    # Divide by 4 to rescale the Sobel Kernel : | -1 0 1 |
    #                                           | -2 0 2 |
    #                                           | -1 0 1 |
    gc_col = gc_col / 4
    gc_row = gc_row / 4

    # Get square root of sum of squares
    gc_lon, gc_lat = multiimage.rescale_gradient(gc_row, gc_col, lon_reg,
                                                 lat_reg)
    gc_lon = numpy.ma.masked_where(abs(gc_lon) > 10, gc_lon)
    gc_lat = numpy.ma.masked_where(abs(gc_lat) > 10, gc_lat)

    dico_output['sst_grad_lon'] = gc_lon
    dico_output['sst_grad_lat'] = gc_lat
    dico_output['sst_grad'] = numpy.hypot(dico_output['sst_grad_lon'],
                                          dico_output['sst_grad_lat'])

    tme = netCDF4.num2date(time, units=time_units)
    dico_output['time'] = cftime2datetime(tme)
    return dico_output
