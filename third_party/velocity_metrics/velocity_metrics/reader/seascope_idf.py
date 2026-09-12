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


import netCDF4
import numpy
import os
import logging
from velocity_metrics.fronts.read_utils import cftime2datetime, get_gradient
logger = logging.getLogger(__name__)


def geoloc_from_gcps(gcplon, gcplat, gcplin, gcppix, lin, pix):
    """"""
    import pyproj
    geod = pyproj.Geod(ellps='WGS84')
    fwd, bwd, dis = geod.inv(gcplon[:, :-1], gcplat[:, :-1],
                             gcplon[:, 1:], gcplat[:, 1:])

    # Find line and column for the top-left corner of the 4x4 GCPs cell which
    # contains the requested locations
    nlin, npix = gcplat.shape
    _gcplin = gcplin[:, 0]
    _gcppix = gcppix[0, :]
    top_line = numpy.searchsorted(_gcplin, lin, side='right') - 1
    left_column = numpy.searchsorted(_gcppix, pix, side='right') - 1

    # Make sure this line and column remain within the matrix and that there
    # are adjacent line and column to define the bottom-right corner of the 4x4
    # GCPs cell
    top_line = numpy.clip(top_line, 0, nlin - 2)
    bottom_line = top_line + 1
    left_column = numpy.clip(left_column, 0, npix - 2)
    right_column = left_column + 1

    # Compute coordinates of the requested locations in the 4x4 GCPs cell
    line_extent = _gcplin[bottom_line] - _gcplin[top_line]
    column_extent = _gcppix[right_column] - _gcppix[left_column]
    line_rel_pos = (lin - _gcplin[top_line]) / line_extent
    column_rel_pos = (pix - _gcppix[left_column]) / column_extent

    # Compute geographical coordinates of the requested locations projected on
    # the top and bottom lines
    lon1, lat1, _ = geod.fwd(gcplon[top_line, left_column],
                             gcplat[top_line, left_column],
                             fwd[top_line, left_column],
                             dis[top_line, left_column] * column_rel_pos)
    lon2, lat2, _ = geod.fwd(gcplon[bottom_line, left_column],
                             gcplat[bottom_line, left_column],
                             fwd[bottom_line, left_column],
                             dis[bottom_line, left_column] * column_rel_pos)

    # Compute the geographical coordinates of the requested locations projected
    # on a virtual column joining the projected points on the top and bottom
    # lines
    fwd12, bwd12, dis12 = geod.inv(lon1, lat1, lon2, lat2)
    lon, lat, _ = geod.fwd(lon1, lat1, fwd12, dis12 * line_rel_pos)

    return lon, lat


def reader(file_path: str, box: list, var: str) -> dict:
    """
    Considers a file name and an area. Returns coordinates and sst np
    arrays. Returns data time also.

    Input :
        - file_path (str): total file path
        - box (list of 4 int): defines the area of the world map to select. The
        area is defined with the list box as [min_lon, max_lon, min_lat,
                                              max_lat].
        - variable (str): name of variable in netcdf

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
    lon_gcp = numpy.array(handler['lon_gcp'][:])
    lat_gcp = numpy.array(handler['lat_gcp'][:])
    # Enforce longitude continuity (to be improved)
    if len(numpy.shape(lon_gcp)) == 2:
        regular = False
        i_gcp = numpy.array(handler['index_row_gcp'][:])
        j_gcp = numpy.array(handler['index_cell_gcp'][:])
    else:
        regular = True
        i_gcp = numpy.array(handler['index_lat_gcp'][:])
        j_gcp = numpy.array(handler['index_lon_gcp'][:])

    # qual = handler['quality_level'][:]
    sst = handler[var][0, :, :]
    fill_value = handler[var]._FillValue
    time = handler['time'][:]
    units_time = handler['time'].units
    handler.close()
    if regular is True:
        ind_lon = numpy.where((lon_gcp > box[0]-1) & (lon_gcp < box[1]+1))
        ind_lat = numpy.where((lat_gcp > box[2]-1) & (lat_gcp < box[3]+1))
        lon_gcp = lon_gcp[ind_lon]
        lat_gcp = lat_gcp[ind_lat]
        j_gcp = j_gcp[ind_lon]
        i_gcp = i_gcp[ind_lat]
        i0 = numpy.min(i_gcp)
        i1 = numpy.max(i_gcp) + 1
        j0 = numpy.min(j_gcp)
        j1 = numpy.max(j_gcp) + 1
        j_gcp = j_gcp - j_gcp[0]
        i_gcp = i_gcp - i_gcp[0]
    else:
        ind_lon_lat = numpy.where((lon_gcp > box[0]) & (lon_gcp < box[1])
                                  & (lat_gcp > box[2]) & (lat_gcp < box[3]))
        i_gcp_0 = numpy.min(ind_lon_lat[0])
        i_gcp_1 = numpy.max(ind_lon_lat[0])
        j_gcp_0 = numpy.min(ind_lon_lat[1])
        j_gcp_1 = numpy.max(ind_lon_lat[1])
        lon_gcp = lon_gcp[i_gcp_0: i_gcp_1 + 1, j_gcp_0: j_gcp_1 + 1]
        lat_gcp = lat_gcp[i_gcp_0: i_gcp_1 + 1, j_gcp_0: j_gcp_1 + 1]
        i0 = i_gcp[i_gcp_0]
        i1 = i_gcp[i_gcp_1] + 1
        j0 = j_gcp[j_gcp_0]
        j1 = j_gcp[j_gcp_1] + 1
        j_gcp = j_gcp - j_gcp[0]
        i_gcp = i_gcp - i_gcp[0]
    if regular is True:
        cond = (lon_gcp[-1] - lon_gcp[0]) > 180.0
    else:
        cond = (lon_gcp[-1, -1] - lon_gcp[0, 0]) > 180.0
    if cond:
        _msg = ('Difference between first and last longitude exceeds '
                '180 degrees, assuming IDL crossing and remapping '
                'longitudes in [0, 360]')
        logger.info(_msg)
        box[0] = numpy.mod(box[0] + 360, 360)
        box[1] = numpy.mod(box[1] + 360, 360)
        lon_gcp = numpy.mod((lon_gcp + 360.0), 360.0)
    sst = sst[i0: i1, j0: j1]
    # Turning float64 sst data into short integers (int16)
    # 2*sst_scale may change and become sst_scale with other data
    # sst_short = numpy.array((sst - sst_offset) / (2*sst_scale),
    #                        dtype=numpy.int16)
    sst = numpy.ma.masked_where(sst == fill_value, sst)
    sst_short = numpy.array((sst * 100),
                            dtype=numpy.int16)
    # Selecting the area thanks to the box
    if numpy.shape(j_gcp)[0] == 0 or numpy.shape(i_gcp)[0] == 0:
        return None

    sst_reg = + sst_short
    sst_reg = numpy.ma.masked_where(sst_reg == fill_value, sst_reg)

    # Restore shape of the GCPs
    # gcps_shape = (8, 8)  # hardcoded in SEAScope
    # i_shaped = numpy.reshape(i_gcp, gcps_shape)
    # j_shaped = numpy.reshape(j_gcp, gcps_shape)
    # lon_shaped = numpy.reshape(lon_gcp, gcps_shape)
    # lat_shaped = numpy.reshape(lat_gcp, gcps_shape)
    if len(numpy.shape(i_gcp)) == 1:
        j_shaped, i_shaped = numpy.meshgrid(j_gcp, i_gcp)
    else:
        i_shaped = i_gcp
        j_shaped = j_gcp
    if len(numpy.shape(lon_gcp)) == 1:
        lon_shaped, lat_shaped = numpy.meshgrid(lon_gcp, lat_gcp[:])
    else:
        lon_shaped = lon_gcp[:, :]
        lat_shaped = lat_gcp[:, :]
    shape = numpy.shape(sst_reg)
    dst_lin = numpy.arange(0, shape[0])
    dst_pix = numpy.arange(0, shape[1])
    _dst_lin = numpy.tile(dst_lin[:, numpy.newaxis], (1, shape[1]))
    _dst_pix = numpy.tile(dst_pix[numpy.newaxis, :], (shape[0], 1))

    lon2d, lat2d = geoloc_from_gcps(lon_shaped, lat_shaped, i_shaped,
                                    j_shaped, _dst_lin, _dst_pix)

    # Return standardized dictionnary format
#    dico_output['lon_reg'] = lon_reg
#    dico_output['lat_reg'] = lat_reg
    dico_output['lon2d'] = lon2d
    dico_output['lat2d'] = lat2d
    dico_output['sst'] = sst_reg
    # Get x-gradient in "sx"
    gc_lon, gc_lat = get_gradient(sst_reg, lon2d, lat2d)


    gc_lon = numpy.ma.masked_where(abs(gc_lon) > 10, gc_lon)
    gc_lat = numpy.ma.masked_where(abs(gc_lat) > 10, gc_lat)
    dico_output['sst_grad_lon'] = gc_lon
    dico_output['sst_grad_lat'] = gc_lat
    dico_output['sst_grad'] = numpy.hypot(dico_output['sst_grad_lon'],
                                          dico_output['sst_grad_lat'])
    tme = netCDF4.num2date(time[0], units=units_time)
    dico_output['time'] = cftime2datetime(tme)

    return dico_output
