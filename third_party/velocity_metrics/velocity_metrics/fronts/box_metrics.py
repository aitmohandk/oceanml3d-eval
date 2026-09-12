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


import pickle
# from statistics import mode
# from collections import Counter
from typing import Optional
import scipy.stats
import os
import sys
import re
import datetime
import numpy
import netCDF4
import logging
import argparse
from matplotlib import pyplot

from velocity_metrics.utils.load_parameters import load_fronts_stat_parameters


logging.getLogger('matplotlib.font_manager').disabled = True
logger = logging.getLogger(__name__)

FMT = "%Y-%m-%dT%H:%M:%S.%fZ"


def mask_values(dic_in: dict, nvar: str, threshold_vel: Optional[float] = 0,
                threshold_grad: Optional[float] = 0) -> numpy.ndarray:
    """
    Mask values above thershold
    Args:
        dic_in (dict): Dictionary that contains values
        nvar (str): Variable name to process
        threshold_vel: Threshold value for velocity
        threshold_grad: Threshold value for tracer gradient
    Returns
        Array with masked values
    """
    _arr = numpy.array(dic_in[nvar])
    _arr[_arr == 0] = numpy.nan
    _mask = numpy.where(_arr > 0.99)
    _arr[_mask[0]] = numpy.nan
    _mask = numpy.where(numpy.array(dic_in['vel']) <= threshold_vel)
    _arr[_mask[0]] = numpy.nan
    _mask = numpy.where(numpy.array(dic_in['gradient_sst']) <= threshold_grad)
    _arr[_mask[0]] = numpy.nan
    return _arr


def process_list(dic_directory: dict, list_var: list,
                 box: list, nvar: str, dbox: float,
                 start: datetime.datetime, stop: datetime.datetime,
                 flag_front_threshold: Optional[float] = 1,
                 thresh_perc: Optional[float] = 0.2,
                 threshold_vel: Optional[float] = 0.1,
                 threshold_grad: Optional[float] = 0.1,) -> dict:
    """
    Process list  of front
    Args:
        dic_directory (list): list of directories where front files are
            stored
        list_var (list): list of model that have been used to compute metrics
        box (list): Domain to compute statistics,
            [lon_min, lon_max, lat_min, lat_max]
        start (datetime.datetime): first date for analyses
        stop (datetime.datetime): first date for analyses
        flag_front_threshold (float): Optional, default is 1,
        thresh_perc (float): Optional, minimum percentage of values to consider
             box, default is 0.2,
        threshold_vel (float): Optional, minimum velocity to perform analyses,
             default is 0.1,
        threshold_grad (float): Optional, minimum tracer gradient to perform
            analyses,default is 0.1,
    Returns
        Dictionary that contains statisticson fronts
    """
    dic_flatten_dic = {}
    cpt_file = 0
    for bn, idir in dic_directory.items():
        cpt_file += 1
        result = flatten_list(idir, list_var, start, stop,
                              flag_front_threshold=flag_front_threshold)
        if result is None:
            continue

        if not result.keys():
            logger.error(f'error for file {bn} in {idir}')
        result[nvar] = mask_values(result, nvar, threshold_vel=threshold_vel,
                                   threshold_grad=threshold_grad)
        dic_flatten_dic[bn] = result.copy()
        # from matplotlib import pyplot
        # pyplot.hist(result[nvar][~numpy.isnan(result[nvar])], alpha=0.3)
        # pyplot.savefig(f'hist_{bn}_{nvar}.png')
        stat = compute_box(result, box, nvar, dbox)
        dic_flatten_dic[f"stat_{bn}"] = stat.copy()
    lkey = list(dic_directory.keys())
    find_min = True
    if 'cos' in nvar or 'vectorial_product' in nvar:
        find_min = False
    _result = compare_nvar(dic_flatten_dic, lkey, nvar, find_min=find_min)
    dic_flatten_dic["comparison"] = _result.copy()
    logger.info('compute comparison')
    #dic_flatten_dic["comparison"][nvar][~numpy.isnan(dic_flatten_dic["comparison"][nvar])])
    stat = compute_box(dic_flatten_dic["comparison"], box, nvar, dbox,
                       compare=True, thresh_perc=thresh_perc)
    dic_flatten_dic["stat_comparison"] = stat.copy()
    logger.info('compute differences')
    for key in dic_flatten_dic['comparison'][f'{nvar}_rank'].keys():
        stat = compute_box(dic_flatten_dic['comparison'], box, f'{nvar}_rank',
                           dbox, key=key)
        dic_flatten_dic[f"stat_{key}_rank"] = stat.copy()
    for key in dic_flatten_dic['comparison'][f'{nvar}_diff'].keys():
        stat = compute_box(dic_flatten_dic['comparison'], box, f'{nvar}_diff',
                           dbox, key=key)
        dic_flatten_dic[f"stat_{key}_diff"] = stat.copy()

    return dic_flatten_dic


def flatten_list(list_file: list, list_var: list, start: datetime.datetime,
                 stop: datetime.datetime,
                 flag_front_threshold: Optional[float] = 1) -> dict:
    """
    Aggregate all fronts from several files
    Args:
        list_file (list): Input file name
        list_var (List): list of variables to process
        start (datetime.datetime): first date for analyses
        stop (datetime.datetime): first date for analyses
        flag_front_threshold (float): Optional, default is 1,
    Returns:
        Dictionary with all the front points
    """
    ldic = {}
    fdic = {}
    for ifile in list_file:
        _dic = flatten_data(ifile, list_var, start, stop,
                            flag_front_threshold=flag_front_threshold)
        if _dic is None:
            continue
        if not _dic:
            continue
        if 'lon' not in _dic.keys():
            continue

        for key, value in _dic.items():
            if key not in ldic.keys():
                ldic[key] = list(value)
            else:
                ldic[key].append(list(value))
    for key, value in ldic.items():
        fdic[key] = []
        for i in range(len(value)):
            if type(value[i]) is list:
                for j in range(len(value[i])):
                    fdic[key].append(value[i][j])
            else:
                fdic[key].append(value[i])
    return fdic


def flatten_data(ifile: str, list_var: list, start: datetime.datetime,
                 stop: datetime.datetime,
                 flag_front_threshold: Optional[float] = 1) -> dict:
    """
    Aggregate front coordinates from a file
    Args:
        ifile (str): Input file name
        list_var (List): list of variables to process
        start (datetime.datetime): first date for analyses
        stop (datetime.datetime): first date for analyses
        flag_front_threshold (float): Optional, default is 1,
    Returns:
        Dictionary with all the front points

    """
    flatten_dic = {}
    for key in list_var:
        flatten_dic[key] = []
    #dico_fronts = retrieve_dico
    try:
        with open(ifile, 'rb') as pickle_in:
            dico_fronts = pickle.load(pickle_in)
    except:  # EOFError :
        logger.error(f'Error opening {ifile}')
        return None
    if len(dico_fronts['lon']) == 0:
        return None
    try:
        start_dt = datetime.datetime.strptime(str(dico_fronts['time_coverage_start'].decode()),
                                              '%Y%m%dT%H%M%SZ')
    except:
        start_dt = dico_fronts['time_coverage_start']

    try:
        stop_dt = datetime.datetime.strptime(str(dico_fronts['time_coverage_end'].decode()),
                                             '%Y%m%dT%H%M%SZ')
    except:
        stop_dt = dico_fronts['time_coverage_end']
    if start_dt > stop or stop_dt < start:
        return None
    for key in list_var:
        for i in range(len(dico_fronts[key])):
            if len(dico_fronts['lon'][i]) != len(dico_fronts['scalar_product'][i]):
                continue
            if dico_fronts['flag_front'][i] > flag_front_threshold:
                continue
            for j in range(len(dico_fronts[key][i])):
                flatten_dic[key].append(dico_fronts[key][i][j])
    return flatten_dic


def compute_box(flatten_dic: dict, box: list, nvar: str, dbox: float,
                compare: Optional[bool] = False,
                thresh_perc: Optional[float] = 0.55,
                key: Optional[str] = None) -> dict:
    """
    Compute statistics on box
    Args:
        flatten_dic (dict): Dictionary that contains coordinate points of
            fronts
        box (list): Domain to compute statistics,
            [lon_min, lon_max, lat_min, lat_max]
        nvar (str): Name of variables
        dbox (float): Size of the box to compute statistics
        compare(bool): Optional, true to compute comparison, default is true
        thresh_perc (float): Minimum percentage of best data, default is 0.55
        key (str): Name of variable
    Returns:
        dictionary with statistics in box

    """
    lon = numpy.arange(box[0], box[1] + dbox, dbox)
    lat = numpy.arange(box[2], box[3] + dbox, dbox)
    dic_result = {}
    if compare is True:
        list_key = ['best', 'perc_best', 'nbpoints']
    else:
        list_key = ['mean', 'std', 'median', 'nbpoints', 'min', 'max']
    for lkey in list_key:
        dic_result[lkey] = numpy.full((len(lat), len(lon)), numpy.nan)
    dic_result['lon_bin'] = lon.copy()
    dic_result['lat_bin'] = lat.copy()
    lons = numpy.array(flatten_dic['lon']).ravel()
    lats = numpy.array(flatten_dic['lat']).ravel()
    for ilon, _lon in enumerate(lon):
        for ilat, _lat in enumerate(lat):
            # normu = numpy.array(flatten_dic['vel']).ravel()
            ix = numpy.where((lons >= _lon - dbox/2) & (lons <= _lon + dbox/2)
                             & (lats >= _lat - dbox/2)
                             & (lats <= _lat + dbox+2)
                             )
            # if ix[0].any():
            if len(ix[0]) > 5:
                _sel = numpy.array(ix[0].astype(int))
                if type(flatten_dic[nvar]) is dict and key is not None:
                    _tmp_array = flatten_dic[nvar][key].copy()
                else:
                    _tmp_array = flatten_dic[nvar].copy()
                _array = numpy.array(_tmp_array)[_sel]
                _list = list(_array[~numpy.isnan(_array)])
                # _list = list(_array)
                if compare is True:
                    # c = Counter(list(numpy.array(flatten_dic[nvar])[_sel]))
                    # _mean = c.most_common(1)[0][0]
                    if len(_list):
                        _best = max(set(_list), key=_list.count)
                        _perc = _list.count(_best) / len(_list)
                        if _perc < thresh_perc:
                            _best = -1
                        _nbp = len(_list)
                    else:
                        _best = numpy.nan
                        _perc = numpy.nan
                        _nbp = 0
                    # print(_best, _list)
                    dic_result['best'][ilat, ilon] = _best
                    dic_result['perc_best'][ilat, ilon] = _perc
                    dic_result['nbpoints'][ilat, ilon] = _nbp
                else:
                    _array[_array == 0] = numpy.nan
                    _array[_array == 0] = numpy.nan
                    if len(_list):
                        _mean = numpy.nanmean(_array)
                        _min = numpy.nanmin(_array)
                        _max = numpy.nanmax(_array)
                        _median = numpy.nanmedian(_array)
                        _std = numpy.nanstd(_array)
                        _nbp = len(_list)
                    else:
                        _mean = numpy.nan
                        _median = numpy.nan
                        _std = numpy.nan
                        _min = numpy.nan
                        _max = numpy.nan
                        _nbp = 0

                    dic_result['mean'][ilat, ilon] = _mean
                    dic_result['min'][ilat, ilon] = _min
                    dic_result['max'][ilat, ilon] = _max
                    dic_result['median'][ilat, ilon] = _median
                    dic_result['std'][ilat, ilon] = _std
                    dic_result['nbpoints'][ilat, ilon] = _nbp
    return dic_result


def read_netcdf_stat(file: str)-> dict:
    dic = {}
    if not os.path.exists(file):
        logger.error(f'file {file} not found')
        sys.exit(1)
    fid = netCDF4.Dataset(file, 'r')
    for key, value in fid.variables.items():
        dic[key] = value[:]
    fid.close()
    return dic


def plot_difference(dic: dict, config_json: str, size: Optional[int] = 4,
                    proj: Optional = None, box: Optional[list] = None,
                    file_out: Optional[str] = 'plot_difference.png',
                    diag: Optional[str] = 'diff_mean',)-> pyplot.figure():
    from mpl_toolkits.axes_grid1 import ImageGrid

    par = load_fronts_stat_parameters(config_json)
    list_data = list(par.keys())
    list_data.remove("global")
    lon = dic["lon"]
    lat = dic["lat"]
    if box is None:
        box = (numpy.min(lon), numpy.max(lon),
               numpy.min(lat), numpy.max(lat))
    n = len(list_data)

    fig = pyplot.figure(figsize=(n * size, n * size))

    if proj is not None:
        import cartopy

        from cartopy.mpl.geoaxes import GeoAxes
        from mpl_toolkits.axes_grid1 import AxesGrid

        axes_class = (GeoAxes, dict(projection=proj))
        ax = AxesGrid(fig, 111, axes_class=axes_class,
                      nrows_ncols=(n, n-1), axes_pad=0.8,
                      cbar_location='bottom', cbar_mode='each',
                      cbar_pad="2%", cbar_size='7%', label_mode='')
    else:
        from mpl_toolkits.axes_grid1 import ImageGrid

        ax = ImageGrid(fig, 111,  # similar to subplot(111)
                       nrows_ncols=(n, n-1),  # creates 2x2 grid of Axes
                       axes_pad=0.8,  # pad between Axes in inch.
                       cbar_location="right", cbar_mode="each",
                        cbar_size="7%", cbar_pad="2%"
                    )
    ind = 0
    #pyplot.title('Statistics Mean and STD')
    pref = 'stat'
    for i, key in enumerate(list_data):
        for j, key2 in enumerate(list_data):
            if i == j:
                #ind += 1
                continue
            key_all = f'{pref}_{key}_{key2}_{diag}'
            if key_all not in dic.keys():
                dic[key_all] = - dic[f'stat_{key2}_{key}_{diag}']
            vmin = -0.5
            vmax = 0.5
            cmap = 'seismic'
            pref = 'stat'

            if proj is not None:
                gl = init_cartopy(ax[ind], box)
                c = ax[ind].pcolor(lon, lat, dic[key_all][0],
                                    vmin=vmin, vmax=vmax,
                                    cmap=cmap, transform=proj)
            else:
                c = ax[ind].pcolor(lon, lat, dic[key_all][0],
                                    vmin=vmin, vmax=vmax, cmap=cmap)
            ax[ind].set_title(f'{key} - {key2}')
            ax.cbar_axes[ind].colorbar(c)
            ind += 1
    pyplot.savefig(file_out)
    fig2, ax = pyplot.subplots(1, 1, figsize=(size+1, size),
                              subplot_kw=dict(projection=proj))
    _bn = os.path.splitext(file_out)
    if proj is not None:
        gl = init_cartopy(ax, box)
        c = ax.pcolor(lon, lat, dic[f'stat_{key}_nbpoints'][0],
                                   vmin=0, vmax=500,
                                   cmap='seismic', transform=proj)
    else:
        c = ax.pcolor(lon, lat, dic[f'stat_{key}_nbpoints'][0],
                                    vmin=0, vmax=400,
                                    cmap='seismic')

    ax.set_title(f'Number of points')
    pyplot.colorbar(c)
    pyplot.savefig(f'{_bn[0]}_nbofpoints.png')
    return fig, fig2


def plot_list_diag(dic: dict, config_json: str, size: Optional[int] = 4,
                   proj: Optional = None, box: Optional[list] = None,
                   file_out: Optional[str] = 'plot_stats.png',
                   list_diag: Optional[list] = ['mean', 'std', 'rank_mean'],
                   )-> pyplot.figure():

    par = load_fronts_stat_parameters(config_json)
    list_data = list(par.keys())
    list_data.remove("global")
    lon = dic["lon"]
    lat = dic["lat"]
    if box is None:
        box = (numpy.min(lon), numpy.max(lon),
               numpy.min(lat), numpy.max(lat))
    n = len(list_data)
    ndiag = len(list_diag)
    fig = pyplot.figure(figsize=(n*size, ndiag*size))
    if proj is not None:
        import cartopy

        from cartopy.mpl.geoaxes import GeoAxes
        from mpl_toolkits.axes_grid1 import AxesGrid

        axes_class = (GeoAxes, dict(projection=proj))
        ax = AxesGrid(fig, 111, axes_class=axes_class,
                      nrows_ncols=(n, ndiag), axes_pad=0.8,
                      cbar_location='bottom', cbar_mode='each',
                      cbar_pad="2%", cbar_size='7%', label_mode='')
    else:
        from mpl_toolkits.axes_grid1 import ImageGrid

        ax = ImageGrid(fig, 111,  # similar to subplot(111)
                       nrows_ncols=(n, ndiag),  # creates 2x2 grid of Axes
                       axes_pad=0.8,  # pad between Axes in inch.
                       cbar_location="right", cbar_mode="each",
                       cbar_size="7%", cbar_pad="2%"
                 )
    #pyplot.title('Statistics Mean and STD')
    _cmap = 'jet'
    color = []
    ind = 0
    for i, key in enumerate(list_data):
        for j, diag in enumerate(list_diag):
            vmin = 0
            vmax = 0.8
            cmap = 'jet'
            pref = 'stat'
            if 'rank' in diag:
                vmax = n
                cmap = 'RdYlGn_r'
                pref = 'stat_rank'
            if proj is not None:
                gl = init_cartopy(ax[ind], box)
                c = ax[ind].pcolor(lon, lat, dic[f'{pref}_{key}_{diag}'][0],
                                       vmin=vmin, vmax=vmax,
                                       cmap=cmap, transform=proj)
            else:
                c = ax[ind].pcolor(lon, lat, dic[f'{pref}_{key}_{diag}'][0],
                                vmin=vmin, vmax=vmax,
                                cmap=cmap)
            ax[ind].set_title(f'{diag} for {key}')
            ax.cbar_axes[ind].colorbar(c)
            ind += 1

    # pyplot.savefig(file_out)
    return fig


def run_plot(file: str, config_json: str, size: Optional[int] = 4,
             dir_out: Optional[str] = './',
             proj:Optional[str] = None,):
    dic = read_netcdf_stat(file)
    fo = os.path.splitext(os.path.basename(file))[0]
    os.makedirs(dir_out, exist_ok=True)
    file_out = os.path.join(dir_out, f'{fo}_stat.png')
    fig1 = plot_list_diag(dic, config_json, size=size,
                          proj=proj, file_out=file_out,
                          list_diag=['mean', 'std', 'rank_mean'])
    file_out = os.path.join(dir_out, f'{fo}_diff.png')

    fig2, fig3 = plot_difference(dic, config_json, size=size,
                           proj=proj, file_out=file_out,
                           diag='diff_mean')
    return fig1, fig2, fig3


def plot_diag_auto(dic_all: dict, nvar: str, box: list, output: str,
                   nx: Optional[list] = (2, 1), vmin: Optional[float] = 0,
                   vmax: Optional[float] = 0.8, cmap: Optional[str] = 'jet',
                   proj: Optional[str] = None):
    """
    plot statistics
    Args
        dic_all (dict): Dictionary with statistics
        nvar (str): Variable name
        box (list): Domain, [lon_min, lon_max, lat_min, lat_max]
        nx (list):number of  lines and columns, optional, default is (2, 1)
        vmin (float): Minimum value for colorbar, optional, default is 0
        vmax (float): Maximum value for colorbar, optional, default is 0.8
        cmap (str): Name of the colorbar, optional, default is jet
        proj (str): Name of the projection, optional, default is None
    """
    from matplotlib import pyplot
    ysize = nx[0]*2
    fig, ax = pyplot.subplots(nx[0], nx[1], figsize=(20, ysize),
                              subplot_kw=dict(projection=proj))
    i = 0
    gl = []

    lon = dic_all['stat_comparison']['lon_bin']
    lat = dic_all['stat_comparison']['lat_bin']
    for key, dic_result in dic_all.items():
        j = 0
        if 'stat' not in key:
            continue
        for subkey in dic_result.keys():
            _cmap = 'jet'
            _vmin = vmin  # numpy.nanmin(dic_result[subkey])
            if 'max' in subkey:
                _vmin = 0.8
            _vmax = vmax  # numpy.nanmax(dic_result[subkey])
            if 'min' in subkey:
                _vmax = 0.01
            if 'lon' in subkey or 'lat' in subkey:
                continue
#   pyplot.figure(figsize=(10, 10))
            if proj is not None:
                _gl = init_cartopy(ax[i][j], box)
                gl.append(_gl)
            if 'diff' in key:
                _cmap = 'seismic'
                _vmin = -0.4  # numpy.nanmin(dic_result[nvar])
                _vmax = 0.41  # numpy.nanmax(dic_result[nvar])
                if 'min' in subkey or 'max' in subkey:
                    _vmin = numpy.nanmin(dic_result[nvar])
                    _vmax = numpy.nanmax(dic_result[nvar])

            if 'best' in subkey:
                _cmap = 'seismic'
                _vmin = numpy.nanmin(dic_result[subkey])
                _vmax = numpy.nanmax(dic_result[subkey])
            if 'rank' in key:
                _cmap = 'RdYlGn_r'
                _vmin = 1
                _vmax = 3.0
            elif 'nbpoints' in subkey:
                _cmap = 'seismic'
                _vmin = 0
                _vmax = 500
            try:
                if proj is None:
                    c = ax[i][j].pcolor(lon, lat, dic_result[subkey],
                                        vmin=_vmin, vmax=_vmax,
                                        cmap=_cmap)
                else:
                    c = ax[i][j].pcolor(lon, lat, dic_result[subkey],
                                        vmin=_vmin, vmax=_vmax,
                                        cmap=_cmap, transform=proj)
            except:
                continue
            ax[i][j].set_title(f'{key} {subkey}')
            pyplot.colorbar(c, ax=ax[i][j])
            j += 1

        i += 1
    fig.suptitle(os.path.basename(output))
    pyplot.savefig(output)
    return fig

def save_diag(dic_all: dict, nvar: str, box: list, dbox: int, output: str,
              listmodel: str, start: datetime.datetime,
              stop: datetime.datetime,  netcdf: Optional[bool] = True):
    """
    Save statistics in netcdf file
    Args:
        dic_all (dict): input dictionnary that contains statistics
        nvar (str): Name of variable
        box (list): Domain (lon_min, lon_max, lat_min, lat_max)
        dbox (float): Box size
        output (str): Output file name
        listmodel
        start (datetime.datetime): First time of analyse
        stop (datetime.datetime): Last time of analyse
        isidf (bool): Optional, save in netcdf format, default is False
    """
    fid = netCDF4.Dataset(output, 'w')
    _lon = dic_all['stat_comparison']['lon_bin']
    _lat = dic_all['stat_comparison']['lat_bin']
    _ = fid.createDimension('lon', len(_lon))
    _ = fid.createDimension('lat', len(_lat))
    _ = fid.createDimension('time', None)
    lat = fid.createVariable('lat', 'f4', ('lat',))
    lat.long_name = "Latitude"
    lat.standard_name = "latitude"
    lat.units = "degrees_north"
    lat[:] = _lat
    lon = fid.createVariable('lon', 'f4', ('lon',))
    lon.long_name = "Longitude"
    lon.standard_name = "longitude"
    lon.units = "degrees_east"
    lon[:] = _lon

    for key, dic_result in dic_all.items():
        if 'stat' not in key:
            continue

        for subkey in dic_result.keys():
            if 'lon' in subkey or 'lat' in subkey:
                continue
            var = + dic_result[subkey]
            nan_mask = ((numpy.ma.getmaskarray(var)) | (numpy.isnan(var)))
            var[nan_mask] = -1.36e9

            bvar = fid.createVariable(f'{key}_{subkey}', 'f4',
                                      ('time', 'lat', 'lon'),
                                      fill_value=-1.36e9)
            bvar[0, :, :] = var
            bvar.units = ""
            if 'best' in subkey:
                bvar.long_name = f"Best model, {listmodel}"
            else:
                bvar.long_name = f"{subkey} of flow over fronts for {key}"

    fid.cdm_data_type = "Grid"
    fid.idf_version = "1.0"
    fid.idf_granule_id = output
    fid.time_coverage_start = start.strftime(FMT)
    fid.time_coverage_end = stop.strftime(FMT)
    fid.idf_subsampling_factor = 0
    fid.idf_spatial_resolution = 8880.203
    fid.idf_spatial_resolution_units = "m"
    fid.comment = "..."
    fid.product_version = "1.0"
    fid.processing_level = "L4"
    fid.close()


def init_cartopy(ax, box: Optional[list] = [-180, 180, -90, 90]):
    """
    Initialize cartopy map
    Args:
        ax (matplotlib axe)
        box: Domain (lon_min, lon_max, lat_min, lat_max), optional, default is
         global
    """
    import cartopy
    # projection = cartopy.crs.Mercator()
    ax.add_feature(cartopy.feature.LAND, zorder=3)
    ax.add_feature(cartopy.feature.COASTLINE, zorder=3)
    ax.add_feature(cartopy.feature.LAKES, alpha=0.5, zorder=3)
    ax.add_feature(cartopy.feature.RIVERS, zorder=3)
    ax.set_extent([box[0], box[1], box[2], box[3]])
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=2,
                      alpha=0.5, color='gray')
    gl.xlabels_top = False
    gl.ylabels_left = False
    # gl.xlocator = mticker.FixedLocator([-180, -45, 0, 45, 180])
    return gl


def compare_nvar(dic_all: dict, list_key: list, nvar: str,
                 find_min: Optional[bool] = True) -> dict:
    """
    Compare statistics to make ranking
    Args:
        dic_all (dict): Input dictionnary with statistics
        list_key (list): Keys to process
        nvar (str): Variable
        find_min (bool): True
    Returns:
        Update dictionary with ranking
    """
    _len = len(dic_all[list_key[0]][nvar])
    if _len != len(dic_all[list_key[1]][nvar]):
        logger.critical(f'ERROR LENGTH {_len} {len(dic_all[list_key[1]][nvar])}')
        sys.exit(1)
    list_ind = []
    list_ind = numpy.full((_len), numpy.nan)
    list_diff = {}  # numpy.full((_len), numpy.nan)
    list_rank = {}
    for key in list_key:
        list_rank[f'rank_{key}'] = numpy.full((_len), numpy.nan)
    range_key = numpy.arange(len(list_key))
    for lk in range_key[:-1]:
        for j in range_key[lk + 1:]:
            _key = f'{list_key[range_key[lk]]}_{list_key[range_key[j]]}'
            list_diff[_key] = numpy.full((_len), numpy.nan)
    for i in range(_len):
        _list = numpy.full((len(list_key)), numpy.nan)
        for k, key in enumerate(list_key):
            try:
                _list[k] = abs(dic_all[key][nvar][i])
                #if _list[k] <= 1E-4 or _list[k] > 0.9:
                #    _list[k] = numpy.nan
            except:
                _list[k] = numpy.nan
        if numpy.isnan(_list).any():
            _ind = numpy.nan
            for key in list_key:
                list_rank[f'rank_{key}'][i] = numpy.nan
        else:
            _sort = scipy.stats.rankdata(_list)
            for k, key in enumerate(list_key):
                list_rank[f'rank_{key}'][i] = _sort[k]
            if find_min is True:
                _ind = numpy.argmin(_list)
            else:
                _ind = numpy.argmax(_list)
        for lk in range_key[:-1]:
            for j in range_key[lk + 1:]:
                _key = f'{list_key[range_key[lk]]}_{list_key[range_key[j]]}'
                list_diff[_key][i] = _list[lk] - _list[j]
        list_ind[i] = _ind
    dic_out = {'lon': dic_all[list_key[0]]['lon'],
               'lat': dic_all[list_key[0]]['lat'], nvar: list_ind,
               f'{nvar}_rank': list_rank, f'{nvar}_diff': list_diff}
    return dic_out


def list_files(_par: dict, _start: datetime.datetime, _stop: datetime.datetime
               ) -> dict:
    dic_dir = {}
    for key in _par.keys():
        if key == 'global':
            continue
        _dir = _par[key]['input_path']
        ref_name = _par[key]['reference_name']
        if ref_name is None:
            dic_dir[key] = [_par[key]['input_path'], ]
            continue
        pattern = ref_name + "(.*?)" 
        pattern = pattern + "(?P<year>\\d{4})(?P<month>\\d{2})(?P<day>\\d{2})T(?P<hour>\\d{2})\\d{4}.pyo"
        regex = re.compile(pattern).search
        _files = []
        _dt = _par[key]['time_step']
        for dir_path, _, filenames in os.walk(_dir):
            for filename in filenames:
                match = regex(filename)
                if match:
                    year = int(match.group("year"))
                    month = int(match.group("month"))
                    day = int(match.group("day"))
                    hour = int(match.group("hour"))
                    _date = datetime.datetime(year, month, day, hour)

                    if _date > _start and _date < _stop:
                        _files.append(os.path.join(dir_path, filename))
        dic_dir[key] = _files
    return dic_dir


def run(config_json: str, dbox: float,
        first_date: Optional[str] = '19000101T000000Z',
        last_date: Optional[str] = '20500101T000000Z',
        number_of_days: Optional[int] = None,
        output_dir: Optional[str] = './',
        plot: Optional[bool] = False) -> dict:

    par = load_fronts_stat_parameters(config_json)
    _fmt = '%Y%m%dT%H%M%SZ'
    dic_list = {}
    if number_of_days is None:
        start = datetime.datetime.strptime(first_date, _fmt)
        stop = datetime.datetime.strptime(last_date, _fmt)
        _dic = box_metrics(par, dbox, start, stop,
                           output_dir=output_dir, isplot=plot)
        dic_list[f'{first_date}_{last_date}'] = _dic
    else:
        list_day = numpy.arange(0, (start - stop).days, number_of_days)
        list_start = [start + datetime.timedeltas(days=x) for x in list_day]
        for _start in list_start:
            _stop = _start + datetime.timedeltas(days=number_of_days)
            _first_date = _start.strftime(_fmt)
            _last_date = _stop.strfitime(_fmt)
            _dic = box_metrics(par, dbox, _start, _stop,
                               output_dir=output_dir, isplot=plot)
            dic_list[f'{_first_date}_{_last_date}'] = _dic
    return dic_list


def box_metrics(par: dict, dbox: float, start: datetime.datetime,
                stop: datetime.datetime, output_dir: Optional[str] = './',
                isplot: Optional[bool] = False) -> None:
    """"
    Parse json config file and compute metrics on boxes
    """

    box = par['global']['box']

    list_var = par['global']['list_var']
    box = par['global']['box']
    nvar = par['global']['nvar']
    pattern = par['global']['pattern']
    thperc = par['global'].setdefault('percentage_threshold', 0.3)
    flag_threshold = par['global']['flag_threshold']

    dic_dir = list_files(par, start, stop)

    dic_list = process_list(dic_dir, list_var, box, nvar, dbox, start, stop,
                            flag_front_threshold=flag_threshold,
                            thresh_perc=thperc,
                            threshold_vel=par['global']['velocity_threshold'],
                            threshold_grad=par['global']['gradient_threshold'])
    _list = list(dic_dir.keys())
    _listmodel = [f"{i}: {key}" for i, key in enumerate(_list)]
    listmodel = ", ".join(_listmodel)
    os.makedirs(output_dir, exist_ok=True)

    ref_start = start.strftime(FMT)
    ref_stop = stop.strftime(FMT)
    _dirout = f"{pattern}_{ref_start}_{ref_stop}"
    #_dout = os.path.join(output_dir, f'{_dirout}_mean')
    logger.info('save diags')
    os.makedirs(output_dir, exist_ok=True)
    _out = os.path.join(output_dir, f"{_dirout}_mean.nc")
    save_diag(dic_list, 'mean', box, dbox, _out, listmodel,
              start, stop)
    if isplot is True:
        out_plot = output_dir
        os.makedirs(out_plot, exist_ok=True)
        logger.info('Plot diags')
        ldiff = [(len(_list) - k) for k in range(1, len(_list))]
        nx = len(_list) + 1 + sum(ldiff) + len(_list)
        _out = os.path.join(out_plot, f"{_dirout}_mean.png")
        _ = plot_diag_auto(dic_list, 'mean', box, _out, nx=(nx, 6),
                  vmin=0.1, vmax=0.7, cmap='jet', proj=None)

    return dic_list


if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('config_json', type=str,  # nargs='+',
                        help='Json description of statistic to consider')
    parser.add_argument('--degree_box', default=2, type=float,
                        help='Size of box to compute statistics')
    parser.add_argument('--first_date', default='19900101T000000Z', type=str,
                        required=False,
                        help='First time considered for analyses')
    parser.add_argument('--last_date', default='20500101T000000Z', type=str,
                        required=False,
                        help='Last time considered for analyses')
    parser.add_argument('--days', default=None, type=int,
                        required=False,
                        help='Number of days considered to compute statistics')
    parser.add_argument('--out', dest='outdir',
                        type=str, default='./',
                        help='Path for output figure and python dictionnary')
    parser.add_argument('--verbose', action='store_true', default=False,
                        required=False)
    parser.add_argument('--debug', action='store_true', default=False,
                        required=False)
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.INFO)
    if args.debug:
        logger.setLevel(logging.DEBUG)

    dic_list = run(args.config_json, args.degree_box,
                   first_date=args.first_date,
                   last_date=args.last_date, output_dir=args.outdir,
                   number_of_days=args.days, plot=True)
    # _ = plot(dic_spectrum, os.path.join(args.output_dir, 'spectrum.png'))
