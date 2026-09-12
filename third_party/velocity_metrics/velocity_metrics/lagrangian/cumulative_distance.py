# vim: ts=4:sts=4:sw=4
#
# @author lucile.gaultier@oceandatalab.com
# @date 2023-09-01
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
Compute Separation Distance Statistics between in-situ
and fictive particule trajectory
"""

import netCDF4
import numpy
import matplotlib
# import cartopy
import glob
import os
import sys
import datetime
from matplotlib import pyplot
import pickle
import gzip
from tqdm import tqdm
from typing import Optional, Tuple
import logging
logger = logging.getLogger(__name__)


def run(input_path: str, drifter_file: str, output_dir: Optional[str] = './',
        output_filename: Optional[str] = None, isplot: Optional[bool] = False,
        ) -> dict:
    dic = process_all_ide(input_path, drifter_file, output_dir,
                          output_filename, isplot=isplot)
    return dic


def plot(list_pickle: list, output_dir: Optional[str] = './',
         list_color: Optional[list] = None,
         output_filename: Optional[str] = 'Sde.png',
         plot_histogram: Optional[bool] = False,
         variable: Optional[str] = 'score',
         alpha: Optional[float] = 0.5,
         plot_range: Optional[dict] = None,
         ):
    outfile = os.path.join(output_dir, output_filename)

    fig1 = plot_all_sde(list_pickle, outfile, list_color=list_color,
                        var=variable, alpha=alpha, plot_range=plot_range)
    fig2 = None
    if plot_histogram is True:
        fig2 = plot_all_sde_hist(list_pickle, outfile, list_color=list_color)
    return fig1, fig2


def plot_all_sde_hist(list_pickle_sde: list, fileout: str,
                      list_color: Optional[list] = None) -> pyplot.figure():
    from mpl_toolkits.axes_grid1 import ImageGrid
    col = 0
    pyplot.figure()
    maxlen = max(len(list_pickle_sde), 10)
    cm = matplotlib.pyplot.cm.tab20(numpy.linspace(0, 1, maxlen))
    fig = pyplot.figure(figsize=(20, 20))
    n0 = 2
    n1 = 30
    step = 3

    ax = ImageGrid(fig, 111,  # similar to subplot(111)
                   nrows_ncols=(3, 4),  # creates 2x2 grid of Axes
                   axes_pad=0.8,  # pad between Axes in inch.
                   cbar_location="right", cbar_mode="each",
                   cbar_size="7%", cbar_pad="2%")
    # t = numpy.arange(n0, n)
    for i, ifile in enumerate(list_pickle_sde):
        ind = 0
        if 'gz' in os.path.splitext(ifile)[-1]:
            with gzip.open(ifile, 'rb') as f:
                dic = pickle.load(f)
        else:
            with open(ifile, 'rb') as f:
                dic = pickle.load(f)
        if len(list(dic.keys())) == 0:
            logger.info(f'File {ifile} is empty')
            continue
        dic_mean = merge_hist(dic)
        if list_color is not None:
            if len(list_pickle_sde) > i:
                dic_mean['color'] = list_color[i]
        if 'color' not in dic_mean.keys():
            dic_mean['color'] = col
        col = col + 1
        n = min(len(dic_mean['time'][n0:]), n1)
        t = dic_mean['time'][n0:n]
        if not isinstance(dic_mean['color'], str):
            dic_mean['color'] = cm[col]
        for it in range(n0, n, step):
            ax[ind].bar(dic_mean['binsde'][it, :-1],
                       dic_mean['histogramsde'][it, :],
                       width=numpy.diff(dic_mean['binsde'][it, :]),
                       edgecolor="black",
                       align="edge", alpha=0.3, color=dic_mean['color'],
                       label=dic_mean['label'])
            ind += 1

    for ind, iax in enumerate(ax):
        tind = ind * step + n0
        if tind < len(t):
            iax.set(ylabel='Normalised cumulative distance (km)',
                    title=f'{t[tind]} days of advection')
            iax.legend()

    pyplot.savefig(fileout)
    return fig


def plot_all_sde(list_pickle_sde: list, fileout: str,
                 list_color: Optional[list] = None,
                 var: Optional[str] = 'sde', plot_range: Optional[dict] = None,
                 alpha: Optional[float] = 0.5) -> pyplot.figure():
    col = 0
    pyplot.figure()
    maxlen = max(len(list_pickle_sde), 10)
    cm = matplotlib.pyplot.cm.tab20(numpy.linspace(0, 1, maxlen))
    fig, ax = pyplot.subplots(nrows=2, ncols=2, figsize=(10, 10))
    n0 = 4
    n1 = 250
    # t = numpy.arange(n0, n)
    if plot_range is None:
        if var == 'score':
            plot_range = {'mean': (0.7, 0.85), 'max': (0.8, .95),
                          'min': (0.6, 0.75)}
        else:
            plot_range = {'mean': (0, 15), 'max': (0, 20),
                          'min': (0, 10)}
    for i, ifile in enumerate(list_pickle_sde):
        if 'png' in os.path.splitext(ifile)[-1]:
            continue
        if 'gz' in os.path.splitext(ifile)[-1]:
            with gzip.open(ifile, 'rb') as f:
                dic = pickle.load(f)
        else:
            with open(ifile, 'rb') as f:
                dic = pickle.load(f)
        if len(list(dic.keys())) == 0:
            logger.info(f'File {ifile} is empty')
            continue
        list_stat = []
        for key in ('mean', 'std', 'min', 'max', 'median'):
            list_stat.append(f'{key}{var}')
        #list_stat = ('meansde', 'stdsde', 'minsde', 'maxsde', 'mediansde')
        dic_mean = average_sde(dic, list_stat)
        if list_color is not None:
            if len(list_pickle_sde) > i:
                dic_mean['color'] = list_color[i]
        if 'color' not in dic_mean.keys():
            dic_mean['color'] = col
        col = col + 1
        n = min(len(dic_mean['time'][n0:]), n1)
        t = dic_mean['time'][n0:n]
        # if 'mean_sde' not in dic_mean.keys():
        #     logger.error(f'mean_sde key is not found in {file}')
        #     continue
        if not isinstance(dic_mean['color'], str):
            dic_mean['color'] = cm[col]
            ax[0][0].plot(t, dic_mean[f'mean{var}'][n0:n], c=dic_mean['color'],
                          label=dic_mean['label'], alpha=alpha)
            ax[0][0].set_title('(a) Mean ')
            ax[0][1].plot(t, dic_mean[f'min{var}'][n0:n], c=dic_mean['color'],
                          label=dic_mean['label'], alpha=alpha)
            ax[0][1].set_title('(b) Minimum ')
            ax[1][0].plot(t, dic_mean[f'std{var}'][n0:n], c=dic_mean['color'],
                          label=dic_mean['label'], alpha=alpha)
            ax[1][0].set_title('(c) Std')
            ax[1][1].plot(t, dic_mean[f'max{var}'][n0:n], c=dic_mean['color'],
                          label=dic_mean['label'], alpha=alpha)
            ax[1][1].set_title('(b) Maximum ')
        else:
            ax[0][0].plot(t, dic_mean[f'mean{var}'][n0:n], dic_mean['color'],
                          label=dic_mean['label'], alpha=alpha)
            ax[0][0].set_title('(a) Mean ')
            ax[0][1].plot(t, dic_mean[f'min{var}'][n0:n], dic_mean['color'],
                          label=dic_mean['label'], alpha=alpha)
            ax[0][1].set_title('(b) Minimum ')
            ax[1][0].plot(t, dic_mean[f'std{var}'][n0:n], dic_mean['color'],
                          label=dic_mean['label'], alpha=alpha)
            ax[1][0].set_title('(c) Std')
            ax[1][1].plot(t, dic_mean[f'max{var}'][n0:n], dic_mean['color'],
                          label=dic_mean['label'], alpha=alpha)
            ax[1][1].set_title('(b) Maximum ')
        if var == 'score':
            ax[1][1].set_ylim(plot_range['max'])
            ax[0][0].set_ylim(plot_range['mean'])
            ax[0][1].set_ylim(plot_range['min'])
    #ax[0][0].legend()
    #ax[1][0].legend()
    #ax[0][1].legend()
    ax[1][1].legend()

    for iax in ax.flat:
        ylabel = 'Normalized cumulative distance'
        if var == 'score':
            ylabel = 'Score'
        iax.set(xlabel='Days of advection',
                ylabel=ylabel)

    pyplot.savefig(fileout)
    return fig


def average_sde(dic: dict, list_stat: Optional[list] = None,
                list_attr: Optional = None) -> dict:
    '''Average Statistical parameters from dictionnary
    Input:
        dic (dict): dictionnary with all statistics
        list_stat (list): list of statistics names to average
        list_attr (list): List of attribute to transfer to the output
                  dictionary
    Returns:
        Dictionary with the averaged statistics for the listed statistics to
        process
    '''
    dic_arr = {}
    dici = {}
    if list_stat is None:
        list_stat = ('meansde', 'stdsde', 'minsde', 'maxsde', 'mediansde')
    if list_attr is None:
        list_attr = ('color', 'depth', 'label', 'data_type', 'time')
    for key, value in dic.items():
        for par in list_stat:
            if par not in value.keys():
                continue
            if par not in dic_arr.keys():
                dic_arr[par] = 0
                dici[par] = 0
            _tmp = value[par]
            _tmp = numpy.ma.array(_tmp, mask=(~numpy.isfinite(_tmp)))
            _tmp[_tmp.mask] = 0
            if 'score' in par:
                dic_arr[par] = dic_arr[par] + _tmp
                dici[par] = dici[par] + 1
            else:
                _tmp = numpy.ma.array(_tmp, mask=(_tmp == 0))
                n = len(_tmp)
                # if (_tmp[:20] ==999).any(): continue # = 0 #numpy.nan
                dic_arr[par] = dic_arr[par] + value[par]
                ni = numpy.ones((n))
                ni[_tmp.mask] = 0
                dici[par] = dici[par] + ni
        for attr in list_attr:
            if attr in value.keys():
                dic_arr[attr] = value[attr]
    for key in list_stat:
        if 'score' in key:
            dic_arr[key] = dic_arr[key][:] / dici[key]

        else:
            dici[key][dici[key] == 0] = 1
            dic_arr[key] = dic_arr[key][:] / dici[key]
    return dic_arr


def merge_hist(dic: dict, list_attr: Optional[list] = None) -> dict:
    '''Merge different histogram with same bins
    Input:
        dic (dict): dictionnary with all histogram
        list_attr (list): List of attribute to transfer to the output
                  dictionary
    Returns:
        Dictionary with the merged histograms, bins and list of attributes
    '''
    dic_arr = {}
    if list_attr is None:
        list_attr = ('color', 'depth', 'label', 'data_type', 'time')
    for key, value in dic.items():
        if 'histogramsde' not in value.keys():
            continue
        tmp = value['histogramsde']
        tmp = numpy.ma.array(tmp, mask=(~numpy.isfinite(tmp)))
        tmp[tmp.mask] = 0
        if 'histogramsde' not in dic_arr.keys():
            dic_arr['histogramsde'] = numpy.full(numpy.shape(tmp), numpy.nan)
        dic_arr['histogramsde'] += tmp
        dic_arr['binsde'] = value['binsde']
        for attr in list_attr:
            if attr in value.keys():
                dic_arr[attr] = value[attr]
    return dic_arr


def dist(lon1: numpy.ndarray, lat1: numpy.ndarray, lon2: numpy.ndarray,
         lat2: numpy.ndarray) -> numpy.ndarray:
    coslat = numpy.cos(numpy.deg2rad(lat1))
    dist = numpy.sqrt(((lon1 - lon2) * coslat)**2 + (lat1 - lat2)**2)
    return dist


def read_fictive_traj_netcdf(ifile: str, nstep: Optional[int] = 250):
    data = netCDF4.Dataset(ifile, 'r')
    dic_attr = {'ide': data.ide, 'data_type': data.data_type,
                'label': data.label, 'depth': data.depth}
    lon = data.variables['lon_hr'][:nstep, :]
    # try:
    lat = data.variables['lat_hr'][:nstep, :]
    mask = data.variables['mask_hr'][:nstep, :]
    _mask = ((abs(lon) > 360) | (abs(lat) > 90) | (mask == 1))
    lon[_mask] = numpy.nan
    lat[_mask] = numpy.nan
    time = data.variables['time_hr'][:nstep]
    data.close()
    
    return lon, lat, time, dic_attrread
def compute_nlcs(lon_d: numpy.ndarray, lat_d: numpy.ndarray,
                 time_d: numpy.ndarray, lon_f: numpy.ndarray,
                 lat_f: numpy.ndarray, time_f: numpy.ndarray) -> numpy.ndarray:
    # Compute Normalized Lagrangian Cumulative Separation
    lon_d_interp = numpy.interp(time_f, time_d, lon_d)
    lat_d_interp = numpy.interp(time_f, time_d, lat_d)
    dde = numpy.zeros(numpy.shape(lon_f))
    sde = numpy.zeros(numpy.shape(lon_f))
    # Distance between points in drifter
    lld = dist(lon_d_interp[1:], lat_d_interp[1:], lon_d_interp[:-1],
               lat_d_interp[:-1])

    len_time, len_pa = numpy.shape(lon_f)
    first_time = 2
    for pa in range(len_pa):
        # separation between drifter and fictive particule
        dde[:, pa] = dist(lon_d_interp[:], lat_d_interp[:], lon_f[:, pa],
                          lat_f[:, pa])
        for it in range(first_time, len_time-1):
            # Normalized cumulative separation
            #sde[it, pa] = sum(dde[2: it, pa]) / sum(lld[1: 
            # Normalized cumulative separation
            dde_sum = 0
            lld_sum = 0
            for i in range(first_time, it+1):
                dde_sum +=  dde[i-1, pa]
                lld_sum += sum(lld[first_time:i])
            #lld_sum += lld[it + i + 1]
            sde[it, pa] = dde_sum / lld_sum if lld_sum != 0 else numpy.nan
#            sde[it, pa] = sum(dde[2: it+1, pa]) / sum(lld[1: it+1])
            #if lld[it] < 5.e-6:
            #    sde[it, pa] = numpy.nan
    return sde, dde, lon_d_interp, lat_d_interp


def compute_statistics(sde: numpy.ndarray,
                       vbins: Optional[list] = range(2, 30, 4),
                       ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray,
                                  numpy.ndarray, numpy.ndarray]:
    if numpy.shape(sde)[1] <2:
        return None, None, None, None, None, None, None
    try:
        meansde = numpy.nanmean(sde[:-1, :], axis=1)
    except RuntimeWarning:
        return None, None, None, None, None, None, None
    stdsde = numpy.nanstd(sde[:-1, :], axis=1)
    minsde = numpy.nanmin(sde[:-1, :], axis=1)
    maxsde = numpy.nanmax(sde[:-1, :], axis=1)
    mediansde = numpy.nanmedian(sde[:-1, :], axis=1)
    nbins = len(vbins)
    hist = numpy.full((numpy.shape(sde)[0] - 1, nbins - 1), numpy.nan)
    bin_edges = numpy.full((numpy.shape(sde)[0] - 1, nbins), numpy.nan)
    for t in range(numpy.shape(sde)[0] - 1):
        hist[t, :], bin_edges[t, :] = numpy.histogram(sde[t, :], bins=vbins)
    return meansde, stdsde, minsde, maxsde, mediansde, hist, bin_edges


def plot_trajectory(file_out: str, lon_f: numpy.ndarray, lat_f: numpy.ndarray,
                    lon_d: numpy.ndarray, lat_d: numpy.ndarray,
                    time: numpy.ndarray, sde: numpy.ndarray,
                    proj: Optional = None, isscore: bool = True
                    ) -> pyplot.figure():
    figure = pyplot.figure(figsize=(14, 7))
    ax0 = pyplot.subplot(121, projection=proj)
    ax1 = pyplot.subplot(122)
    extent = (numpy.nanmin(lon_f)-1, numpy.nanmax(lon_f)+1,
              numpy.nanmin(lat_f)-1, numpy.nanmax(lat_f)+1)
    if proj is not None:
        ax0.gridlines(crs=proj, draw_labels=True, color='gray', linestyle='--',
                      alpha=0.5)
    for pa in range(0, numpy.shape(lon_f)[1], 1):
        if proj is None:
            ax0.plot(lon_f[:, pa], lat_f[:, pa], 'b')
        else:
            ax0.plot(lon_f[:, pa], lat_f[:, pa], 'b', transform=proj)
        ax1.plot(time[:-1], sde[:-1, pa])
    ax1.plot(time[:-1], numpy.nanmean(sde[:-1, :], axis=1), linewidth=3,
             color='black')
    if isscore:
        ax1.set_ylim(0.25, 1)
        ax1.set_ylabel('Score')
    else:
        ax1.set_ylim(25, 300)
        ax1.set_ylabel('cumulative separation (km)')
    ax1.set_xlabel('time from start of advection (days)')
    if proj is None:
        ax0.plot(lon_d, lat_d, '-r')
    else:
        ax0.plot(lon_d, lat_d, '-r', transform=proj)
    pyplot.savefig(file_out)
    return figure


def read_fictive_traj_netcdf(ifile: str, nstep: Optional[int] = 250):
    data = netCDF4.Dataset(ifile, 'r')
    dic_attr = {'ide': data.ide, 'data_type': data.data_type,
                'label': data.label, 'depth': data.depth}
    lon = data.variables['lon_hr'][:nstep, :]
    # try:
    lat = data.variables['lat_hr'][:nstep, :]
    mask = data.variables['mask_hr'][:nstep, :]
    _mask = ((abs(lon) > 360) | (abs(lat) > 90) | (mask == 1))
    lon[_mask] = numpy.nan
    lat[_mask] = numpy.nan
    time = data.variables['time_hr'][:nstep]
    data.close()
    return lon, lat, time, dic_attr


def read_fictive_traj_pickle(data: dict, nstep: Optional[int] = 250):
    dic_attr = {'ide': data['ide'], 'data_type': data['data_type'],
                'label': data['label'], 'depth': data['depth'],
                'first_date': data['first_date']}
    lon = data['lon_hr'][:nstep, :]
    # try:
    lat = data['lat_hr'][:nstep, :]
    mask = data['mask_hr'][:nstep, :]
    _mask = ((abs(lon) > 360) | (abs(lat) > 90) | (mask == 1))
    lon[_mask] = numpy.nan
    lat[_mask] = numpy.nan
    time = data['time_hr'][:nstep]
    return lon, lat, time, dic_attr


def process_all_ide(input_path: str, drifter_pyo: str, dir_out: str,
                    file_out: str, isplot: Optional[bool] = True,
                    projection: Optional[str] = None) -> dict:
    if 'gz' in os.path.splitext(drifter_pyo[0])[-1]:
        with gzip.open(drifter_pyo[0], 'rb') as f:
            dic_drif = pickle.load(f)
    else:
        with open(drifter_pyo[0], 'rb') as f:
            dic_drif = pickle.load(f)
    if os.path.isdir(input_path):
        input_netcdf = True
        list_advection = glob.glob(os.path.join(input_path, '*nc'))
    else:
        try:
            if 'gz' in os.path.splitext(input_path)[-1]:
                with gzip.open(input_path, 'rb') as f:
                    dic_all = pickle.load(f)
            else:
                with open(input_path, 'rb') as f:
                    dic_all = pickle.load(f)
        except pickle.UnpicklingError:
            logger.error(f'{input_path} should be a pickle object')
            sys.exit(1)
        input_netcdf = False
        list_advection = list(dic_all.keys())
    os.makedirs(dir_out, exist_ok=True)
    dic_result = {}
    for ifile in tqdm(list_advection[:]):
        if input_netcdf is True:
            _fname = os.path.basename(ifile)
            _fname_out = os.path.splitext(_fname)[0]
            logging.debug(f'Read Netcdf fictive trajectory {ifile}')
            hrlon, hrlat, hrtime, dic_attr = read_fictive_traj_netcdf(ifile)
        else:
            _fname_out = ifile
            res = read_fictive_traj_pickle(dic_all[ifile])
            hrlon, hrlat, hrtime, dic_attr = res
        hrlon = numpy.mod(hrlon + 180, 360) - 180
        if dic_attr['ide'] not in dic_drif:
            continue
        logging.debug(f'Read drifter data {drifter_pyo}')
        _lon = numpy.mod(numpy.array(dic_drif[dic_attr['ide']]['lon']) + 180,
                         360) - 180
        _lat = dic_drif[dic_attr['ide']]['lat']
        _time = numpy.array(dic_drif[dic_attr['ide']]['time'])
        first_day = datetime.datetime.timestamp(dic_attr['first_date'])
        # ddays = [(x - first_day).total_seconds() / 86400 for x in _time]
        ddays = [(x - first_day) / 86400 for x in _time]
        if hrtime[-1] > ddays[-1]:
            continue

        sde, dde, _lon_interp, _lat_interp = compute_nlcs(_lon, _lat, ddays,
                                                          hrlon, hrlat, hrtime)
        if numpy.all(sde[:-1] != sde[:-1]):
            continue
        res = compute_statistics(sde)
        meansde, stdsde, minsde, maxsde, mediansde, hist, nbins = res
        threshold = 4
        score = 1 - sde / threshold
        score[sde == numpy.nan] = 0
        score[score < 0] = 0
        ress = compute_statistics(score)
        meanscore, stdscore, minscore, maxscore, medianscore, hist, nbins = ress
        if meansde is None:
            continue
        file_plot_out = os.path.join(dir_out, f'{_fname_out}_sde.png')
        if isplot is True:
            _ = plot_trajectory(file_plot_out, hrlon, hrlat, _lon_interp,
                                _lat_interp, hrtime, dde*111.11, isscore=False,
                                proj=projection)
        file_plot_out = os.path.join(dir_out, f'{_fname_out}_score.png')
        if isplot is True:
            _ = plot_trajectory(file_plot_out, hrlon, hrlat, _lon_interp,
                                _lat_interp, hrtime, score, isscore=True,
                                proj=projection)

        dic_result[_fname_out] = {'time': hrtime,
                                  'meansde': meansde,
                                  'meanscore': meanscore,
                                  'stdsde':  stdsde,
                                  'stdscore': stdscore,
                                  'minsde': minsde,
                                  'minscore': minscore,
                                  'maxsde': maxsde,
                                  'maxscore': maxscore,
                                  'mediansde': mediansde,
                                  'medianscore': medianscore,
                                  'histogramsde': hist,
                                  'binsde': nbins}
        for key, value in dic_attr.items():
            dic_result[_fname_out][key] = value
    data_type = dic_attr['data_type']
    depth = dic_attr['depth']
    if depth is None:
        depth = 15
    if data_type not in file_out:
        file_out = f'{file_out}_{data_type}'
    if f'{int(depth):02d}m' not in file_out:
        file_out = f'{file_out}_{int(depth):02d}m'
    file_out = os.path.join(dir_out, f'{file_out}.pyo.gz')
    logger.info(f'Save results in pickle {file_out}')
    with gzip.open(file_out, 'wb') as f:
        pickle.dump(dic_result, f)
    return dic_result
