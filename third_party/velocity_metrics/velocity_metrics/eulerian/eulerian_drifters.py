# vim: ts=4:sts=4:sw=4
#
# @author <lucile.gaultier@oceandatalab.com>
# @date 2024-01-10
#
# Copyright (C) 2020-2024 OceanDataLab
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

"""This module provides Compares velocity with the one from drifters,
Drifters data are formatted in read_drifters function

"""

import argparse
import numpy
import sys
import os
import scipy.interpolate as interpolate
import datetime
import pickle
import logging
from typing import Tuple, Optional
from scipy.ndimage import gaussian_filter
from velocity_metrics.reader import read_utils_xr
from velocity_metrics.utils import load_parameters
from velocity_metrics.reader import read_drifters
import velocity_metrics.utils.constant as const
# from scipy.signal import correlate

logger = logging.getLogger()
handler = logging.StreamHandler()
logger.addHandler(handler)


def run(drifter_paths: list, data_type_file: str,
        first_date: datetime.datetime,
        last_date: datetime.datetime, region: Optional[str] = None,
        sdepth: Optional[int] = 0, output_dir: Optional[str] = './',
        temporal_filtering: Optional[float] = 0,
        size_box: Optional[float] = 3) -> None:
    x = load_parameters.sel_region(region)
    lllon, urlon, lllat, urlat, coords, name = x
    logger.info(f'Load velocity {data_type_file}')
    x = load_parameters.sel_data(data_type_file)
    cpath, cpattern, match, dic_name, depth, data_type, label, spatial_cov_h = x
    box = [lllon, urlon, lllat, urlat]
    _read = read_utils_xr.read_velocity
    VEL, coord = _read(cpath, cpattern, match, first_date, last_date,
                       box=box, depth=depth[sdepth], dict_name=dic_name,
                       stationary=False, compute_strain=False,
                       compute_ow=False, compute_rv=False)
    strlist = ",".join(list(drifter_paths))
    logger.info(f'Load drifter data {strlist}')
    if drifter_paths is None:
        logger.error('Please provide at list one drifter file path')
        sys.exit(1)
    dic_drif = read_drifters.load_drifter_pickle(list(drifter_paths))
    logger.warn(f'Compute and save statistics in {output_dir} directory')
    mean_dic, std_dic, res_map = run_compute_error(VEL, coord, dic_drif,
                                          averaging=spatial_cov_h*3600,
                                          temporal_filtering=temporal_filtering,
                                          box=box, size_box=size_box)
    os.makedirs(output_dir, exist_ok=True)
    _file = os.path.join(output_dir, f'Eulerian_RMS_{data_type}.pyo')
    with open(_file, 'wb') as f:
        pickle.dump(mean_dic, f)
    os.makedirs(output_dir, exist_ok=True)
    _file = os.path.join(output_dir, f'Eulerian_STD_{data_type}.pyo')
    with open(_file, 'wb') as f:
        pickle.dump(std_dic, f)
    _file = os.path.join(output_dir, f'Eulerian_BINNED_{data_type}.pyo')
    with open(_file, 'wb') as f:
        pickle.dump(res_map, f)


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
    gl = ax.gridlines(draw_labels=["top", "left"], linestyle='--', linewidth=2,
                      alpha=0.2, color='gray')
    gl.xlabels_top = False
    gl.ylabels_left = False
    return gl


def plot_bin_diff(ifile1: str, ifile2: str, output_dir: str, var: str,
                   box: Optional[list] = None,
                   vmin: Optional[float] = None, vmax: Optional[float] = None,
                   cmap: Optional[str] = 'seismic', proj = None):
    from matplotlib import pyplot
    from mpl_toolkits.axes_grid1 import ImageGrid
    # list_diag = ('Quadratic Error (%)', 'Correlation',
    #              'Explained Variance (%)', 'RMSD', 'field RMS', 'drifter RMS',
    #              'difference STD', 'drifter STD')
    # list_comp = ('Eastward', 'Northward', 'Norm')
    # list_plot = [f'{comp} {diag}' for diag in list_diag for comp in list_comp]

    with open(ifile1, 'rb') as f:
        dic1 = pickle.load(f)
    with open(ifile2, 'rb') as f:
        dic2 = pickle.load(f)
    n = 4
    if box is None:
        box = (numpy.min(dic1['lon']), numpy.max(dic1['lon']),
               numpy.min(dic1['lat']), numpy.max(dic1['lat']))
    fig = pyplot.figure(figsize=(n*4, n))
    if proj is not None:
        import cartopy

        from cartopy.mpl.geoaxes import GeoAxes
        from mpl_toolkits.axes_grid1 import AxesGrid

        axes_class = (GeoAxes, dict(projection=proj))
        ax = AxesGrid(fig, 111, axes_class=axes_class,
                      nrows_ncols=(1, 4), axes_pad=0.6,
                      cbar_location='bottom', cbar_mode='each',
                      cbar_pad="2%", cbar_size='7%', label_mode='all')
    else:
        ax = ImageGrid(fig, 111, nrows_ncols=(1, 4), axes_pad=0.6,
                       cbar_location="right", cbar_mode="each",
                       cbar_size="7%", cbar_pad="2%")
    ind = 0
    for comp in ('Eastward', 'Northward', 'Norm'):
        if vmin is None:
            vmin = 0
        if vmax is None:
            vmax = 0.3
        value = dic1[f'{comp} {var}'] - dic2[f'{comp} {var}']
        if proj is not None:
            gl = init_cartopy(ax[ind], box)
            c = ax[ind].pcolor(dic1['lon'], dic1['lat'], value,
                                vmin=vmin, vmax=vmax,
                                cmap=cmap, transform=proj)
        else:
            c = ax[ind].pcolor(dic1['lon'], dic1['lat'], value,
                               vmin=vmin, vmax=vmax,
                               cmap=cmap)
        ax[ind].set_title(f'{comp} {var}')
        ax.cbar_axes[ind].colorbar(c)

        ind += 1

    vmin = 0
    value = dic1[f'Number of Points']
    vmax = numpy.nanmax(value)
    if proj is not None:
        gl = init_cartopy(ax[ind], box)
        c = ax[ind].pcolor(dic1['lon'], dic1['lat'], value,
                           vmin=vmin, vmax=vmax,
                           cmap='seismic', transform=proj)
    else:
        c = ax[ind].pcolor(dic1['lon'], dic1['lat'], value, vmin=vmin, vmax=vmax,
                        cmap='seismic')
    ax[ind].set_title(f'Number of Points')
    ax.cbar_axes[ind].colorbar(c)
    pyplot.savefig(os.path.join(output_dir, f'{var}.png'))
    return fig

def plot_bin(ifile: str, output_dir: str, var: str,
             box: Optional[list] = None,
             vmin: Optional[float] = None, vmax: Optional[float] = None,
             cmap: Optional[str] = 'seismic', proj = None):
    from matplotlib import pyplot
    from mpl_toolkits.axes_grid1 import ImageGrid
    # list_diag = ('Quadratic Error (%)', 'Correlation',
    #              'Explained Variance (%)', 'RMSD', 'field RMS', 'drifter RMS',
    #              'difference STD', 'drifter STD')
    # list_comp = ('Eastward', 'Northward', 'Norm')
    # list_plot = [f'{comp} {diag}' for diag in list_diag for comp in list_comp]

    with open(ifile, 'rb') as f:
        dic = pickle.load(f)
    n = 4
    if box is None:
        box = (numpy.min(dic['lon']), numpy.max(dic['lon']),
               numpy.min(dic['lat']), numpy.max(dic['lat']))
    fig = pyplot.figure(figsize=(n*4, n))
    if proj is not None:
        import cartopy

        from cartopy.mpl.geoaxes import GeoAxes
        from mpl_toolkits.axes_grid1 import AxesGrid

        axes_class = (GeoAxes, dict(projection=proj))
        ax = AxesGrid(fig, 111, axes_class=axes_class,
                      nrows_ncols=(1, 4), axes_pad=0.6,
                      cbar_location='bottom', cbar_mode='each',
                      cbar_pad="2%", cbar_size='7%', label_mode='')
    else:
        ax = ImageGrid(fig, 111, nrows_ncols=(1, 4), axes_pad=0.6,
                       cbar_location="right", cbar_mode="each",
                       cbar_size="7%", cbar_pad="2%")
    ind = 0
    for comp in ('Eastward', 'Northward', 'Norm'):
        if vmin is None:
            vmin = 0
        if vmax is None:
            vmax = 0.3
        value = dic[f'{comp} {var}']
        if proj is not None:
            gl = init_cartopy(ax[ind], box)
            c = ax[ind].pcolor(dic['lon'], dic['lat'], value,
                                vmin=vmin, vmax=vmax,
                                cmap=cmap, transform=proj)
        else:
            c = ax[ind].pcolor(dic['lon'], dic['lat'], value,
                               vmin=vmin, vmax=vmax,
                               cmap=cmap)
        ax[ind].set_title(f'{comp} {var}')
        ax.cbar_axes[ind].colorbar(c)

        ind += 1

    vmin = 0
    value = dic[f'Number of Points']
    vmax = numpy.nanmax(value)
    if proj is not None:
        gl = init_cartopy(ax[ind], box)
        c = ax[ind].pcolor(dic['lon'], dic['lat'], value,
                           vmin=vmin, vmax=vmax,
                           cmap='seismic', transform=proj)
    else:
        c = ax[ind].pcolor(dic['lon'], dic['lat'], value, vmin=vmin, vmax=vmax,
                        cmap='seismic')
    ax[ind].set_title(f'Number of Points')
    ax.cbar_axes[ind].colorbar(c)
    pyplot.savefig(os.path.join(output_dir, f'{var}.png'))
    return fig


def run_compute_error(VEL: dict, coord: dict, dic_drif: dict,
                      averaging: Optional[float] = 3600,
                      temporal_filtering: Optional[float] = 0,
                      box: Optional[list] = None, size_box: Optional[int] = 2
                      ) -> Tuple[dict, dict]:
    # Compute difference velocity-drifters
    out_dic = compute_error(VEL, coord, dic_drif, averaging,
                            temporal_filtering)
    mean_dic = {}
    std_dic = {}
    # Compute statistics
    logger.info('Compute statistics')
    mean_dic = {}
    std_dic = {}

    for comp in out_dic.keys():
        _tmp = numpy.array(out_dic[comp])

        mean_dic[comp] = numpy.sqrt(numpy.nanmean(_tmp**2))
        std_dic[comp] = numpy.nanstd(_tmp)
    for comp in ('Eastward', 'Northward', 'Norm'):
        x = numpy.array(out_dic[f'{comp} speed']).ravel()
        y = numpy.array(out_dic[f'{comp} drifter speed']).ravel()
        mean_dic[f'{comp} Quadratic Error (%)'] = 100 * (mean_dic[f'{comp} difference']
                                               / mean_dic[f'{comp} drifter speed'])
        mask = (~numpy.isnan(x)) & (~numpy.isnan(y))
        x_nm = x[mask]
        y_nm = y[mask]
        if len(x_nm) > 1:
            msd = numpy.nanmean(numpy.array(out_dic[f'{comp} difference'])**2)
            mean_dic[f'{comp} Correlation'] = numpy.corrcoef(x_nm, y_nm)[0, 1]
            cov = numpy.cov(x_nm, y_nm, ddof=1)[0][1]
            var = numpy.var(y_nm, ddof=1)
            mean_dic[f'{comp} Explained Variance (%)'] = (100 * cov / var)
            mean_dic[f'{comp} MSD Explained Variance (%)'] = (1 - (msd / var))*100

        mean_dic[f'{comp} RMSD'] = mean_dic.pop(f'{comp} difference')
        mean_dic[f'{comp} field RMS'] = mean_dic.pop(f'{comp} speed')
        mean_dic[f'{comp} drifter RMS'] = mean_dic.pop(f'{comp} drifter speed')
        mean_dic[f'{comp} difference STD'] = std_dic.pop(f'{comp} difference')
        mean_dic[f'{comp} field STD'] = std_dic.pop(f'{comp} speed')
        mean_dic[f'{comp} drifter STD'] = std_dic.pop(f'{comp} drifter speed')
    mean_dic['Number of Points'] = len(_tmp)
    std_dic['Number of Points'] = len(_tmp)
    res_map = run_bin_error(out_dic, box=box, dbox=size_box)
    return mean_dic, std_dic, res_map


def run_bin_error(out_dic: dict, box: Optional[list] = None,
                  dbox: Optional[float] = 2):
    lons = out_dic['lon']
    lats = out_dic['lat']
    if box is None:
        box = [numpy.min(lons), numpy.max(lons),
               numpy.min(lats), numpy.max(lats)]
    lon = numpy.arange(box[0], box[1] + dbox, dbox)
    lat = numpy.arange(box[2], box[3] + dbox, dbox)
    result = {}
    list_diag = ('Quadratic Error (%)', 'Correlation',
                 'Explained Variance (%)', 'MSD Explained Variance (%)','RMSD', 'field RMS', 'drifter RMS',
                 'difference STD', 'field STD', 'drifter STD')
    for comp in ('Eastward', 'Northward', 'Norm'):
        for diag in list_diag:
            result[f'{comp} {diag}'] = numpy.full((len(lat), len(lon)),
                                                  numpy.nan)
    result['lon'] = lon
    result['lat'] = lat
    result['Number of Points'] = numpy.full((len(lat), len(lon)), 0)
    for ilon, _lon in enumerate(lon):
        for ilat, _lat in enumerate(lat):
            ix = numpy.where((lons >= _lon - dbox/2) & (lons <= _lon + dbox/2)
                             & (lats >= _lat - dbox/2)
                             & (lats <= _lat + dbox+2))
            if len(ix[0]) < 3:
                continue
            _sel = numpy.array(ix[0].astype(int))
            for comp in ('Eastward', 'Northward', 'Norm'):
                x = numpy.array(out_dic[f'{comp} speed'])[_sel].ravel()
                y = numpy.array(out_dic[f'{comp} drifter speed'])[_sel].ravel()
                _tmp = numpy.array(out_dic[f'{comp} difference'])[_sel].ravel()
                msd = numpy.nanmean(_tmp**2)
                result[f'{comp} RMSD'][ilat, ilon] = numpy.sqrt(numpy.nanmean(_tmp**2))
                result[f'{comp} difference STD'][ilat, ilon] = numpy.nanstd(_tmp)
                result[f'{comp} field RMS'][ilat, ilon] = numpy.sqrt(numpy.nanmean(x**2))
                result[f'{comp} drifter RMS'][ilat, ilon] = numpy.sqrt(numpy.nanmean(y**2))
                result[f'{comp} field STD'][ilat, ilon] = numpy.nanstd(x)
                result[f'{comp} drifter STD'][ilat, ilon] = numpy.nanstd(y)
                mask = ~numpy.isnan(x) & ~numpy.isnan(y)
                x_nm = x[mask]
                y_nm = y[mask]
                result['Number of Points'][ilat, ilon] = len(x_nm)
                val = numpy.sqrt(numpy.nanmean(_tmp**2)) / numpy.sqrt(numpy.nanmean(y**2)) * 100
                result[f'{comp} Quadratic Error (%)'][ilat, ilon] = val

                if len(x_nm) > 1:
                    result[f'{comp} Correlation'][ilat, ilon] = numpy.corrcoef(x_nm, y_nm)[0, 1]
                    cov = numpy.cov(x_nm, y_nm, ddof=1)[0][1]
                    var = numpy.var(y_nm, ddof=1)
                    result[f'{comp} Explained Variance (%)'][ilat, ilon] = (100 * cov / var)
                    result[f'{comp} MSD Explained Variance (%)'][ilat, ilon] = (1 - (msd / var))*100
    return result


def compute_error(dic: dict, coord: dict, dic_drif: dict, _av: float, filt=0
                  ) -> dict:
    out_dic = {}
    vec_verr = []
    vec_uerr = []
    vec_nerr = []
    vec_derr = []
    vec_norm = []
    vec_u = []
    vec_v = []
    vec_du = []
    vec_dv = []
    vec_dnorm = []
    lon = []
    lat = []
    dic['norm'] = {}
    dic['norm']['array'] = numpy.sqrt(dic['ums']['array']**2
                                      + dic['vms']['array']**2)
    dic['dir'] = {}
    dic['dir']['array'] = numpy.arctan2(dic['vms']['array'],
                                        dic['ums']['array']) % (2 * numpy.pi)
    for iid in dic_drif.keys():
        dt = [numpy.datetime64(x) for x in dic_drif[iid]['date']]
        dlon = dic_drif[iid]['lon']
        dlat = dic_drif[iid]['lat']
        due = numpy.array(dic_drif[iid]['ums']).ravel()  # * 10**(-2)
        if filt != 0:
            due = gaussian_filter(due, sigma=filt)
        dvn = numpy.array(dic_drif[iid]['vms']).ravel()  # * 10**(-2)
        if filt != 0:
            dvn = gaussian_filter(dvn, sigma=filt)
        due[abs(due) > 20] = numpy.nan
        dvn[abs(dvn) > 20] = numpy.nan
        due[due == 0] = numpy.nan
        dvn[dvn == 0] = numpy.nan

        dnorm = numpy.sqrt(due**2 + dvn**2)
        ddir = numpy.arctan2(dvn, due) % (2 * numpy.pi)
        _ind_t = numpy.where((coord['time'] >= dt[0])
                             & (coord['time'] <= dt[-1]))[0]
        if len(_ind_t) < 2:
            continue
        if dt[-1] < (dt[0] + 2 * _av):
            continue
        _sl = slice(_ind_t[0], _ind_t[-1])
        vnorm = dic['norm']['array'][_ind_t, :, :]
        vdir = dic['dir']['array'][_ind_t, :, :]
        ums = dic['ums']['array'][_ind_t, :, :]
        vms = dic['vms']['array'][_ind_t, :, :]
        time = numpy.array(coord['time'])[_ind_t]
        for t in range(numpy.shape(time)[0]):
            _time = time[t]
            _av0 = _time - numpy.timedelta64(int(_av / 2), 's')
            _av1 = _time + numpy.timedelta64(int(_av / 2), 's')
            _it = numpy.where((dt >= (_av0)) & (dt <= (_av1)))[0]
            if not _it.any():
                continue
            _sl = slice(_it[0], _it[-1] + 1)
            for tind in _it:
                _dlont = dlon[tind].ravel()
                # dlont = numpy.mod(dlont + 360, 360)
                _dlatt = dlat[tind].ravel()
                _duet = due[tind].ravel()
                _dvnt = dvn[tind].ravel()
                _dnormt = dnorm[tind].ravel()
                _ddirt = numpy.mod(ddir[tind], numpy.pi).ravel()
                # if dnormt < 0.5:
                #     continue
                func = interpolate.RegularGridInterpolator((dic['ums']['lat'],
                                                            dic['ums']['lon']),
                                                           ums[t, :, :])
                try:
                    intu = func((_dlatt, _dlont))
                except:
                    continue
                func = interpolate.RegularGridInterpolator((dic['vms']['lat'],
                                                            dic['vms']['lon']),
                                                           vms[t, :, :])
                intv = func((_dlatt, _dlont))
                func = interpolate.RegularGridInterpolator((dic['ums']['lat'],
                                                            dic['ums']['lon']),
                                                           vnorm[t, :, :])
                intn = func((_dlatt, _dlont))
                func = interpolate.RegularGridInterpolator((dic['ums']['lat'],
                                                            dic['ums']['lon']),
                                                           vdir[t, :, :])
                intd = numpy.mod(func((_dlatt, _dlont)), numpy.pi)
                if intu == 0 or intv == 0 or intd == 0 or intn == 0:
                    intu = numpy.nan
                    intv = numpy.nan
                    intd = numpy.nan
                    intn = numpy.nan
                lat.append(_dlatt)
                lon.append(_dlont)
                vec_uerr.append(abs(intu - _duet))
                vec_verr.append(abs(intv - _dvnt))
                vec_nerr.append(abs(intn - _dnormt))
                vec_derr.append(abs(intd - _ddirt))
                vec_norm.append(intn)
                vec_u.append(intu)
                vec_v.append(intv)
                vec_dnorm.append(_dnormt)
                vec_du.append(_duet)
                vec_dv.append(_dvnt)

    out_dic['Eastward difference'] = vec_uerr
    out_dic['Northward difference'] = vec_verr
    out_dic['Norm difference'] = vec_nerr
    out_dic['Direction difference'] = numpy.rad2deg(vec_derr)
    out_dic['Eastward speed'] = vec_u
    out_dic['Northward speed'] = vec_v
    out_dic['Eastward drifter speed'] = vec_du
    out_dic['Northward drifter speed'] = vec_dv
    out_dic['Norm speed'] = vec_norm
    out_dic['Norm drifter speed'] = vec_dnorm
    out_dic['lon'] = lon
    out_dic['lat'] = lat
    return out_dic


if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('list_drifters', type=str, nargs='+',
                        help='List of pyo files which contains drifter to consider for analyses')
    parser.add_argument('-d', '--data_type', type=str, required=True,
                        help='Parameter file to read velocity to analyse')
    parser.add_argument('--region', default='T1', type=str, required=False,
                        help='Region json file name')
    parser.add_argument('--depth', default=0, type=int, required=False,
                        help='Depth index')
    parser.add_argument('--temporal_filtering', default=0, type=float,
                        required=False,
                        help='Number of days for filtering')
    parser.add_argument('--first_date', default='19900101T000000Z', type=str,
                        required=False,
                        help='First time considered for analyses')
    parser.add_argument('--last_date', default='20500101T000000Z', type=str,
                        required=False,
                        help='Last time considered for analyses')
    parser.add_argument('--out', dest='outdir',
                        type=str, default='/tmp',
                        help='Path for output python dictionnary')
    parser.add_argument('--verbose', action='store_true', default=False,
                        required=False)
    parser.add_argument('--debug', action='store_true', default=False,
                        required=False)

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.INFO)
    if args.debug:
        logger.setLevel(logging.DEBUG)
    first_date = datetime.datetime.strptime(args.first_date, const.FMT)
    last_date = datetime.datetime.strptime(args.last_date, const.FMT)
    run(args.list_drifters, args.data_type, first_date, last_date,
        region=args.region, temporal_filtering=args.temporal_filtering,
        sdepth=args.depth, output_dir=args.out)
