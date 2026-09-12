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


import argparse
import numpy
import datetime
import os
import logging
import pickle
from scipy.interpolate import RBFInterpolator
from matplotlib import pyplot
from velocity_metrics.reader import read_utils_xr
from velocity_metrics.utils.load_parameters import sel_data, sel_region
from velocity_metrics.spectrum import mod_spectrum
from typing import Optional, Tuple

logger = logging.getLogger()
handler = logging.StreamHandler()
logger.addHandler(handler)
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.ticker').disabled = True

pyplot.rcParams["axes.edgecolor"] = "black"
pyplot.rcParams["axes.linewidth"] = 1.5

dico_label = {
    "duacs_15m_8th": "DUACS",
    "globcurrent_15m_4th": "GlobCurrent",
    "unet_uv_aoml_15m_10y_11d_bathy_no_sst_mae_duacs_RonanUnet": "IMT-OSC$_{duacs}$",
    "unet_uv_aoml_15m_10y_11d_bathy_no_sst_mae_neurost_RonanUnet": "IMT-OSC$_{neurost}$",
    "neurost_sst_ssh_15m_10th": "NeurOST",
    # Ajoutez d'autres correspondances ici
}

DATE_FMT = "%Y%m%dT%H%M%SZ"


def load_data(data_type_file: str, first_date: datetime.datetime,
              last_date: datetime.datetime, sdepth: int, box: list
              ) -> Tuple[dict, dict, float, float]:
    logger.info(f'Load velocity {data_type_file}')
    cpath, cpattern, match, dic_name, depth, data_type, label, _ = sel_data(data_type_file)

    _read = read_utils_xr.read_velocity
    dic_vel, coord = _read(cpath, cpattern, match, first_date, last_date,
                           box=box, depth=depth[sdepth], dict_name=dic_name,
                           stationary=False, compute_strain=False,
                           compute_ow=False, compute_rv=False)

    dx = None
    dy = None
    return dic_vel, coord, dx, dy, data_type, label


def interpolate_missing_values(var: numpy.ndarray):
    # Mask of valid values
    mask = numpy.isfinite(var)
    # Indices of the array
    x = numpy.arange(var.shape[1])
    # Interpolate along each row
    for i in range(var.shape[0]):
        valid_x = x[mask[i]]
        valid_y = var[i][mask[i]]
        try:
            var[i][~mask[i]] = numpy.interp(x[~mask[i]], valid_x, valid_y)
        except:
            var[i][~mask[i]] = numpy.nanmean(var)
    return var

def interpolate_rbf(lon2d, lat2d, grid_points, var):
    if numpy.isnan(var).any():
        valid = ~numpy.isnan(var)
        valid_points = numpy.array((lon2d[valid].flatten(),
                                    lat2d[valid].flatten())).T
        valid_values = var[valid].flatten()
        rbf = RBFInterpolator(valid_points, valid_values, kernel='cubic',
                              smoothing=0)
        var = rbf(grid_points).reshape((jmt, imt))
    return var


def largest_valid_rectangle(arr):
    # Create a binary mask where 1 indicates valid (non-NaN) values
    valid_mask = ~numpy.isnan(arr)

    # Initialize variables to track the largest rectangle
    max_area = 0
    max_rect = None

    # Array to store the height of continuous 1's in the valid_mask
    height = numpy.zeros_like(valid_mask, dtype=int)

    # Iterate over each row to build the height matrix and find the max area rectangle
    for i in range(valid_mask.shape[0]):
        for j in range(valid_mask.shape[1]):
            # Update height (number of consecutive 1s up to the current row)
            if valid_mask[i, j]:
                height[i, j] = height[i-1, j] + 1 if i > 0 else 1

        # Find the maximum rectangle in the current row's histogram
        stack = []
        for j in range(valid_mask.shape[1] + 1):
            h = height[i, j] if j < valid_mask.shape[1] else 0
            while stack and h < height[i, stack[-1]]:
                height_idx = stack.pop()
                width = j if not stack else j - stack[-1] - 1
                area = height[i, height_idx] * width
                if area > max_area:
                    max_area = area
                    max_rect = (i - height[i, height_idx] + 1, stack[-1] + 1 if stack else 0, i, j - 1)
            stack.append(j)
    # Extract the subarray corresponding to the largest rectangle
    r1, c1, r2, c2 = max_rect
    return arr[r1:r2+1, c1:c2+1], max_rect


def compute_spectrum(dic: dict, coord: dict, key_u: str, key_v: str,
                     dx: Optional[float] = None, dy: Optional[float] = None,
                     kmin: Optional[float] = 3*10**-3,
                     tmax: Optional[int] = 36000,
                     mode: Optional[str] = 'cospectrum_uv'
                     ) -> Tuple[numpy.ndarray, numpy.ndarray]:

    if dx is None:
        dx = (numpy.mean(abs(coord['lonu'][1:] - coord['lonu'][:-1]))
              * 111.11 * numpy.cos(numpy.deg2rad(numpy.mean(coord['latu']))))
    if dy is None:
        dy = numpy.mean(abs(coord['latu'][1:] - coord['latu'][:-1])) * 111.11
    #lon2d, lat2d = numpy.meshgrid(coord['lonu'], coord['latu'])
    #grid_points = numpy.array((lon2d.flatten(), lat2d.flatten())).T
    logger.debug(f'dx: {dx}, dy: {dy}')
    list_key = [key_u, key_v]
    t = 0
    all_spec_eke = 0
    sel_area = None
    tmax = min(tmax, len(coord['time']))
    logger.debug(f'Number of time considered {tmax}')
    mask_var = (numpy.isnan(dic[key_u]['array'][0, :, :])
                | ~numpy.isfinite(dic[key_u]['array'][0, :, :])
                | numpy.isnan(dic[key_v]['array'][0, :, :])
                | ~numpy.isfinite(dic[key_v]['array'][0, :, :]))
    var = dic[key_u]['array'][0, :, :].copy()
    var[mask_var] = numpy.nan
    ars2, rect = largest_valid_rectangle(var)
    start = rect[:2]
    end = rect[2:]
    logger.debug(f'Spectrum will be computed on subregion indexes {start} {end}')
    while t < tmax:
        for key in list_key:
            varf = dic[key]['array'][t, :, :]
            var = varf[start[0] + 1: end[0], start[1] + 1: end[1]]
            jmt, imt = numpy.shape(var)
            var[numpy.isnan(var)] = 0 #numpy.nanmean(numpy.nanmean(var))
            # ijt = jmt * imt
            #var = interpolate_missing_values(var)
            var2 = numpy.zeros((2*jmt, 2*imt))
            var2[0: jmt, 0: imt] = var
            var2[: jmt, 0: imt] = var[jmt::-1, :]
            var2[:, imt:] = var2[:, imt-1::-1]

            # # Detrend
            var2 = var2 - numpy.nanmean(numpy.nanmean(var2))
            #var2[numpy.isnan(var2)] = 0.
            # var2=signal.detrend(var2, axis=0)
            # var2=signal.detrend(var2, axis=1)
            dic[key]['double'] = var2
            var3 = var2 * var2
            var2_rms = numpy.sqrt(numpy.nanmean(var3))
            # used to diagnose script
            dic[key]['rms'] = var2_rms
            # # Compute EKE
            [kx, ky, kkx, kky, kk] = mod_spectrum.get_kxky(2 * jmt, 2 * imt,
                                                           dx, dy, dim=True)

        # # EKE SPECTRUM
        [spec_eke, keke] = mod_spectrum.co_spec_uv(dx, dy,
                                                   dic[key_u]['double'],
                                                   dic[key_v]['double'])
        # norm = numpy.sqrt(dic[key_u]['double']**2 + dic[key_v]['double']**2)
        # eke = norm / 2
        # [spec_eke, phase, keke] = mod_spectrum.spec_2d(dx, dy, eke)
        # dk = keke[1] - keke[0]
        # u_rms = numpy.sqrt(sum(spec_eke * dk) / 2)
        # if (int(u_rms*100)==int(spec_rms*100)):
        #    logger.info('Fourier Transform is ok')
        # else:
        #    logger.error('Fourier Transform is not ok')

        # Dimensionalize variables
        kteke = keke[::1] / 2 / numpy.pi
        spec_eke = spec_eke[::1] * 2 * numpy.pi
        t += 1
        all_spec_eke += spec_eke
    _ind = numpy.where(kteke > kmin)
    all_spec_eke /= t #tmax
    return kteke[_ind], all_spec_eke[_ind]


def plot_spectrum(dic_spectrum: dict, filename: str,
                  alpha: Optional[float] = 0.5,
                  figsize: Optional[list] = (5, 5)):
    fig = pyplot.figure(figsize=figsize)
    cm = pyplot.cm.tab20(numpy.linspace(0, 1, 10))
    for key, data in dic_spectrum.items():
        # # EKE SPECTRUM
        if isinstance(data['color'], str):
            col = data['color']
            pyplot.loglog(data['k'][:-1], data['spec_eke'][:-1],
                          col, alpha=alpha,
                          label=dico_label.get(data['label'],data['label']))
        else:
            col = cm[data['color']]
            pyplot.loglog(data['k'][:-1], data['spec_eke'][:-1],
                          c=col, alpha=alpha,
                          label=dico_label.get(data['label'],data['label']))
        pyplot.legend(loc='lower left',prop={'size': 6})
    pyplot.grid()
    pyplot.xlabel('1/km')
    pyplot.ylabel('Co-spectrum u, v')
    logging.info(f'saving spectrum figure in {filename}')
    #pyplot.savefig(filename)
    #logger.info(f'Saving figure in {filename}')
    return fig


def plot(list_pickle: list, outfile: str, list_color: Optional[list] = None):
    dic_spectrum = {}
    for i, file in enumerate(list_pickle):
        with open(file, 'rb') as f:
            dic = pickle.load(f)
        if list_color is not None:
            if len(list_color) > i:
                dic['color'] = list_color[i]
            else:
                logger.warning('Not enough color provided')
        dic_spectrum[dic['data_type']] = dic
    fig = plot_spectrum(dic_spectrum, outfile)
    return fig

def run(list_data_type: list, region: Optional[str] = None,
        depth: Optional[int] = None,
        first_date: Optional[str] = '19900101T000000Z',
        last_date: Optional[str] = '20500101T000000Z',
        kmin: Optional[float] = 1e-3,
        tmax: Optional[int] = 36000,
        output_file: Optional[str] = None, output_dir: Optional[str] = './'):
    lllon, urlon, lllat, urlat, coords, name = sel_region(region)
    filename = f'spectrum_{name}_{depth}.png'
    box = [lllon, urlon, lllat, urlat]
    # dic_color = {'001_024_d': 0, '015_004':1, '015_007':2, '015_002':3,
    #             '008_047': 4, 'SST_SSH': 5}
    dic_spectrum = {}
    col = 0
    first_date = datetime.datetime.strptime(first_date, DATE_FMT)
    last_date = datetime.datetime.strptime(last_date, DATE_FMT)
    os.makedirs(output_dir, exist_ok=True)
    for data_type_file in list_data_type:
        # _av = 1/4
        #  Load Velocity
        dic, coord, dx, dy, data_type, label = load_data(data_type_file,
                                                         first_date, last_date,
                                                         depth, box)
        dic_spectrum[data_type] = {}

        k, spec_eke = compute_spectrum(dic, coord, 'ums', 'vms', dx=dx, dy=dy,
                                       kmin=kmin, tmax=tmax)
        dic_data = {'k': k, 'spec_eke': spec_eke, 'label': label,
                    'color': col, 'data_type': data_type}
        dic_spectrum[data_type]['k'] = k
        dic_spectrum[data_type]['spec_eke'] = spec_eke
        dic_spectrum[data_type]['label'] = label
        dic_spectrum[data_type]['color'] = col  # dic_color[data_type]
        col += 1
        if output_file is None:
            reg = os.path.splitext(os.path.basename(region))[0]
            name = f'spectrum_{data_type}_{reg}_{depth}'
            # name = f'{name}_{first_date}_{last_date}'
            filename = os.path.join(output_dir, name)
        else:
            filename = os.path.join(output_dir, f'{output_file}_{data_type}')
        logging.info(f'saving spectrum dictionary in {filename}.pyo')
        with open(f'{filename}.pyo', 'wb') as f:
            pickle.dump(dic_data, f)
    return dic_spectrum


if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('list_data_type', type=str, nargs='+',
                        help='List of data type to consider for analyses')
    parser.add_argument('-r', '--region', default='T1', type=str,
                        help='Region, select among T1, T2_DTU, T2_SOC, T3_SAR, T3_ARC')
    parser.add_argument('--first_date', default='19900101T000000Z', type=str,
                        required=False,
                        help='First time considered for analyses')
    parser.add_argument('--last_date', default='20500101T000000Z', type=str,
                        required=False,
                        help='Last time considered for analyses')
    parser.add_argument('-d', '--depth', default=0, type=int, required=False,
                        help='Depth index')
    parser.add_argument('-l', '--length', default=3000, type=int,
                        required=False,
                        help='Maximum extent to compute spectrum in km')
    parser.add_argument('--out', dest='outdir',
                        type=str, default='/tmp',
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

    list_data_type = list(args.list_data_type)
    dic_spectrum = run(list_data_type, args.region, args.depth,
                       first_date=args.first_date, kmin=1/args.length,
                       last_date=args.last_date, output_dir=args.outdir)
    _ = plot(dic_spectrum, os.path.join(args.output_dir, 'spectrum.png'))
