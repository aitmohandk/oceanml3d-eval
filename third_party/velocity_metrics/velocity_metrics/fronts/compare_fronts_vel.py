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
import numpy
import glob
import os
import datetime
import json
from scipy import interpolate
import logging
import argparse
import tqdm
import sys
from typing import Optional, Tuple
from velocity_metrics.reader import read_utils_xr
import velocity_metrics.utils.tools as tools
from velocity_metrics.utils import load_parameters

logger = logging.getLogger(__name__)
FRONT_FMT = '%Y%m%dT%H%M%SZ'


def dic_open(ifile: str, ext: str, syntool: Optional[bool] = False
             ) -> Tuple[dict, datetime.datetime, datetime.datetime]:
    if ext == 'json':
        if syntool is True:
            dic_front = read_syntooljson_file(ifile)
        else:
            dic_front = read_json_file(ifile)
        if dic_front is not None:
            fstart = dic_front['time_coverage_start']
            fstop = dic_front['time_coverage_end']
        else:
            return None, None, None
    else:
        with open(ifile, 'rb') as f:
            dic_front = pickle.load(f)
        fstart = dic_front['time_coverage_start']
        try:
            _str = str(fstart.decode())
        except ValueError:
            _str = str(fstart)
        fstart = datetime.datetime.strptime(_str, FRONT_FMT)
        fstop = dic_front['time_coverage_end']
        try:
            _str = str(fstop.decode())
        except ValueError:
            _str = str(fstop)
        fstop = datetime.datetime.strptime(_str, FRONT_FMT)
    return dic_front, fstart, fstop


def read_syntooljson_file(file_path: str) -> dict:
    if not os.path.exists(file_path):
        raise IOError('File not found: {}'.format(file_path))
    with open(file_path, 'rb') as f:
        try:
            djson = json.load(f)
        except ValueError:
            logger.error(f'Decoding Json file {file_path} has failed')
            return None
    dic_front = {}
    dic_front['lon'] = []
    dic_front['lat'] = []
    dic_front['sst_grad_lon'] = []
    dic_front['sst_grad_lat'] = []
    dic_front['flag_front'] = []
    kprop = ['sst_grad_lon', 'sst_grad_lat', 'flag_front']
    shape = ('LINE', 'POLYGON')
    _date0 = datetime.datetime(1970, 1, 1)
    for i in len(djson):
        if any(_type in djson['type'] for _type in shape):
            _len = len(djson['points'])
            _lon = [djson[i]['points'][j][0] for j in range(_len)]
            _lat = [djson[i]['points'][j][1] for j in range(_len)]
            dic_front['lon'].append(_lon)
            dic_front['lat'].append(_lat)
            dic_front['time_coverage_start'] = djson[i]['start'] + _date0
            dic_front['time_coverage_end'] = djson[i]['end'] + _date0
            for key in kprop:
                for _gr in djson[i]['properties'].keys():
                    if key in djson[i]['properties'][_gr].keys():
                        dic_front[key].append(djson[i]['properties'][key])

    return dic_front


def read_json_file(file_path: str) -> dict:
    if not os.path.exists(file_path):
        raise IOError('File not found: {}'.format(file_path))
    with open(file_path, 'rb') as f:
        try:
            djson = json.load(f)
        except ValueError:
            logger.error(f'Decoding Json file {file_path} has failed')
            return None
    for key in ('time_coverage_start', 'time_coverage_end'):
        djson[key] = datetime.datetime.strptime(djson[key], '%Y%m%dT%H%M%SZ')
    for i in range(len(djson['lon'])):
        listkey = ('sst_final', 'sst_grad_lon', 'sst_grad_lat', 'sst_grad')
        for key in listkey:
            if key not in djson.keys():
                logger.error(f'missing key {key} in {file_path}')
                return None
            _tmp = djson[key][i]
            _tmp2 = [float(i or numpy.nan) for i in _tmp]
            djson[key][i] = _tmp2
    return djson


def smooth(x: numpy.ndarray, window_len: Optional[int] = 5,
           window: Optional[str] = 'hanning') -> numpy.ndarray:
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd
                    integer
        window: the type of window from 'flat', 'hanning', 'hamming',
                'bartlett', 'blackman'
                 flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t = linspace(-2, 2, 0.1)
    x = sin(t) + randn(len(t)) * 0.1
    y = smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman,
    numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead
    of a string
    NOTE: length(output) != length(input), to correct this:
    return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if window_len < 3:
        return x

    s = numpy.r_[x[window_len-1: 0: -1], x, x[-2: -window_len-1: -1]]
    if window == 'flat':  # moving average
        w = numpy.ones(window_len, 'd')
    else:
        w = eval(f'numpy.{window}(window_len)')

    y = numpy.convolve(w / w.sum(), s, mode='valid')
    return y


def comp_front_vel(dic_front: dict, dic_vel: dict,
                   gradient_threshold: Optional[float] = 0.05,
                   vel_threshold: Optional[float] = 0.1) -> dict:
    u_func = interpolate.RectBivariateSpline(dic_vel['ums']['lat'],
                                             dic_vel['ums']['lon'],
                                             dic_vel['ums']['array'],
                                             kx=1, ky=1)
    v_func = interpolate.RectBivariateSpline(dic_vel['vms']['lat'],
                                             dic_vel['vms']['lon'],
                                             dic_vel['vms']['array'],
                                             ky=1, kx=1)
    out_dic = {}
    out_dic['scalar_product'] = []
    out_dic['vectorial_product'] = []
    out_dic['flux_along'] = []
    out_dic['flux_across'] = []
    out_dic['vel'] = []
    out_dic['lon'] = []
    out_dic['lat'] = []
    out_dic['flag_front'] = []
    out_dic['gradient_sst'] = []
    for i in range(len(dic_front['lon'])):
        _lon = smooth(numpy.array(dic_front['lon'][i]))
        _lat = smooth(numpy.array(dic_front['lat'][i]))
        _gradx = smooth(numpy.array(dic_front['sst_grad_lon'][i]))
        _grady = smooth(numpy.array(dic_front['sst_grad_lat'][i]))
        # TODO flag before smoothing fronts
        # _flag = numpy.array(dic_front['flag_front'][i])
        _u = (_lon[2:] - _lon[:-2])*numpy.cos(numpy.nanmean(_lat)*numpy.pi/180)
        _v = _lat[2:] - _lat[:-2]
        _norm = numpy.sqrt(_u**2 + _v**2)
        _u = _u/_norm
        _v = _v/_norm
        fu = u_func.ev(_lat[1:-1], _lon[1:-1])
        fv = v_func.ev(_lat[1:-1], _lon[1:-1])

        fu[abs(fu) > 100] = numpy.nan
        # numpy.ma.masked_where(fu > 100 or fv >100, fu)
        fv[abs(fv) > 100] = numpy.nan    # numpy.ma.masked_where(fv > 100, fv)
        _vel = numpy.sqrt(fu**2 + fv**2)
        fu /= _vel
        fv /= _vel
        ldot = [numpy.nan, ]
        lvec = [numpy.nan, ]
        nldot = [numpy.nan, ]
        nlvec = [numpy.nan, ]
        _gradient_sst = numpy.sqrt(_gradx**2 + _grady**2)
        for k in range(len(_u)):
            if _gradient_sst[k] > gradient_threshold:
                nlvec.append(abs(numpy.dot([_v[k], -_u[k]], [fu[k], fv[k]])))
                lvec.append(abs(numpy.dot([_v[k], -_u[k]], [fu[k], fv[k]])
                                * _vel[k] * _norm[k]))
                nldot.append(abs(numpy.dot([_u[k], _v[k]], [fu[k], fv[k]])))
                ldot.append(abs(numpy.dot([_u[k], _v[k]], [fu[k], fv[k]])
                                * _vel[k] * _norm[k]))
            else:
                ldot.append(numpy.nan)
                lvec.append(numpy.nan)
                nldot.append(numpy.nan)
                nlvec.append(numpy.nan)
        ldot.append(numpy.nan)
        lvec.append(numpy.nan)
        nldot.append(numpy.nan)
        nlvec.append(numpy.nan)
        # if dic_front['flag_front'][i] < 2 :
        # if abs(numpy.nanmean(_lon-26.45)) < .1:
        out_dic['vectorial_product'].append(numpy.array(ldot[2:-2]))
        out_dic['scalar_product'].append(numpy.array(lvec[2:-2]))
        out_dic['flux_along'].append(numpy.array(nldot[2:-2]))
        out_dic['flux_across'].append(numpy.array(nlvec[2:-2]))
        out_dic['vel'].append(numpy.array(_vel[2:-2]))
        out_dic['gradient_sst'].append(numpy.array(_gradient_sst[2:-2]))
        out_dic['lon'].append(numpy.array(_lon[2:-2]))
        out_dic['lat'].append(numpy.array(_lat[2:-2]))
        out_dic['flag_front'].append(dic_front['flag_front'][i])
    out_dic['time_coverage_start'] = dic_front['time_coverage_start']
    out_dic['time_coverage_end'] = dic_front['time_coverage_end']
    return out_dic


def process_vel_file(ifile: str, dic_vel: dict, dic_front: dict,
                     fstart: datetime.datetime, fstop: datetime.datetime,
                     gradient_threshold: float,
                     pattern: Optional[str] = 'Fronts',
                     outdir: Optional[str] = './',
                     syntool: Optional[bool] = False,
                     ext: Optional[str] = 'json',
                     save: Optional[bool] = False):
    dic_vel['ums']['array'][numpy.isnan(dic_vel['u']['array'])] = 0
    dic_vel['vms']['array'][numpy.isnan(dic_vel['v']['array'])] = 0
    if dic_front is None:
        return None
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    nfirst_date = tools.datetime2npdate(fstart)
    nlast_date = tools.datetime2npdate(fstop)
    # for it in range(len(dic_vel['time'])):
    #     cond = ((nfirst_date > dic_vel['time'][it])
    #             or (nlast_date <= dic_vel['time'][it]))
    #     if cond is True:
    #         return None
    sdate = tools.npdate2datetime(dic_vel['time'][int(len(dic_vel['time'])/2)])
    sdate = sdate.strftime("%Y%m%dT%H%M%S")
    dic_front_out = comp_front_vel(dic_front, dic_vel,
                                   gradient_threshold=gradient_threshold)
    _bn = os.path.basename(ifile)
    _bn, _ext = os.path.splitext(_bn)
    output = os.path.join(outdir, f'{pattern}_{sdate}_{_bn}.pyo')
    if save is True:
        with open(output, 'wb') as f:
            pickle.dump(dic_front_out, f)

    return dic_front_out, output


def run(par_out: dict, par_fronts: dict, data_type_file: str,
        sdepth: Optional[int] = 0,
        first_date: Optional[str] = '19000101T000000Z',
        last_date: Optional[str] = '20500101T000000Z',
        region: Optional[str] = None, syntool: Optional[bool] = False,
        depth: Optional[int] = 0, ext: Optional[str] = 'json',
        save: Optional[bool] = True):
    logger.debug(f'Fronts directory: {par_fronts["dir_front"]}')
    logging.info(f'Load Fronts in {par_fronts["dir_front"]}')
    _path = os.path.join(par_fronts["dir_front"], '**',
                         f'{par_fronts["pattern"]}*.{ext}')
    listfile = glob.glob(_path, recursive=True)
    if len(listfile) == 0:
        logging.error(f'No fronts found in {_path}')
        sys.exit(1)

    logger.debug(f'Pattern for fronts name: {par_fronts["pattern"]}')
    logger.info('Loading Velocity')
    x = load_parameters.sel_region(region)
    lllon, urlon, lllat, urlat, coords, name = x
    logger.info(f'Load velocity {data_type_file}')
    x = load_parameters.sel_data(data_type_file)
    cpath, cpattern, match, dic_name, depth, data_type, label, time_cov_h = x
    box = [lllon, urlon, lllat, urlat]
    _read = read_utils_xr.read_velocity
    vfirst_date = datetime.datetime.strptime(first_date, '%Y%m%dT%H%M%SZ')
    vlast_date = datetime.datetime.strptime(last_date, '%Y%m%dT%H%M%SZ')

    VEL, coord = _read(cpath, cpattern, match, vfirst_date, vlast_date,
                       box=box, depth=depth[sdepth], dict_name=dic_name,
                       stationary=False, compute_strain=False,
                       compute_ow=False, compute_rv=False)

    for ifile in tqdm.tqdm(listfile):
        dic_front, fstart, fstop = dic_open(ifile, ext, syntool=syntool)
        if dic_front is None:
            continue
        nfirst_date = (tools.datetime2npdate(fstart)
                       - numpy.timedelta64(int(time_cov_h * 60 / 2), 'm'))
        nlast_date = (tools.datetime2npdate(fstop)
                       + numpy.timedelta64(int(time_cov_h * 60 / 2), 'm'))
        _ind = numpy.where((nfirst_date < coord['time'])
                           & (coord['time'] < nlast_date))[0]
        if len(_ind) < 1:
            continue
        VEL_small = {}
        VEL_small['time'] = coord['time'][_ind]
        for key, value in VEL.items():
            if len(numpy.shape(value['array'])) > 2:
                VEL_small[key] = {'array': VEL[key]['array'][_ind, :, :],
                                  'lon': VEL[key]['lon'],
                                  'lat': VEL[key]['lat']}
                VEL_small[key]['array'] = numpy.mean(VEL_small[key]['array'],
                                                     axis=0)
        dic, out = process_vel_file(ifile, VEL_small, dic_front, fstart, fstop,
                         par_fronts["gradient_threshold"], syntool=syntool,
                         ext=ext, pattern=par_out["pattern"],
                         outdir=par_out["outdir"], save=save)
#TODO
#        dic_all[out] = dic
#    file_out = os.path.join(par_out["outdir"], f'{par_out["pattern"]}.pyo')
#    with open(file_out, 'wb') as f:
#        pickle.dump(dic_all, f)

if '__main__' == __name__:
    # Setup logging
    main_logger = logging.getLogger()
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    main_logger.addHandler(handler)
    main_logger.setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_type', type=str, required=True,
                        help='data type of Velocity file input')
    parser.add_argument('--region', default=None, type=str, required=False,
                        help='Region json file name')
    parser.add_argument('--depth', default=0, type=int, required=False,
                        help='Depth index')
    parser.add_argument('--first_date', default='19900101T000000Z', type=str,
                        required=False,
                        help='First time considered for analyses')
    parser.add_argument('--last_date', default='20500101T000000Z', type=str,
                        required=False,
                        help='Last time considered for analyses')
    parser.add_argument('--front_dir', dest='front_dir',
                        type=str, default='',
                        help='Path of the front directory')
    parser.add_argument('--front_pattern', dest='front_pattern',
                        type=str, default='',
                        help='Pattern to name front file')
    parser.add_argument('--gradient_threshold', dest='gradient_threshold',
                        type=float, default=0.05,
                        help='Minimum gradient value to be kept')
    parser.add_argument('--out', dest='outdir',
                        type=str, default='/tmp',
                        help='Path for output dictionnary')
    parser.add_argument('--syntool', dest='syntool',
                        type=bool, default=False,
                        help='Set to True if format is syntool json format')
    parser.add_argument('--ext', dest='ext',
                        type=str, default='json',
                        help='Set extension of file, choose between json or pyo')
    args = parser.parse_args()
    reg = os.path.splitext(os.path.basename(args.region))[0]
    vel = os.path.splitext(os.path.basename(args.data_type))[0]
    par_out = {"pattern": f'frontsvel_{reg}_{vel}_{args.depth}m',
               "outdir": args.outdir}
    par_fronts = {"dir_front": args.front_dir, "pattern": args.front_pattern,
                  "gradient_threshold": args.gradient_threshold}
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)
    run(par_out, par_fronts, args.data_type,
        region=args.region, depth=args.depth, syntool=args.syntool,
        ext=args.ext)
