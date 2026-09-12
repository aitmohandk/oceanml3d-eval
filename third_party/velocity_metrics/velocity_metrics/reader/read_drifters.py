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

"""This module provides function to load drifters from CMEMS and AOML and
convert them to pyo objects

"""

import netCDF4
import pickle
import json
import logging
from typing import Optional
from shapely.geometry import Polygon, Point
import numpy
import datetime
import tqdm
import os
import glob
import argparse
import gzip
from velocity_metrics.utils.load_parameters import sel_region


logger = logging.getLogger(__name__)
DATE_FMT = "%Y%m%dT%H%M%SZ"


def load_drifter_cmems(path: str, depth: float) -> dict:
    dic = {}

    return dic


def load_drifter_aoml(path: str, depth: float, date_start: datetime.datetime,
                      date_end: datetime.datetime, coords: list,
                      outfile: Optional[str] = None):
    '''Load NETCDF hourly or 6 hourly drifters from AOML and save points that
    cross the area in a gzip pickle file
    ARGS:
        path (str): Directory that contains netcdf files
        depth (float): Depth to consider (0 or 15)
        date_start (datetime): First date to include in dataset
        date_end (datetime): Last date to include in dataset
        coords (list): point to build polygon ((lon1, lat1), (lon2, lat2), ...)
        outfile (str): Filename for pickle file (ending with pyo.gzip)
    RETURN
        None, Save dataset in gzip pickle file under the file name outfile
    '''
    listfile = glob.glob(os.path.join(path, '*.nc'))
    dic_out = {}
    poly = Polygon(coords)
    IDL = False
    if (numpy.array(coords) > 180).any():
        IDL = True
    for ifile in tqdm.tqdm(listfile):
        dic = {}
        try:
            fid = netCDF4.Dataset(ifile, 'r')
            fid.close()
        except OSError as e:
            logger.error(f'OS error: {e}')
            continue

        fid = netCDF4.Dataset(ifile, 'r')
        iid = fid['WMO'][0]
        time = fid['time'][0, :]
        start = fid['start_date'][0]
        try:
            dstart = netCDF4.num2date(start, units=fid['start_date'].units,
                                      only_use_cftime_datetimes=False,
                                      only_use_python_datetimes=True)
        except: # AttributeError:
            dstart = datetime.datetime(1990, 1, 1)
        end = fid['end_date'][0]
        try:
            dend = netCDF4.num2date(end, units=fid['end_date'].units,
                                    only_use_cftime_datetimes=False,
                                    only_use_python_datetimes=True)
        except: # AttributeError:
            dend = datetime.datetime(2050, 1, 1)
        lost_drogue = fid['drogue_lost_date'][0]
        if dstart > date_end:
            continue
        if dend < date_start:
            continue
        if depth > 0:
            _ind = numpy.where((time >= start) & (time < end)
                               & (time < lost_drogue))[0]
        else:
            _ind = numpy.where((time >= start) & (time < end)
                               & (time > lost_drogue))[0]
        if len(_ind) == 0:
            continue
        time = time[_ind]
        lon = fid['longitude'][0, _ind]
        if IDL is True:
            lon360 = fid['longitude360'][0, _ind]
            lon = lon360
        lat = fid['latitude'][0, _ind]
        ve = fid['ve'][0, _ind]
        vn = fid['vn'][0, _ind]
        date = netCDF4.num2date(time, units=fid['time'].units,
                                only_use_cftime_datetimes=False,
                                only_use_python_datetimes=True)
        _ind = numpy.where((date >= date_start) & (date <= date_end))[0]
        if len(_ind) == 0:
            continue
        dic_tmp = {'time': numpy.array(time[_ind]),
                   'lon': numpy.array(lon[_ind], dtype=float),
                   'lat': numpy.array(lat[_ind], dtype=float),
                   'vms': numpy.array(vn[_ind], dtype=float),
                   'ums': numpy.array(ve[_ind], dtype=float),
                   'date': numpy.array(date[_ind])}
        for key in ('time', 'lon', 'lat', 'date', 'ums', 'vms'):
            dic[key] = []
        for i in range(len(dic_tmp['lon'])):
            point = Point(dic_tmp['lon'][i], dic_tmp['lat'][i])
            if point.within(poly) is True:
                for key in ('time', 'lon', 'lat', 'date', 'ums', 'vms'):
                    dic[key].append(dic_tmp[key][i])
        if len(dic['lon']) < 3:
            continue
        fid.close()
        dic_out[f'{iid:015d}'] = dic
    if outfile is not None:
        with gzip.open(outfile, 'wb') as f:
            pickle.dump(dic_out, f)
    return dic_out


def load_drifter_cmems_ARGO(path: str, depth: float, date_start: datetime.datetime,
                            date_end: datetime.datetime, coords: list,
                            outfile: Optional[str] = None):
    '''Load NETCDF hourly or 6 hourly drifters from AOML and save points that
    cross the area in a gzip pickle file
    ARGS:
        path (str): Directory that contains netcdf files
        depth (float): Depth to consider (0 or 15)
        date_start (datetime): First date to include in dataset
        date_end (datetime): Last date to include in dataset
        coords (list): point to build polygon ((lon1, lat1), (lon2, lat2), ...)
        outfile (str): Filename for pickle file (ending with pyo.gzip)
    RETURN
        None, Save dataset in gzip pickle file under the file name outfile
    '''
    listfile = glob.glob(os.path.join(path, '*.nc'))
    dic_out = {}
    poly = Polygon(coords)
    IDL = False
    if (numpy.array(coords) > 180).any():
        IDL = True
    for ifile in tqdm.tqdm(listfile):
        dic = {}
        try:
            fid = netCDF4.Dataset(ifile, 'r')
            fid.close()
        except OSError as e:
            logger.error(f'OS error: {e}')
            continue

        fid = netCDF4.Dataset(ifile, 'r')
        try:
            strtraj = [x.decode() for x in fid['TRAJECTORY'][:] if x]
        except: 
            print(fid['TRAJECTORY'][:], ifile)
            continue

        iid = int(''.join(strtraj))
        #iid = '-'.join((iid, fid.wmo_platform_code))
        time = fid['TIME'][:]
        #start = fid['start_date'][0]
        try:
            dstart = netCDF4.num2date(time[0], units=fid['TIME'].units,
                                      only_use_cftime_datetimes=False,
                                      only_use_python_datetimes=True)
        except: # AttributeError:
            dstart = datetime.datetime(1990, 1, 1)
        #end = fid['end_date'][0]
        try:
            dend = netCDF4.num2date(time[-1], units=fid['TIME'].units,
                                    only_use_cftime_datetimes=False,
                                    only_use_python_datetimes=True)
        except: # AttributeError:
            dend = datetime.datetime(2050, 1, 1)
        grounded = numpy.array([x.decode() for x in fid['GROUNDED'][:]])
        # print(dstart, date_end, dend, date_start)
        if dstart > date_end:
            continue
        if dend < date_start:
            continue
        _ind = numpy.where(grounded == 'N')[0]
        if len(_ind) == 0:
            continue

        time = [time[x] for x in _ind]
        lon = [fid['LONGITUDE'][x] for x in _ind]
        if IDL is True:
            lon360 = numpy.mod(lon + 360, 360)
            lon = lon360
        lat = [fid['LATITUDE'][x] for x in _ind]
        ve = [fid['EWCT'][x, 0] for x in _ind]
        vn = [fid['NSCT'][x, 0] for x in _ind]

        date = netCDF4.num2date(time, units=fid['TIME'].units,
                                only_use_cftime_datetimes=False,
                                only_use_python_datetimes=True)
        _ind = numpy.where((date >= date_start) & (date <= date_end))[0]
        if len(_ind) == 0:
            continue
        _sl = slice(_ind[0], _ind[-1])
        dic_tmp = {'time': numpy.array(time[_sl]),
                   'lon': numpy.array(lon[_sl], dtype=float),
                   'lat': numpy.array(lat[_sl], dtype=float),
                   'vms': numpy.array(vn[_sl], dtype=float),
                   'ums': numpy.array(ve[_sl], dtype=float),
                   'date': numpy.array(date[_sl])}
        for key in ('time', 'lon', 'lat', 'date', 'ums', 'vms'):
            dic[key] = []
        for i in range(len(dic_tmp['lon'])):
            point = Point(dic_tmp['lon'][i], dic_tmp['lat'][i])
            if point.within(poly) is True:
                for key in ('time', 'lon', 'lat', 'date', 'ums', 'vms'):
                    dic[key].append(dic_tmp[key][i])
        if len(dic['lon']) < 3:
            continue
        fid.close()
        dic_out[f'{iid:015d}'] = dic
    if outfile is not None:
        with gzip.open(outfile, 'wb') as f:
            pickle.dump(dic_out, f)
    return dic_out


def load_drifter_cmems(path: str, depth: float, date_start: datetime.datetime,
                      date_end: datetime.datetime, coords: list,
                      outfile: Optional[str] = None):
    '''Load NETCDF hourly or 6 hourly drifters from AOML and save points that
    cross the area in a gzip pickle file
    ARGS:
        path (str): Directory that contains netcdf files
        depth (float): Depth to consider (0 or 15)
        date_start (datetime): First date to include in dataset
        date_end (datetime): Last date to include in dataset
        coords (list): point to build polygon ((lon1, lat1), (lon2, lat2), ...)
        outfile (str): Filename for pickle file (ending with pyo.gzip)
    RETURN
        None, Save dataset in gzip pickle file under the file name outfile
    '''
    listfile = glob.glob(os.path.join(path, 'GL*.nc'))
    dic_out = {}
    poly = Polygon(coords)
    IDL = False
    if (numpy.array(coords) > 180).any():
        IDL = True
    for ifile in tqdm.tqdm(listfile):
        dic = {}
        try:
            fid = netCDF4.Dataset(ifile, 'r')
            fid.close()
        except OSError as e:
            logger.error(f'OS error: {e}')
            continue

        fid = netCDF4.Dataset(ifile, 'r')
        try:
            strtraj = [x.decode() for x in fid['TRAJECTORY'][:] if x]
        except:
            logger.info(f'TRAJECTORY could not be read properly in {ifile}')
            continue
        try:
            iid = ''.join(strtraj)
        except:
            logger.info(f'IID could not be read properly for {ifile}')
            continue
        #iid = '-'.join((iid, fid.wmo_platform_code))
        time = fid['TIME'][:]
        #start = fid['start_date'][0]
        try:
            dstart = netCDF4.num2date(time[0], units=fid['TIME'].units,
                                      only_use_cftime_datetimes=False,
                                      only_use_python_datetimes=True)
        except: # AttributeError:
            dstart = datetime.datetime(1990, 1, 1)
        #end = fid['end_date'][0]
        try:
            dend = netCDF4.num2date(time[-1], units=fid['TIME'].units,
                                    only_use_cftime_datetimes=False,
                                    only_use_python_datetimes=True)
        except: # AttributeError:
            dend = datetime.datetime(2050, 1, 1)
        dep = numpy.array([fid['DEPH'][:]])
        # print(dstart, date_end, dend, date_start)
        if dstart > date_end:
            continue
        if dend < date_start:
            continue
        d_ind = numpy.where(dep == depth)[-1]
        if len(d_ind) == 0:
            continue

        lon = fid['LONGITUDE'][:]
        if IDL is True:
            lon360 = numpy.mod(lon + 360, 360)
            lon = lon360
        lat = fid['LATITUDE'][:]
        masku = numpy.ma.masked_greater(fid['EWCT_QC'][:, d_ind], 3)  
        maskv = numpy.ma.masked_greater(fid['NSCT_QC'][:, d_ind], 3)
        mask = (masku | maskv)
        ve = fid['EWCT'][:, d_ind]
        vn = fid['NSCT'][:, d_ind]
        ve = numpy.ma.array(ve, mask=mask)
        vn = numpy.ma.array(vn, mask=mask)
        isws = False
        if 'EWCT_WS' in fid.variables.keys():
            isws = True
            vews = fid['EWCT_WS'][:, d_ind]
            vews = numpy.ma.array(vews, mask=mask)
            vnws = fid['NSCT_WS'][:, d_ind]
            vnws = numpy.ma.array(vnws, mask=mask)
        date = netCDF4.num2date(time, units=fid['TIME'].units,
                                only_use_cftime_datetimes=False,
                                only_use_python_datetimes=True)
        _ind = numpy.where((date >= date_start) & (date <= date_end))[0]
        if len(_ind) == 0:
            continue
        _sl = slice(_ind[0], _ind[-1])

        dic_tmp = {'time': numpy.array(time[_sl]),
                   'lon': numpy.array(lon[_sl], dtype=float),
                   'lat': numpy.array(lat[_sl], dtype=float),
                   'vms': numpy.array(vn[_sl], dtype=float),
                   'ums': numpy.array(ve[_sl], dtype=float),
                   'date': numpy.array(date[_sl])}
        if isws:
            dic_tmp['vwsms'] = numpy.array(vnws[_sl], dtype=float)
            dic_tmp['uwsms'] = numpy.array(vews[_sl], dtype=float)
            dic['uwsms'] = []
            dic['vwsms'] = []
        for key in ('time', 'lon', 'lat', 'date', 'ums', 'vms'):
            dic[key] = []
        for i in range(len(dic_tmp['lon'])):
            point = Point(dic_tmp['lon'][i], dic_tmp['lat'][i])
            if point.within(poly) is True:
                for key in ('time', 'lon', 'lat', 'date', 'ums', 'vms'):
                    dic[key].append(dic_tmp[key][i])
                if isws:
                    for key in ('uwsms', 'vwsms'):
                        dic[key].append(dic_tmp[key][i])

        if len(dic['lon']) < 3:
            continue
        fid.close()

        dic_out[f'{iid:>15s}'] = dic #015d
    if outfile is not None:
        with gzip.open(outfile, 'wb') as f:
            pickle.dump(dic_out, f)
    return dic_out

def load_drifter_pickle(list_drifter_file: list) -> dict:
    '''
    Load Drifters data in pickle
    Args:
        list_drifter_file (list): list of pyo file name to process
    Returns:
        Dictionnary that contains concatenated data from all drifter files
    '''
    dic_out = {}
    for _file in list_drifter_file:
        if not os.path.exists(_file):
            logger.error(f'file {_file} not found, skipping this file')
        else:
            if 'gz' in os.path.splitext(_file)[-1]:
                with gzip.open(_file, 'rb') as f:
                    dic = pickle.load(f)
            else:
                with open(_file, 'rb') as f:
                    dic = pickle.load(f)
            for key_id in dic.keys():
                if key_id in dic_out.keys():
                    for var_key in dic_out[key_id].keys():
                        dic_out[key_id][var_key].extend(dic[key_id][var_key])
                else:
                    dic_out[key_id] = dic[key_id]
    return dic_out


def fictive_particles_json(pyo_file: str, outpath: str,
                           time_step: Optional[int] = 4):
    with gzip.open(pyo_file, 'rb') as f:
        dic = pickle.load(f)
    dic_total = {}
    for ide, value in dic.items():
        for i in range(0, len(value['date']), time_step):
            date = value['date'][i].strftime('%Y%m%dT%H%M%SZ')
            key = f'{ide}_{date}'
            dic_total[key] = {}
            dic_total[key]['first_date'] = date
            dic_total[key]['lon'] = value['lon'][i]
            dic_total[key]['lat'] = value['lat'][i]
            dic_total[key]['id'] = ide
    os.makedirs(outpath, exist_ok=True)
    fn = os.path.splitext(os.path.basename(pyo_file))[0]
    outfile = f'Fictive_pos_{fn}.json'
    _out = os.path.join(outpath, outfile)
    logger.warning(f'Saving fictive position in {_out}')
    with open(_out, 'w') as f:
        json.dump(dic_total, f)

def run(datatype: str, input_path: str, depth: float, region: str,
        first_date: str, last_date: str, out: str,
        make_json: bool = True,
        time_step: Optional[int] = 4):
    lllon, urlon, lllat, urlat, coords, name = sel_region(region)
    if out is not None:
        reg = os.path.splitext(os.path.basename(region))[0]
        if name == 'global':
            reg = 'GL'
        strdate = f'_{first_date}_{last_date}'
        filename = f'Drifters_{datatype}_{reg}_{depth:02d}m{strdate}.pyo.gz'
        os.makedirs(out, exist_ok=True)
        filename = os.path.join(out, filename)
    else:
        filename = None
        logger.warning('Drifters not saved, provide --out [path] to save them')
    first_date = datetime.datetime.strptime(first_date, DATE_FMT)
    last_date = datetime.datetime.strptime(last_date, DATE_FMT)
    if datatype == 'AOML':
        _ = load_drifter_aoml(input_path, depth, first_date, last_date, coords,
                              filename)
        logger.info('Save fictive particles in json for lagrangian advection')
        if filename is not None and make_json is True:
            fictive_particles_json(filename, out, time_step=time_step)
    elif datatype == 'CMEMS-ARGO':
        _ = load_drifter_cmems_ARGO(input_path, depth, first_date, last_date, coords,
                                    filename)
        logger.info('Save fictive particles in json for lagrangian advection')
        if filename is not None and make_json is True:
            fictive_particles_json(filename, out, time_step=time_step)
    elif datatype == 'CMEMS':
        _ = load_drifter_cmems(input_path, depth, first_date, last_date, coords,
                               filename)
        logger.info('Save fictive particles in json for lagrangian advection')
        if filename is not None and make_json is True:
            fictive_particles_json(filename, out, time_step=time_step)
        logger.error('CMEMS drifter reading not implemented yet')
    else:
        logger.error(f'Wrong Datatype {datatype}, ')
        logger.error('choose beetween "AOML and "CMEMS"')


if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    msg = 'data type to consider for analyses (AOML or CMEMS)'
    parser.add_argument('data_type', type=str, choices=['AOML', 'CMEMS-ARGO', 'CMEMS'],
                        help=msg)
    parser.add_argument('-i, --input_path', type=str, dest='input_path',
                        help='Directory to scan for netCDF files')
    msg = 'Select one of the region_[].json file in the share directory'
    parser.add_argument('-r', '--region', required=True, type=str,
                        help=msg)
    msg = 'First time considered for analyses, format is "%Y%m%dT%H%M%S"'
    parser.add_argument('--first_date', default='19900101T000000Z', type=str,
                        required=False,
                        help=msg)
    msg = 'Last time considered for analyses, format is "%Y%m%dT%H%M%S"'
    parser.add_argument('--last_date', default='20500101T000000Z', type=str,
                        required=False,
                        help=msg)
    parser.add_argument('-d', '--depth', default=0, type=int, required=False,
                        help='Depth index in velocity file')
    parser.add_argument('--out', dest='outdir',
                        type=str, default=None,
                        help='Output path for the pyo file')

    parser.add_argument('--time_step', dest='time_step',
                        type=int, default=4,
                        help='Time step to save fictive particle position')

    parser.add_argument('--make_json', dest='make_json', action='store_true',
                        default=False, required=False,
                        help='True to save fictive particle position')
    parser.add_argument('--verbose', action='store_true', default=False,
                        required=False)
    parser.add_argument('--debug', action='store_true', default=False,
                        required=False)
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.INFO)
    if args.debug:
        logger.setLevel(logging.DEBUG)

    run(args.data_type, args.input_path, args.depth, args.region,
        args.first_date, args.last_date, args.outdir, time_step=args.time_step,
        make_json=args.make_json)
