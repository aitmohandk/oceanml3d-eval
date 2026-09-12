# vim: ts=4:sts=4:sw=4
#
# @author lucile.gaultier@oceandatalab.com
# @date 2023-09-01
#
# Copyright (C) 2022-2023 OceanDataLab
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
Read netcdf files and aggregate them in time using xarray
"""


import numpy
import os
import sys
import datetime
import glob
import xarray
import velocity_metrics.utils.constant as const
import velocity_metrics.utils.tools as tools
from typing import Tuple, Optional
import logging
logger = logging.getLogger(__name__)


DIC_NAME = {'lon': 'lon', 'lat': 'lat', 'u': 'u', 'v': 'v'}


def sort_files(input_dir: str, pattern: str, pMATCH,
               first_date: datetime.datetime, last_date: datetime.datetime,
               stationary: bool) -> Tuple[list, list, float]:
    """
    Sort velocity files in time and check regularity in time
    Args:
        input_dir (str): Input directory that contains files
        pattern (str): pattern to apply to filter file name
        first_date (datetime.datetime): first time in list
        last_date (datetime.datetime): last time in list
        stationart (bool): True if velocity field is constant in time
    Returns
        list of file names, list of dates and frequency between to files
        (in seconds)
    """

    list_file = sorted(glob.glob(os.path.join(input_dir, f'*{pattern}*'),
                                 recursive=True))
    list_date = []
    list_name = []
    frequency = []
    if not list_file:
        logger.error(f'No file found in {input_dir} with pattern *{pattern}*')
    for ifile in list_file:
        match = None
        if pMATCH is not None:
            match = pMATCH(ifile)
        if match is None:
            logger.debug('Listing all files in the directory as match is None')
            list_name.append(ifile)
        else:
            _date = datetime.datetime(int(match.group(1)), int(match.group(2)),
                                      int(match.group(3)))
            fdate = first_date
            ldate = last_date
            if first_date > last_date:
                fdate = last_date
                ldate = first_date
            if _date >= fdate and _date <= ldate:
                list_date.append(_date)
                list_name.append(ifile)
    if not list_name:
        logger.error(f'{pattern} files not found in {input_dir}')
        sys.exit(1)
    # The frequency between the grids must be constant.
    if len(list_date) > 2:
        s = len(list_date) - 1
        # _ind = numpy.argsort(list_date)
        # list_name = list_name[_ind]
        diff = [(list_date[x + 1] - list_date[x]).total_seconds()
                for x in range(s)]
        # _ind = numpy.where(numpy.array(diff) > 86400)

        frequency = list(set(diff))
    if frequency:
        frequency = frequency[0]
    if (stationary) is True and list_date:
        list_name = [list_name[0]]
        list_date = [list_date[0]]
    # if len(frequency) != 1:
    #    raise RuntimeError(f"Time series does not have a constant step
    # between two grids: {frequency} seconds")
    return list_name, list_date, frequency


def check_crossing(lon1: float, lon2: float):
    """
    Assuming a minimum travel distance between two provided longitude,
    checks if the 180th meridian (antimeridian) is crossed.
    Args:
        lon1 (float): First point longitude
        lon2(float): Second point longitude
    Returns:
        Boolean, true if IDL is crossed
    """
    if lon1 > 180:
        lon1 -= 360
    if lon2 > 180:
        lon2 -= 360
    return abs(lon2 - lon1) > 180.0


def read_velocity(input_dir: str, pattern: str, MATCH,
                  first_date: datetime.datetime, last_date: datetime.datetime,
                  box: Optional[list] = [0, 360, -90, 90],
                  depth: Optional[float] = None,
                  dict_name: Optional[dict] = DIC_NAME,
                  dict_dim: Optional[dict] = None,
                  arakawa: Optional[bool] = False,
                  stationary: Optional[bool] = False,
                  compute_strain: Optional[bool] = True,
                  compute_rv: Optional[bool] = True,
                  compute_ow: Optional[bool] = True,
                  ):

    """
    Read and aggregate all velocity file in a time range using xarray
    Args:
        input_dir (str): Input data directory
        pattern (str): Pattern to filter filenames
        MATCH (re): Regex to mach for filtering purposes
        first_date (datetime.datetime): First date in time serie to process
        last_date (datetime.datetime): Last date in time serie to process
        box (list): box area [lon_min, lon_max, lat_min, lat_max], default is
                    global
        depth (float): Depth to select if 4d field
        dict_name (dict): Dictionary for field name in netcdf file, default is
                          DIC_NAME
        stationary (bool): True if velocity is considered stationary (one field
                           is loaded, default is False
        compute_strain (bool): True to compute strain, add ss and sn variables
                                to output dictionary, default is True
        compute_rv (bool): True to compute relative vorticity, add rv varible
                           to output dictionary, default is True
        compute_ow (bool): True to compute Okubo Weiss, add ss and sn
                           variables, default is True
    Returns
        Dictionary with coordinates, velocities and relevant diagnostics
    """
    print("Domaine : ")
    print(box)
    # Make_list_velocity
    list_vel, list_date, freq = sort_files(input_dir, pattern, MATCH,
                                           first_date, last_date, stationary)
    if len(list_vel) == 0:
        msg = (f'{input_dir} does not contain any *{pattern}* in time range')
        logger.error(msg)
    # Read velocity
    for key in ('lon', 'lat', 'time', 'depth'):
        if key not in dict_name.keys():
            dict_name[key] = key
    ds2 = xarray.open_mfdataset(list_vel, concat_dim=dict_name["time"], combine="nested")
    IDL = check_crossing(box[0], box[1])
    IDL = False

    name_lon = dict_name['lon']
    name_lat = dict_name['lat']
    _box1 = box[1] #lon_max
    _box0 = box[0] #lon_min
    #ds2 = ds2.drop_vars(("lat_bnds", "lon_bnds"), errors="ignore")
    ds2 = ds2.drop_dims("nv", errors="ignore")
    if arakawa is True:
        if depth is None:
            ds2 = ds2.transpose(dict_name['time'], dict_name['lat_u'],
                               dict_name['lon_u'], missing_dims="ignore")
            ds2 = ds2.transpose(dict_name['time'], dict_name['lat_v'],
                                dict_name['lon_v'], missing_dims="ignore")
        else:
            ds2 = ds2.transpose(dict_name['time'], dict_name['depth'],
                                dict_name['lat_u'], dict_name['lon_u'],
                                missing_dims="ignore")
            ds2 = ds2.transpose(dict_name['time'], dict_name['depth'],
                                dict_name['lat_v'], dict_name['lon_v'],
                                missing_dims="ignore")

    else:
        print(depth)
   #     if depth is None:
   #         ds2 = ds2.transpose(dict_name['time'], name_lat, name_lon,
   #                             missing_dims="ignore")
   #     else:
   #         ds2 = ds2.transpose(dict_name['time'], dict_name['depth'],
   #                             name_lat, name_lon, missing_dims="ignore")
    #if (lon_ctor.values > 185).any():
    #    _box0 = numpy.mod(box[0] + 360, 360)
    #    _box1 = numpy.mod(box[1] + 360, 360)
    #    if box[0] > box[1]:
    #        IDL = True
    if True: #if dict_dim:
        ds = xarray.Dataset()
        udata = ds2[dict_name['u']].data
        shapeu = numpy.shape(udata)

        uattr = ds2[dict_name['u']].attrs
        vdata = ds2[dict_name['v']].data
        vattr = ds2[dict_name['v']].attrs
        timedata = ds2[dict_name['time']].data
        timeattr =  ds2[dict_name['time']].attrs
        ac = ds.assign_coords
        if arakawa is True:
            lonudata = ds2[dict_name['lon_u']].data
            lonuattr = ds2[dict_name['lon_u']].attrs
            lonvdata = ds2[dict_name['lon_v']].data
            lonvattr = ds2[dict_name['lon_v']].attrs
            latudata = ds2[dict_name['lat_u']].data
            latuattr = ds2[dict_name['lat_u']].attrs
            latvdata = ds2[dict_name['lat_v']].data
            latvattr = ds2[dict_name['lat_v']].attrs
            if udata.shape[-1] != lonudata.shape[-1]:
                udata = numpy.moveaxis(udata, -1, -2)
                uattr = {}
                vdata = numpy.moveaxis(vdata, -1, -2)
                vattr = {}
            if depth is not None:
                ds = ac(coords={'time': (['time'], timedata, timeattr),
                                'lon_u': (['lon_u'],lonudata, lonuattr),
                                'lon_v': (['lon_v'], lonvdata, lonvattr),
                                'lat_u': (['lat_u'], latudata, latuattr),
                                'lat_v': (['lat_v'], latvdata, latvattr),
                                'depth': (['depth'], ds2[dict_name['depth']].data, ds2[dict_name['depth']].attrs)
                                                })
                ds = ds.assign({dict_name['u']: (['time', 'depth', 'lat_u', 'lon_u'], udata, uattr),
                               dict_name['v']: (['time', 'depth', 'lat_v', 'lon_v'], vdata, vattr),
                               })
            else:
                ds = ac(coords={'time': (['time'], timedata, timeattr),
                                'lon_u': (['lon_u'],lonudata, lonuattr),
                                'lon_v': (['lon_v'], lonvdata, lonvattr),
                                'lat_u': (['lat_u'], latudata, latuattr),
                                'lat_v': (['lat_v'], latvdata, latvattr),
                                })
                ds = ds.assign({dict_name['u']: (['time', 'lat_u', 'lon_u'],  udata, uattr),
                               dict_name['v']: (['time', 'lat_v', 'lon_v'],  vdata, vattr),
                               })
        else:
            londata = ds2[dict_name['lon']].data
            lonattr = ds2[dict_name['lon']].attrs
            latdata = ds2[dict_name['lat']].data
            latattr = ds2[dict_name['lat']].attrs
            if udata.shape[-1] != londata.shape[-1]:
                udata = numpy.moveaxis(udata, -1, -2)
                uattr = {}
                vdata = numpy.moveaxis(vdata, -1, -2)
                vattr = {}
            if depth is not None:
                ds = ac(coords={'time': (['time'], timedata, timeattr),
                                'lon': (['lon'], londata, lonattr),
                                'lat': (['lat'], latdata, latattr),
                                'depth': (['depth'], ds2[dict_name['depth']].data, ds2[dict_name['depth']].attrs)
                                                    })
                ds = ds.assign({dict_name['u']: (['time', 'depth', 'lat', 'lon'], udata, uattr),
                               dict_name['v']: (['time', 'depth', 'lat', 'lon'], vdata, vattr),
                                   })
            else:
                ds = ac(coords={'time': (['time'], timedata, timeattr),
                                'lon': (['lon'], londata, lonattr),
                                'lat': (['lat'], latdata, latattr),
                                })
                ds = ds.assign({dict_name['u']: (['time', 'lat', 'lon'],udata, uattr),
                               dict_name['v']: (['time', 'lat', 'lon'], vdata, vattr),
                               })
    #else:
    #    ds = ds2
    if arakawa is True:
        lonu_ctor = getattr(ds, 'lon_u')
        lonv_ctor = getattr(ds, 'lon_v')
    else:
        lonu_ctor = getattr(ds, 'lon')
        lonv_ctor = getattr(ds, 'lon')
    if IDL is True:
        ds = ds.sortby(numpy.mod(lonu_ctor + 360, 360))
        ds = ds.sortby(numpy.mod(lonv_ctor + 360, 360))
        lonu_ctor = numpy.mod(lonu_ctor + 360, 360)
        lonv_ctor = numpy.mod(lonv_ctor + 360, 360)
        ds = ds.sortby(ds['lat'])

    else:
        #TODO arakawa
        ds['lon'] = numpy.mod(ds['lon'] + 180, 360) - 180
        ds = ds.sortby(ds['lon'])
        ds['lon'] = numpy.mod(ds['lon'] + 180, 360) - 180
        ds = ds.sortby(ds['lon'])
        ds = ds.sortby(ds['lat'])
    #    ds = ds2
    if arakawa is True:
        lonu_ctor = getattr(ds, 'lon_u')
        lonv_ctor = getattr(ds, 'lon_v')
    else:
        lonu_ctor = getattr(ds, 'lon')
        lonv_ctor = getattr(ds, 'lon')
    if depth is not None:
        if arakawa is True:
            VEL = ds.sel(depth=depth, lat_u=slice(box[2], box[3]),
                         lon_u=slice(box[0], box[1]),
                         lat_v=slice(box[2], box[3]),
                         lon_v=slice(box[0], box[1]))
        else:
            VEL = ds.sel(depth=depth, lat=slice(box[2], box[3]),
                         lon=slice(_box0, _box1))
    else:
        if arakawa is True:
            VEL = ds.sel(lat_u=slice(box[2], box[3]),
                         lat_v=slice(box[2], box[3]),
                         lon_u=slice(box[0], box[1]),
                         lon_v=slice(box[0], box[1]))
        else:
            VEL = ds.sel(lat=slice(box[2], box[3]),
                         lon=slice(box[0], box[1]))
        if 'depth' in VEL.dims:
            VEL = VEL.isel(depth=0)


    # Intialize empty matrices
    ds2.close()
    ds.close()
    del ds
    coord = {}
    dic = {}
    #coord['lonu'] = numpy.mod(lon_ctor[:].values + 360., 360.)
    if arakawa is True:
        coord['lonu'] = VEL['lon_u'].values
        coord['latu'] = VEL['lat_u'].values
        #coord['lonv'] = numpy.mod(lon_ctor[:].values + 360., 360.)
        coord['lonv'] = VEL['lon_v'].values
        coord['latv'] = VEL['lat_v'].values
    else:
        coord['lonu'] = VEL['lon'].values
        coord['latu'] = VEL['lat'].values
        #coord['lonv'] = numpy.mod(lon_ctor[:].values + 360., 360.)
        coord['lonv'] = VEL['lon'].values
        coord['latv'] = VEL['lat'].values
    if (not coord['lonu'].any()) or (not coord['latu'].any()):
        logger.error('Check box or parameter_grid keys in your parameter file'
                     ' as no data are found in your area')
        sys.exit(1)
    # TODO check this format?
    if len(lonu_ctor.shape) == 2:
        lon2du, lat2du = (coord['lonu'], coord['latu'])
        lon2dv, lat2dv = (coord['lonv'], coord['latv'])
    else:
        lon2du, lat2du = numpy.meshgrid(coord['lonu'], coord['latu'])
        lon2dv, lat2dv = numpy.meshgrid(coord['lonv'], coord['latv'])
    # coord['lon2du'] = numpy.mod(lon2du + 360., 360.)
    # coord['lat2du'] = lat2du
    # coord['lon2dv'] = numpy.mod(lon2du + 360., 360.)
    # coord['lon2dv'] = lat2dv
    coord['time'] = [numpy.datetime64(x) for x in VEL.time.values]
    coord['time'] = numpy.array(coord['time'])
    # Mask data
    # VEL.fillna(0)
    uo_ctor = getattr(VEL, dict_name['u'])
    vo_ctor = getattr(VEL, dict_name['v'])
    dic['ums'] = {'array': uo_ctor.values, 'lon': coord['lonu'],
                  'lat': coord['latu']}
    dic['vms'] = {'array': vo_ctor.values, 'lon': coord['lonv'],
                  'lat': coord['latv']}
    dic['u'] = {'array': numpy.zeros(uo_ctor.shape), 'lon': coord['lonu'],
                'lat': coord['latu']}
    dic['v'] = {'array': numpy.zeros(vo_ctor.shape), 'lon': coord['lonv'],
                'lat': coord['latv']}
    if 'h' in dict_name.keys():
        ho_ctor = getattr(VEL, dict_name['h'])
        dic['h'] = {'array': ho_ctor, 'lon': coord['lonu'],
                    'lat': coord['latv']}
    if compute_strain or compute_rv or compute_ow:
        dic['sn'] = {'array': numpy.zeros(uo_ctor.shape), 'lon': coord['lonu'],
                     'lat': coord['latv']}
        dic['ss'] = {'array': numpy.zeros(uo_ctor.shape), 'lon': coord['lonu'],
                     'lat': coord['latv']}
        dic['rv'] = {'array': numpy.zeros(uo_ctor.shape), 'lon': coord['lonu'],
                     'lat': coord['latv']}
    for t in range(VEL.time.shape[0]):
        if stationary is False:
            perc = float(t / (VEL.time.shape[0]))
            tools.update_progress(perc, '', '')
        # Convert velocity from m/s to degd
        utmp, vtmp = tools.ms2degd(lon2du, lat2du,
                                   dic['ums']['array'][t, :, :],
                                   dic['vms']['array'][t, :, :])
        dic['u']['array'][t, :, :] = utmp
        dic['v']['array'][t, :, :] = vtmp
        if compute_strain or compute_rv or compute_ow:
            mlat = numpy.deg2rad(numpy.mean((lat2du + lat2dv)/2))
            gfo = (const.g / numpy.sin(mlat) / (2 * const.omega))
            dxlatu = (lat2du[2:, 1: -1] - lat2du[: -2, 1: -1])
            dylonu = (lon2du[1: -1, 2:] - lon2du[1: -1, : -2])
            dxlatv = (lat2dv[2:, 1: -1] - lat2dv[:-2, 1: -1])
            dylonv = (lon2dv[1: -1, 2:] - lon2dv[1: -1, : -2])
            dyutmp = utmp[1: -1, 2:] - utmp[1: -1, : -2]
            dyvtmp = vtmp[1: -1, 2:] - vtmp[1: -1, : -2]
            dxutmp = utmp[2:, 1: -1] - utmp[:-2, 1: -1]
            dxvtmp = vtmp[2:, 1: -1] - vtmp[:-2, 1: -1]
        if compute_rv or compute_ow:
            RV = gfo * (dyvtmp / dylonv - dxutmp / dxlatu)
            dic['rv']['array'][t, 1: -1, 1: -1] = RV
        if compute_strain or compute_ow:
            Sn = gfo * (dyutmp / dylonu - dxvtmp / dxlatv)
            Ss = gfo * (dxutmp / dxlatu + dyvtmp / dylonv)
            dic['sn']['array'][t, 1: -1, 1: -1] = Sn
            dic['ss']['array'][t, 1: -1, 1: -1] = Ss
    
    #print(VEL)
    #sys.exit()

    del VEL, lon2du, lat2du
    return dic, coord
