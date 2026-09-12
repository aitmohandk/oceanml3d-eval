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
Advect fictive particles at a given position
"""


import datetime
import numpy
import os
import pickle
import gzip
from velocity_metrics.reader import read_utils_xr
from velocity_metrics.utils import load_parameters
import velocity_metrics.utils.constant as const
from velocity_metrics.lagrangian import mod_io
from velocity_metrics.lagrangian import mod_advection
import velocity_metrics.lagrangian.general_utils as utils
import tqdm
import logging
import argparse
from typing import Optional
logger = logging.getLogger(__name__)

DATE_FMT = '%Y%m%dT%H%M%SZ'


def run_all(par: str, data_type_file: str, list_drifter_position: str,
            days_of_advection: Optional[float] = 10,
            region: Optional[str] = None, sdepth: Optional[int] = 0,
            output_dir: Optional[str] = None, pattern: Optional[str] = None,
            save_netcdf: Optional[bool] = False
            ) -> dict:
    dico_fictive_drifter = load_parameters.load_file(list_drifter_position)
    dic_all = {}
    for ide, part in tqdm.tqdm(dico_fictive_drifter.items()):
        first_date = datetime.datetime.strptime(part['first_date'], const.FMT)
        last_date = first_date + datetime.timedelta(days=days_of_advection)
        if first_date > datetime.datetime(2019, 12, 20): continue
        kide, _ = ide.split('_')
        dic_d = run(par, data_type_file, first_date, last_date,
                    region=region, sdepth=sdepth,
                    fictive_drifter=[float(part['lon']), float(part['lat'])],
                    output_dir=output_dir, pattern=f'{int(kide):015d}',
                    save_netcdf=save_netcdf)
        last_str = last_date.strftime(const.FMT)
        first_str = part['first_date']
        dic_d['first_date'] = first_date
        dic_d['last_date'] = last_date
        dic_all[f'{int(kide):015d}_{first_str}_{last_str}'] = dic_d
    if output_dir is None:
        output_dir = './'
    os.makedirs(output_dir, exist_ok=True)
    if pattern is not None:
        file_dic = os.path.join(output_dir, f'{pattern}.pyo')
    else:
        file_dic = os.path.splitext(os.path.basename(data_type_file))[0]
        if region is not None:
            _reg = os.path.splitext(os.path.basename(region))[0]
            file_dic = f'{file_dic}_{_reg}_dep{sdepth}'
        file_dic = os.path.join(output_dir, f'{file_dic}.pyo')
    logger.info(f'Saving pickle in {file_dic}')
    with gzip.open(f'{file_dic}.gz', 'wb') as f:
        pickle.dump(dic_all, f)
    return dic_all


def run_all_load_once(par: str, data_type_file: str,
                      list_drifter_position: str,
                      days_of_advection: Optional[float] = 10,
                      region: Optional[str] = None, sdepth: Optional[int] = 0,
                      output_dir: Optional[str] = None,
                      pattern: Optional[str] = None,
                      first_date: Optional[str] = '19000101T000000Z',
                      last_date: Optional[str] = '20500101T000000Z',
                      save_netcdf: Optional[bool] = False
                      ) -> dict:
    dico_fictive_drifter = load_parameters.load_file(list_drifter_position)
    vfirst_date = datetime.datetime.strptime(first_date, DATE_FMT)
    vlast_date = datetime.datetime.strptime(last_date, DATE_FMT)
    dic_all = {}
    logger.info('Loading Velocity')
    x = load_parameters.sel_region(region)
    lllon, urlon, lllat, urlat, coords, name = x

    logger.info(f'Load velocity {data_type_file}')
    x = load_parameters.sel_data(data_type_file)
    cpath, cpattern, match, dic_name, depth, data_type, label, time_cov_h = x
    box = [lllon, urlon, lllat, urlat]
    _read = read_utils_xr.read_velocity
    VEL, coord = _read(cpath, cpattern, match, vfirst_date, vlast_date,
                       box=box, depth=depth[sdepth], dict_name=dic_name,
                       stationary=False, compute_strain=False,
                       compute_ow=False, compute_rv=False)
    if pattern is None:
        pattern2 = f'Advection_{data_type}'
    else:
        pattern2 = f'Advection_{data_type}_{pattern}'
    npfmt = '%Y-%m-%dT%H:%M:%S'

    ### ADD BY TP ####
    # Use the box to filter the drifter 
    
    dico_fictive_drifter = {
        key: value
        for key, value in dico_fictive_drifter.items()
        if lllon <= value['lon'] <= urlon and lllat <= value['lat'] <= urlat
    }

    for ide, part in tqdm.tqdm(dico_fictive_drifter.items()):
        dfirst_date = datetime.datetime.strptime(part['first_date'], const.FMT)
        dlast_date = dfirst_date + datetime.timedelta(days=days_of_advection)
        if dlast_date >= vlast_date: continue
        nfirst_date = numpy.datetime64(dfirst_date.strftime(npfmt))
        # nfirst_date = numpy.datetime64(nfirst_date)
        nlast_date = numpy.datetime64(dlast_date.strftime(npfmt))
        # nlast_date = numpy.datetime64(nlast_date)
        _ind = numpy.where((nfirst_date <= coord['time'])
                           & (coord['time'] <= nlast_date))[0]
        if len(_ind) < 2: continue
        VEL_small = {}
        coord_small = coord.copy()
        coord_small['time'] = coord['time'][_ind]
        for key, value in VEL.items():
            if len(numpy.shape(value['array'])) > 2:
                VEL_small[key] = {'array': VEL[key]['array'][_ind, :, :],
                                  'lon': VEL[key]['lon'],
                                  'lat': VEL[key]['lat']}

        kide, _ = ide.split('_')
        dic_global_attr = {'ide': kide, 'pattern': pattern2,
                           'data_type': data_type,
                           'depth': depth[sdepth], 'label': label,
                           }
        dic_d = drifter(par, VEL_small, dfirst_date, dlast_date, coord_small,
                        dic_global_attr=dic_global_attr,
                        fictive_drifter=[float(part['lon']), float(part['lat'])],
                        output_dir=output_dir, verbose=False,
                        pattern=f'{int(kide):015d}', save_netcdf=save_netcdf)
        for key, value in dic_global_attr.items():
            dic_d[key] = value
        last_str = dlast_date.strftime(const.FMT)
        first_str = part['first_date']
        dic_d['first_date'] = dfirst_date
        dic_d['last_date'] = dlast_date
        dic_all[f'{int(kide):015d}_{first_str}_{last_str}'] = dic_d
    if output_dir is None:
        output_dir = './'
    os.makedirs(output_dir, exist_ok=True)
    if pattern is not None:
        file_dic = os.path.join(output_dir, f'{pattern}.pyo')
    else:
        file_dic = os.path.splitext(os.path.basename(data_type_file))[0]
        if region is not None:
            _reg = os.path.splitext(os.path.basename(region))[0]
            file_dic = f'{file_dic}_{_reg}_dep{sdepth}'
        file_dic = os.path.join(output_dir, f'{file_dic}.pyo')
    logger.warn(f'Saving pickle in {file_dic}')
    with gzip.open(f'{file_dic}.gz', 'wb') as f:
        pickle.dump(dic_all, f)
    return dic_all


def run(par: str, data_type_file: str, first_date: datetime.datetime,
        last_date: datetime.datetime, region: Optional[str] = None,
        sdepth: Optional[int] = 0, fictive_drifter: Optional[list] = None,
        output_dir: Optional[str] = None, pattern: Optional[str] = None,
        save_netcdf: Optional[bool] = True) -> dict:
    logger.info('Loading Velocity')
    x = load_parameters.sel_region(region)
    lllon, urlon, lllat, urlat, coords, name = x

    logger.info(f'Load velocity {data_type_file}')
    x = load_parameters.sel_data(data_type_file)
    cpath, cpattern, match, dic_name, depth, data_type, label, time_cov_h = x
    box = [lllon, urlon, lllat, urlat]
    _read = read_utils_xr.read_velocity
    VEL, coord = _read(cpath, cpattern, match, first_date, last_date,
                       box=box, depth=depth[sdepth], dict_name=dic_name,
                       stationary=False, compute_strain=False,
                       compute_ow=False, compute_rv=False)
    fictive_drifter = [float(x) for x in fictive_drifter]
    if pattern is None:
        pattern2 = f'Advection_{data_type}'
    else:
        pattern2 = f'Advection_{data_type}_{pattern}'
    dic_global_attr = {'ide': kide, 'pattern': pattern2,
                       'data_type': data_type,
                       'depth': depth[sdepth], 'label': label,
                       }
    dic_drift = drifter(par, VEL, first_date, last_date, coord,
                        dic_global_attr=dic_global_attr,
                        fictive_drifter=fictive_drifter, output_dir=output_dir,
                        pattern=pattern2, save_netcdf=save_netcdf)
    for key, value in dic_global_attr.items():
        dic_drift[key] = value
    if save_netcdf is False:
        return dic_drift
    else:
        return None


def drifter(parameter_file: str, VEL: dict, first_date: datetime.datetime,
            last_date: datetime.datetime, coord: dict,
            parallelisation: Optional[bool] = False,
            save_netcdf: Optional[bool] = False,
            fictive_drifter: Optional[list] = None,
            output_dir: Optional[str] = None,
            dic_global_attr: Optional[dict] = {},
            pattern: Optional[str] = 'Advection',
            verbose: Optional[bool] = False) -> dict:
    # - Initialize variables from parameter file
    # ------------------------------------------
    _load = load_parameters.load_advection_parameters
    par_advection, par_grid, par_output = _load(parameter_file)
    par_advection["first_date"] = first_date
    par_advection["last_date"] = last_date
    par_advection["reference"] = datetime.datetime.strptime(par_advection["reference"], const.FMT)
    if output_dir is None:
        par_output["outdir"] = output_dir
    if pattern is None:
        par_output["pattern"] = pattern
    if fictive_drifter is not None:
        n = par_grid["size"] / 2
        par_grid["parameter_grid"] = [fictive_drifter[0] - n - par_grid["dx"],
                                      fictive_drifter[0] + n + par_grid["dx"],
                                      par_grid["dx"],
                                      fictive_drifter[1] - n - par_grid["dy"],
                                      fictive_drifter[1] + n + par_grid["dy"],
                                      par_grid["dy"]]
    comm = None
    parallelisation, size, rank, comm = utils.init_mpi(parallelisation)

    #print(rank)
    #print(size)

    # - Load tracer and make or read output grid
    # ------------------------------------------
    if rank == 0:
        logger.info(f'Start time {datetime.datetime.now()}')
        logger.info(f'Loading grid for advection for processor {rank}')
        # - Read or make advection grid
        if par_grid["make_grid"] is False:
            grid = mod_io.read_points(par_grid)
        else:
            grid = mod_io.make_grid(par_grid, VEL, coord)
        grid.lon = numpy.mod(grid.lon + 180, 360) - 180
        # Make a list of particles out of the previous grid
        utils.make_list_particles(grid)

        # - Read tracer to collocate
        list_tracer = None
        par_tracer = {}
        if list_tracer is not None:
            dict_tracer = utils.available_tracer_collocation()
            logger.info('Loading tracer')
            listTr = list(mod_io.read_list_tracer(par_tracer, dict_tracer))
            logger.info('Loading tracer grid')
            listGr = list(mod_io.read_list_grid(par_tracer, dict_tracer))
        else:
            listTr = None
            listGr = None
    else:
        grid = None
        listTr = None
        listGr = None
    if parallelisation is True:
        grid = comm.bcast(grid, root=0)
        listTr = comm.bcast(listTr, root=0)
        listGr = comm.bcast(listGr, root=0)

    # - Read velocity
    dic_vel = None
    if rank == 0:
        logger.info('Loading Velocity')
        dic_vel = mod_io.interp_vel(VEL, coord)
    if parallelisation is True:
        dic_vel = comm.bcast(dic_vel, root=0)

    # - Initialise empty variables and particles
    init = utils.init_empty_variables(par_advection, par_output, grid, listTr,
                                      size, rank)
    dim_hr, dim_lr, grid_size, reducepart, i0, i1 = init
    #print(init)
    # - Perform advection
    try:
        list_var_adv = mod_advection.advection(reducepart, dic_vel, par_advection,
                                               par_output, i0, i1, listGr,
                                               grid, rank=rank, size=size,
                                               verbose=verbose)
    except:
        logger.error(f'Error in advection for drifter {ide}')
        return None
    dim_hr[0] = numpy.shape(list_var_adv['lon_hr'])[0]
    dim_lr[0] = numpy.shape(list_var_adv['lon_lr'])[0]
    # - Save output in netcdf file
    if parallelisation is True:
        drifter = utils.gather_data_mpi(par_advection, list_var_adv, listGr,
                                        listTr, dim_lr, dim_hr, comm, rank,
                                        size, grid_size)
    else:
        drifter = utils.gather_data(par_advection, list_var_adv, listGr,
                                    listTr)

    if rank == 0:
        if save_netcdf is True:
            mod_io.write_drifter(par_advection, par_output, drifter, listTr,
                                 output_dir=par_output["outdir"],
                                 dic_global_attr=dic_global_attr,
                                 file_pattern=pattern)
        end_time = datetime.datetime.now()
        logger.info(f'End time {end_time}')
    return drifter


if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('parameter_file', type=str,
                        help='Parameter file to define advection, output parameters')
    parser.add_argument('-d', '--data_type', type=str, required=True,
                        help='List of data type to consider for analyses')
    parser.add_argument('--region', default='T1', type=str, required=False,
                        help='Region json file name')
    parser.add_argument('--depth', default=0, type=int, required=False,
                        help='Depth index')
    parser.add_argument('--coordinate', default=None, type=float, nargs=2,
                        required=False,
                        help='Coordinate to launch ensemble of drifters')
    parser.add_argument('--first_date', default='19900101T000000Z', type=str,
                        required=False,
                        help='First time considered for analyses')
    parser.add_argument('--last_date', default='20500101T000000Z', type=str,
                        required=False,
                        help='Last time considered for analyses')
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
    print(first_date, last_date)
    run(args.parameter_file, args.data_type, first_date, last_date,
        region=args.region, fictive_drifter=args.coordinate, sdepth=args.depth)
