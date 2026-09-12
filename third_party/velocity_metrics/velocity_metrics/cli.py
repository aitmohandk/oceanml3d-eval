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

"""This module provides methods to run command line and validate velocity
   products

"""

import argparse
import logging
import datetime
import os
import velocity_metrics.spectrum.spectrum as spectrum
import velocity_metrics.lagrangian.drifters as drifters
import velocity_metrics.fronts.compare_fronts_vel as compare_fronts_vel
import velocity_metrics.fronts.box_metrics as box_metrics

import velocity_metrics.eulerian.eulerian_drifters as eulerian_drifters
import velocity_metrics.lagrangian.cumulative_distance as sde
import velocity_metrics.utils.constant as const

# Setup logging
logger = logging.getLogger()
logger.handlers = []
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)
logger.setLevel(logging.WARN)

MSG_PARAMETER_FILE = 'Parameter file to define advection, output parameters'
MSG_DATA_TYPE = 'Data type to consider for analyses'
MSG_REGION = 'Region, select among T1, GS, Mediterranean, Tropics, T2_DTU, T2_SOC, T3_SAR, T3_ARC'
MSG_FIRST_DATE = 'First time considered for analyses, format is {const.FMT}'
MSG_LAST_DATE = 'Last time considered for analyses, format is {const.FMT}'
MSG_DEPTH_INDEX = 'Depth index to retrieve velocity in netCDF file'
MSG_PATH_OUTPUT = 'Path to save outputs'
MSG_FILENAME_OUTPUT = 'Output file name'
MSG_DRIFTER = 'pickle files which contains drifter to consider for analyses'


def run_spectrum():
    parser = argparse.ArgumentParser()
    parser.add_argument('list_data_type', type=str, nargs='+',
                        help=f'List of {MSG_DATA_TYPE}')
    parser.add_argument('-r', '--region', default='T1', type=str,
                        help=MSG_REGION)
    parser.add_argument('--first_date', default='19900101T000000Z', type=str,
                        required=False, help=MSG_FIRST_DATE)
    parser.add_argument('--last_date', default='20500101T000000Z', type=str,
                        required=False, help=MSG_LAST_DATE)
    parser.add_argument('-d', '--depth', default=0, type=int, required=False,
                        help=MSG_DEPTH_INDEX)
    parser.add_argument('-l', '--length', default=1000, type=int, required=False,
                        help='Maximum extent to compute spectrum in km')
    parser.add_argument('--numfiles', default=10000, type=int, required=False,
                        help='Maximum number of data to compute spectrum')
    parser.add_argument('--out', dest='outdir',
                        type=str, default='./',
                        help=f'{MSG_PATH_OUTPUT} figure and python dictionnary'
                        )
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

    _ = spectrum.run(list_data_type, args.region, args.depth,
                     first_date=args.first_date, last_date=args.last_date,
                     output_dir=args.outdir, tmax=args.numfiles, kmin=1/args.length)


def plot_spectrum():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, nargs='+',
                        help='List of spectrum pickle files to plot')
    parser.add_argument('--out', dest='outdir',
                        type=str, default='./',
                        help=f'{MSG_PATH_OUTPUT} figure'
                        )
    parser.add_argument('--filename', dest='filename',
                        type=str, default='spectrum.png',
                        help=f'{MSG_FILENAME_OUTPUT} to save figures'
                        )
    parser.add_argument('--list_color', dest='list_color', nargs='+',
                        type=str, default=None,
                        help=f'List of color to plot data'
                        )
    parser.add_argument('--verbose', action='store_true', default=False,
                        required=False)
    parser.add_argument('--debug', action='store_true', default=False,
                        required=False)

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.INFO)
    if args.debug:
        logger.setLevel(logging.DEBUG)
    _ = spectrum.plot(list(args.input), list_color=list(args.list_color),
                      outfile=os.path.join(args.outdir, args.filename))


def run_drifter():
    parser = argparse.ArgumentParser()
    parser.add_argument('parameter_file', type=str,
                        help=MSG_PARAMETER_FILE)
    parser.add_argument('-d', '--data_type', type=str, required=True,
                        help=MSG_DATA_TYPE)
    parser.add_argument('--region', default='T1', type=str,
                        help=MSG_REGION)
    parser.add_argument('--depth', default=0, type=int, required=False,
                        help=MSG_DEPTH_INDEX)
    parser.add_argument('--coordinate', default=None, type=float, nargs=2,
                        required=False,
                        help='Coordinate to launch ensemble of drifters')
    parser.add_argument('--first_date', default='19900101T000000Z', type=str,
                        required=False, help=MSG_FIRST_DATE)
    parser.add_argument('--last_date', default='20500101T000000Z', type=str,
                        required=False, help=MSG_LAST_DATE)
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
    _ = drifters.run(args.parameter_file, args.data_type, first_date,
                     last_date, region=args.region,
                     fictive_drifter=args.coordinate,
                     sdepth=args.depth)


def run_all_drifters2():
    parser = argparse.ArgumentParser()
    parser.add_argument('parameter_file', type=str,
                        help=MSG_PARAMETER_FILE)
    parser.add_argument('-d', '--data_type', type=str, required=True,
                        help='List of data type to consider for analyses')
    parser.add_argument('--region', default='T1', type=str,
                        help='Region json file name')
    parser.add_argument('--depth', default=0, type=int, required=False,
                        help=MSG_DEPTH_INDEX)
    parser.add_argument('--out', dest='outdir',
                        type=str, default='./',
                        help=f'{MSG_PATH_OUTPUT} for files (pickle/netcdf)')
    parser.add_argument('--days', default=10, type=float,
                        required=False,
                        help='Number of days to advect fictive drifters')
    parser.add_argument('--list_drifter_position', required=True, type=str,
                        help='JSON file which contains dictionnary of drifters')
    parser.add_argument('--verbose', action='store_true', default=False,
                        required=False)
    parser.add_argument('--debug', action='store_true', default=False,
                        required=False)

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.INFO)
    if args.debug:
        logger.setLevel(logging.DEBUG)
    _ = drifters.run_all(args.parameter_file, args.data_type,
                         args.list_drifter_position,
                         days_of_advection=args.days, output_dir=args.outdir,
                         region=args.region, sdepth=args.depth)


def run_all_drifters():
    parser = argparse.ArgumentParser()
    parser.add_argument('parameter_file', type=str,
                        help=MSG_PARAMETER_FILE)
    parser.add_argument('-d', '--data_type', type=str, required=True,
                        help='List of data type to consider for analyses')
    parser.add_argument('--region', default='T1', type=str,
                        help='Region json file name')
    parser.add_argument('--depth', default=0, type=int, required=False,
                        help=MSG_DEPTH_INDEX)
    parser.add_argument('--out', dest='outdir',
                        type=str, default='./',
                        help=f'{MSG_PATH_OUTPUT} for files (pickle/netcdf)')
    parser.add_argument('--first_date', default='19900101T000000Z', type=str,
                        required=False, help=MSG_FIRST_DATE)
    parser.add_argument('--last_date', default='20500101T000000Z', type=str,
                        required=False, help=MSG_LAST_DATE)
    parser.add_argument('--days', default=10, type=float,
                        required=False,
                        help='Number of days to advect fictive drifters')
    parser.add_argument('--list_drifter_position', required=True, type=str,
                        help='JSON file which contains dictionnary of drifters')
    parser.add_argument('--verbose', action='store_true', default=False,
                        required=False)
    parser.add_argument('--debug', action='store_true', default=False,
                        required=False)

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.INFO)
    if args.debug:
        logger.setLevel(logging.DEBUG)
    _ = drifters.run_all_load_once(args.parameter_file, args.data_type,
                         args.list_drifter_position,
                         days_of_advection=args.days, output_dir=args.outdir,
                         region=args.region, sdepth=args.depth,
                         first_date=args.first_date, last_date=args.last_date)


def run_sde():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str,
                        help='Path to pickle or directory that contains netcdf Advection files')
    parser.add_argument('--drifter', type=str, nargs='+',
                        help='{MSG_DRIFTER}')
    parser.add_argument('--out', dest='outdir',
                        type=str, default='./',
                        help=f'{MSG_PATH_OUTPUT} for pickle file')
    parser.add_argument('--filename', dest='filename',
                        type=str, default='Sde',
                        help=f'{MSG_FILENAME_OUTPUT} for pickle file')
    parser.add_argument('--verbose', action='store_true', default=False,
                        required=False)
    parser.add_argument('--plot', action='store_true', default=False,
                        required=False)
    parser.add_argument('--debug', action='store_true', default=False,
                        required=False)

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.INFO)
    if args.debug:
        logger.setLevel(logging.DEBUG)
    _ = sde.run(args.input, args.drifter, output_dir=args.outdir,
                output_filename=args.filename, isplot=args.plot)


def plot_sde():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, nargs='+',
                        help='List of Path to pickle sde files')
    parser.add_argument('--out', dest='outdir',
                        type=str, default='./',
                        help=f'{MSG_PATH_OUTPUT} for pickle file')
    parser.add_argument('--filename', dest='filename',
                        type=str, default='Sde',
                        help=f'{MSG_FILENAME_OUTPUT} for pickle file')
    parser.add_argument('--list_color', dest='list_color', nargs='+',
                        type=str, default=None,
                        help=f'List of color to plot data'
                        )
    parser.add_argument('--verbose', action='store_true', default=False,
                        required=False)
    parser.add_argument('--debug', action='store_true', default=False,
                        required=False)

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.INFO)
    if args.debug:
        logger.setLevel(logging.DEBUG)
    _, _ = sde.plot(list(args.input), output_dir=args.outdir,
                 list_color=list(args.list_color),
                 output_filename=args.filename)


def run_eulerian_drifters():
    parser = argparse.ArgumentParser()
    parser.add_argument('list_drifters', type=str, nargs='+',
                        help='List of pyo files which contains drifter to consider for analyses')
    parser.add_argument('-d', '--data_type', type=str, required=True,
                        help='Parameter file to read velocity to analyse')
    parser.add_argument('--region', default='T1', type=str, required=False,
                        help='Region json file name')
    parser.add_argument('--depth', default=0, type=int, required=False,
                        help='Depth index')
    parser.add_argument('--temporal_filtering', default=1, type=float, required=False,
                        help='Number of days for filtering')
    parser.add_argument('--first_date', default='19900101T000000Z', type=str,
                        required=False,
                        help='First time considered for analyses')
    parser.add_argument('--last_date', default='20500101T000000Z', type=str,
                        required=False,
                        help='Last time considered for analyses')
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
    first_date = datetime.datetime.strptime(args.first_date, const.FMT)
    last_date = datetime.datetime.strptime(args.last_date, const.FMT)
    eulerian_drifters.run(args.list_drifters, args.data_type, first_date,
                          last_date, region=args.region,
                          temporal_filtering=args.temporal_filtering,
                          sdepth=args.depth, output_dir=args.outdir)


def run_compare_front_vel():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_type', type=str, required=True,
                        help=MSG_DATA_TYPE)
    parser.add_argument('--region', default=None, type=str, required=False,
                        help=MSG_REGION)
    parser.add_argument('--depth', default=0, type=int, required=False,
                        help=MSG_DEPTH_INDEX)
    parser.add_argument('--first_date', default='19900101T000000Z', type=str,
                        required=False, help=MSG_FIRST_DATE)
    parser.add_argument('--last_date', default='20500101T000000Z', type=str,
                        required=False, help=MSG_LAST_DATE)
    parser.add_argument('--front_dir', dest='front_dir',
                        type=str, default='',
                        help='Path of the front directory')
    parser.add_argument('--front_pattern', dest='front_pattern',
                        type=str, default='',
                        help='Pattern to name front file')
    parser.add_argument('--gradient_threshold', dest='gradient_threshold',
                        type=float, default=0.05,
                        help='Minimum gradient value to be kept')
    parser.add_argument('--out', dest='outdir', type=str, default='./',
                        help=MSG_PATH_OUTPUT)
    parser.add_argument('--syntool', dest='syntool',
                        type=bool, default=False,
                        help='Set to True if format is syntool json format')
    msg = 'Set extension of file, choose between json or pyo'
    parser.add_argument('--ext', dest='ext', type=str, default='.pyo',
                        help=msg)
    parser.add_argument('--verbose', action='store_true', default=False,
                        required=False)
    parser.add_argument('--debug', action='store_true', default=False,
                        required=False)
    args = parser.parse_args()


    if args.verbose:
        logger.setLevel(logging.INFO)
    if args.debug:
        logger.setLevel(logging.DEBUG)

    reg = os.path.splitext(os.path.basename(args.region))[0]
    vel = os.path.splitext(os.path.basename(args.data_type))[0]
    par_out = {"pattern": f'frontsvel_{reg}_{vel}_{args.depth}m',
               "outdir": args.outdir}
    par_fronts = {"dir_front": args.front_dir, "pattern": args.front_pattern,
                  "gradient_threshold": args.gradient_threshold}
    if not os.path.exists(args.outdir):
         os.makedirs(args.outdir, exist_ok=True)
    compare_fronts_vel.run(par_out, par_fronts, args.data_type,
                           region=args.region, depth=args.depth,
                           syntool=args.syntool, ext=args.ext)


def run_statistics_front_box():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_json', type=str,  # nargs='+',
                        help='Json description of statistic to consider')
    parser.add_argument('--degree_box', default=2, type=float,
                        help='Size of box to compute statistics')
    parser.add_argument('--first_date', default='19900101T000000Z', type=str,
                        required=False,
                        help=MSG_FIRST_DATE)
    parser.add_argument('--last_date', default='20500101T000000Z', type=str,
                        required=False,
                        help=MSG_LAST_DATE)
    parser.add_argument('--days', default=None, type=int,
                        required=False,
                        help='Number of days considered to compute statistics')
    parser.add_argument('--out', dest='outdir',
                        type=str, default='./',
                        help=MSG_PATH_OUTPUT)
    parser.add_argument('--verbose', action='store_true', default=False,
                        required=False)
    parser.add_argument('--debug', action='store_true', default=False,
                        required=False)
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.INFO)
    if args.debug:
        logger.setLevel(logging.DEBUG)

    dic_list = box_metrics.run(args.config_json, args.degree_box,
                               first_date=args.first_date,
                               last_date=args.last_date,
                               output_dir=args.outdir,
                               number_of_days=args.days, plot=True)

def plot_statistics_front_box():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_json', type=str,  # nargs='+',
                        help='Json description of statistic to consider')
    parser.add_argument('--input', type=str,
                        help='Input netcdf file which contains statistics')
    parser.add_argument('--out', dest='outdir',
                        type=str, default='./',
                        help=f'{MSG_PATH_OUTPUT} for plot')
    parser.add_argument('--size', dest='size',
                        type=int, default=4,
                        help=f'{MSG_FILENAME_OUTPUT} for pickle file')
    parser.add_argument('--projection', type=str, default=None,
                        help='Cartopy projection name')
    parser.add_argument('--verbose', action='store_true', default=False,
                        required=False)
    parser.add_argument('--debug', action='store_true', default=False,
                        required=False)

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.INFO)
    if args.debug:
        logger.setLevel(logging.DEBUG)
    _, _, _ = box_metrics.run_plot(args.input, args.config_json, size=args.size,
                                dir_out=args.outdir, proj=args.projection)
