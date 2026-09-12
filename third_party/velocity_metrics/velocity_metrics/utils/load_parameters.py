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

import logging
import re
import json
import os
import sys
from configparser import ConfigParser
from typing import Optional
logger = logging.getLogger()


def load_file(filename: str, section: Optional[str] = None) -> dict:
    """ Load Parameter file in ini or json format
    Args:
        filename (str): File name, json or ini file
        section (Optional, str): section in ini file
    Returns:
        Dictionary with parameters from file
    """
    if not os.path.exists(filename):
        logger.error(f'File {filename} not found')
        sys.exit(1)
    _, ext = os.path.splitext(filename)
    if ext == '.json':
        with open(filename, 'r') as f:
            dic = json.load(f)
    elif section is not None:
        parser = ConfigParser()
        parser.read(filename)
        dic = parser._sections[section]
    else:
        logger.critical('Please provide json file or valid section parameter')
        sys.exit(1)
    return dic


def str2float(dico: dict, list_key: list) -> dict:
    for key in list_key:
        if key in dico.keys():
            dico[key] = float(dico[key])
    return dico


def str2bool(dico: dict) -> dict:
    for key, value in dico.items():
        if value == 'True':
            dico[key] = True
        elif value == 'False':
            dico[key] = False
    return dico


def sel_region(region_file: str):
    if region_file is None:
        logger.warning('Global region set up')
        dic = {'lllon': -180, 'urlon': 180, 'lllat': -90, 'urlat': 90,
               'name': 'global'}
    elif os.path.exists(region_file):
        dic = load_file(region_file, section='region')
    else:
        logger.warning(f'file region file {region_file} not found')
        logger.warning('Global region set up')
        dic = {'lllon': -180, 'urlon': 180, 'lllat': -90, 'urlat': 90,
               'name': 'global'}
    if "name" not in dic.keys():
        dic["name"] = ''
    if "coords" not in dic.keys():
        dic["coords"] = ((dic["lllon"], dic["lllat"]),
                         (dic["lllon"], dic["urlat"]),
                         (dic["urlon"], dic["urlat"]),
                         (dic["urlon"], dic["lllat"]))

    return (dic["lllon"], dic["urlon"], dic["lllat"], dic["urlat"],
            dic["coords"], dic["name"])


def sel_data(data_type_file: str, ):
    if os.path.exists(data_type_file):
        dic = load_file(data_type_file, section='data')
    else:
        logger.critical(f'please provide data_type file {data_type_file}')
        sys.exit(1)
    match = re.compile(dic["match"]).search
    if "depth" not in dic.keys():
        dic["depth"] = [None, None]
    if "time_coverage_hours" not in dic.keys():
        dic["time_coverage_hours"] = 24
    if "ntime" not in dic.keys():
        dic["ntime"] = "time"
    if "ndepth" not in dic.keys():
        dic["ndepth"] = "depth"
    dic_name = {'u': dic["varu"], 'v': dic["varv"], 'lon': dic["nlon"],
                'lat': dic["nlat"], 'time': dic["ntime"], 'depth': dic["ndepth"]}
    return (dic["path"], dic["pattern"], match, dic_name, dic["depth"],
            dic["data_type"], dic["label"], dic["time_coverage_hours"])


def load_advection_parameters(parameter_file: str):
    if os.path.exists(parameter_file):
        dic_adv = load_file(parameter_file, section='advection')
        dic_grid = load_file(parameter_file, section='grid')
        dic_output = load_file(parameter_file, section='output')
    else:
        msg = f'please provide advection parameter file {parameter_file}'
        logger.critical(msg)
        sys.exit(1)
    dic_grid = str2float(dic_grid, ["dx", "dy", "size"])
    dic_grid = str2bool(dic_grid)
    list_key = ["adv_time_step", "tadvection", "scale", "sigma", "ka",
                "weight_part", "radius_part", "gamma"]
    dic_adv = str2float(dic_adv, list_key)
    dic_adv = str2bool(dic_adv)
    dic_output = str2float(dic_output, ["fill_value", "output_step"])
    dic_output = str2bool(dic_output)
    dic_adv["output_step"] = dic_output["output_step"]
    return dic_adv, dic_grid, dic_output


def load_fronts_stat_parameters(parameter_file: str):
    if os.path.exists(parameter_file):
        dic = load_file(parameter_file)
    else:
        logger.critical(f'please provide statistics file {parameter_file}')
        sys.exit(1)
    list_var = ["lon", "lat", "vel", "gradient_sst", "flux_across",
                "vectorial_product"]
    dic["global"].setdefault("list_var", list_var)
    dic["global"].setdefault("nvar", "flux_across")
    dic["global"].setdefault("flag_threshold", 3)
    dic["global"].setdefault("velocity_threshold", 0.0)
    dic["global"].setdefault("gradient_threshold", 0.15)
    dic["global"].setdefault("percentage_threshold", 0.35)
    return dic
