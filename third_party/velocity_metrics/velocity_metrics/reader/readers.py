# vim: ts=4:sts=4:sw=4

# @date 2022-10-06
# @author lucile.gaultier@oceandatalab.com

# Copyright (C) 2022-2023 OceanDataLab
# This file is part of fronts_detection

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
Read data using reader plugins
"""


import numpy
import pickle
import json
import os
import sys
import datetime
from .modis import reader as modis_reader
from .fsle import reader as fsle_reader
from .regular_grid import reader as regular_grid_reader
from .seascope_idf import reader as seascope_idf_reader
import logging
logger = logging.getLogger(__name__)


def read_data(nfile: str, global_params: dict) -> dict:
    """
    Input :
        - nfile (str) : name of the file
        - global_params (dict) : parameters (directory, box, data_type)

    Output :
        - dico_data (dict) : dictionary containing obtained thanks to the
        the reader
        - params1, params23_fusion_proba, params4_fusion_proba (dict) :
        parameters for each part of the algorithm:w

    """

    file_path = nfile
    bfile = os.path.basename(nfile)
    box = global_params['box']
    data_type = global_params['data_type']
    variable = None
    per = 70
    if 'variable' in global_params.keys():
        variable = global_params['variable']
    else:
        logger.error('missing variable key in parameter file')
        sys.exit(1)
    if data_type == 'REGULAR_OBS':
        dico_data = regular_grid_reader(file_path, box, variable)
    if data_type == 'MODIS':
        dico_data = modis_reader(file_path, bfile, box)
    if data_type == 'IDF':
        dico_data = seascope_idf_reader(file_path, box, variable)
    if data_type == 'FSLE':
        dico_data = fsle_reader(file_path, box, per)
    return dico_data


def pickle2json(pickle_path: str,
                json_path: str,
                time_init: datetime,
                minutes_delta: int):
    time_init = datetime.datetime(1970, 1, 1)
    pickle_in = open(pickle_path, "rb")
    dico_fronts = pickle.load(pickle_in)
    pickle_in.close()

    json_list_fronts = []
    for i in range(len(dico_fronts['start'])):
        _slice = slice(dico_fronts['start'][i],
                       (dico_fronts['start'][i] + dico_fronts['length'][i]))
        lons = dico_fronts['lon'][_slice]
        lats = dico_fronts['lat'][_slice]

        points = numpy.swapaxes([lons, lats], 0, 1)
        points = points.tolist()
        start = (dico_fronts['time'] - time_init).total_seconds()
        # begin = dico_fronts['time_seconds']
        end = start + datetime.timedelta(minutes=minutes_delta).total_seconds()
        type = "LINE"
        json_list_fronts.append({"end": int(end)*1000,
                                 "points": points,
                                 "start": int(start)*1000,
                                 "type": type})
    with open(json_path, 'w') as fp:
        json.dump(json_list_fronts, fp)
