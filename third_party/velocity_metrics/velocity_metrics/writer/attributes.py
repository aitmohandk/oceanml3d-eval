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
Define constant and units
"""

fill_value = -1.36e9

# - UNITS AND NAME
unit = {"U": "m/s", "V": "m/s", "T": "degC", "lambda": "/days", "lon": "deg E",
        "lat": "deg N", "time": "seconds since 1970-01-01T00:00:00.000000Z",
        "FSLE": "1/day", "FTLE": "1/day",
        "lat_hr": "deg N", "lon_hr": "deg E", "lat_lr": "deg N",
        "lon_lr": "deg E", "time_hr": "day", }

long_name = {"U": "zonal velocity",
             "V": "meridional velocity",
             "T": "Sea Surface temperature",
             "FSLE": "Finite-Size Lyapunov Exponent",
             "FTLE": "Finite-Time Lyapunov Exponent",
             "lon": "longitude", "lat": "latitude", "time": "time",
             "lon_hr": "High temporal resolution longitude",
             "lat_hr": "'High temporal resolution latitude",
             "time_hr": "High temporal resolution time",
             }
