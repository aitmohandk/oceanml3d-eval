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
Define geophysical, conversion constant and units
"""

import numpy

deg2km = 111000.
sec2day = 1/86400.
day2sec = 86400.
FMT = '%Y%m%dT%H%M%SZ'


# - GEOPHYSICAL CONSTANTS
Rearth = 6371000.0  # m
factor = 180. / (numpy.pi * Rearth)  # m-1
visc = 1.83e-6
omega = 7.2921*10**(-4)  # rad/s
f0 = 1 / omega
g = 9.81
