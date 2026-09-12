 # vim: ts=4:sts=4:sw=4
 #
 # @author lucile.gaultier@oceandatalab.com
 # @date 2023-08-11
 #
 # Copyright (C) 2023-2023 OceanDataLab
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

"""
#-----------------------------------------------------------------------
#                       Additional Documentation
# Authors: Lucile Gaultier
#
# Modification History:
# - Aug 2023:  Original by Lucile Gaultier, ODL
# Notes:
# - Written for Python 3.10 and above
#
#
#
#-----------------------------------------------------------------------
"""
# -----------------------------------------------------------------------

# ---------------- Module General Import and Declarations ---------------
import os
import sys

# - Set module version to package version:


__version__ = '0'
__author__ = 'Lucile Gaultier'
__date__ = '2023-08-11'
__email__ = 'lucile.gaultier@oceandatalab.com'
__url__ = ''
__description__ = ('Metrics to validate or intercompare velocity in the ocean')
__author_email__ = ('lucile.gaultier@oceandatalab.com')
__keywords__ = ()

# - If you're importing this module in testing mode, or you're running
#  pydoc on this module via the command line, import user-specific
#  settings to make sure any non-standard libraries are found:

if (__name__ == "__main__") or \
   ("pydoc" in os.path.basename(sys.argv[0])):
    import user


# - Find python version number
__python_version__ = sys.version[:3]

# - Import numerical array formats
try:
    import numpy
except ImportError:
    print(''' Numpy is not available on this platform,
          ''')

# - Import scientific librairies
try:
    import scipy
except ImportError:
    print("""Scipy is not available on this platform,
          """)


# - Import netcdf reading librairies
try:
    import netCDF4
except ImportError:
    print(''' netCDF4 is not available on this machine,
          ''')
    # reading and writing netcdf functions in rw_data.py won't work'''
