import os
import datetime
import numpy
import lap.utils.tools as tools
import lap.utils.write_utils as write_utils
import lap.mod_io as mod_io
import logging
logger = logging.getLogger(__name__)


def compute_eulerian_diags(p) -> None:
    logger.info(f'Start time {datetime.datetime.now()}')
    tools.make_default(p)

    # - Read velocity
    logger.info('Loading Velocity')
    VEL = mod_io.read_velocity(p, get_time=1)
    mask_h = numpy.isnan(VEL.h)
    mask_us = numpy.isnan(VEL.us)
    mask_vs = numpy.isnan(VEL.vs)
    mask = (mask_h | mask_us | mask_vs)
    VEL.us[mask] = numpy.nan
    VEL.vs[mask] = numpy.nan
    VEL.Ss[mask] = numpy.nan
    VEL.Sn[mask] = numpy.nan
    VEL.RV[mask] = numpy.nan
    VEL.h[mask] = numpy.nan

    # - Compute Okubo-Weiss:
    VEL.OW = VEL.Ss**2 + VEL.Sn**2 - VEL.RV**2
    VEL.OW[mask] = numpy.nan

    # - Write all data:
    logger.info('Writing Velocities')
    ntime = numpy.shape(VEL.us)[0]
    for t in numpy.arange(0, ntime, 1):
        index_t = int(t)
        jj = int(VEL.time[index_t])
        file_default = f'eulerian_{jj:06d}.nc'
        default_output = os.path.join(p.output_dir, file_default)
        p.output = default_output
        # getattr(p, 'output', default_output)
        write_utils.write_aviso(p.output, VEL, index_t)
    logger.info(f'End time {datetime.datetime.now()}')
