import numpy
import logging
from typing import Optional
logger = logging.getLogger(__name__)


# - Initialization
def init_mpi(isMPI: bool):
    ''' Initialize mpi variables '''
    if isMPI is True:
        try:
            from mpi4py import MPI
        except ImportError:
            logger.info("Module MPI not installed, no parallelisation")
            isMPI = False
    if isMPI is True:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        print(f"Parallelisation with {size} cpu ...")

        if size == 1:
            isMPI = False

        # name = MPI.Get_processor_name()
    else:
        comm = None
        size = 1
        rank = 0


    return isMPI, size, rank, comm


def init_empty_variables(par_adv: dict, paro: dict, grid, listTr: list,
                         size: int, rank: int, list_tra: Optional[list] = None
                         ):
    # - Initialize new tracer variables
    if rank == 0:
        logger.info('Advecting variables')
    Gridsize = numpy.shape(grid.lon1d)[0]
    # Tsize=numpy.shape(Tr.lon)
    npixel = int((Gridsize - 1)/size) + 1
    i0 = npixel * rank
    i1 = min(npixel*(rank + 1), Gridsize)
    # List of particles
    # particules = numpy.arange(0, Gridsize)
    # Sublist of particles when parallelised
    reducepart = numpy.arange(i0, i1)
    # Number of Timesteps to store
    tadvection = (par_adv["last_date"] - par_adv["first_date"]).total_seconds()
    tadvection = tadvection / 86400.
    Timesize_lr = int(abs(tadvection + 1) / par_adv["output_step"] + 1)
    # initialization + advection for t in [0, tadvection] with adv_time_step
    # time step
    Timesize_hr = int(abs(tadvection + 1) / par_adv["output_step"]
                      / abs(par_adv["adv_time_step"]) + 1)
    dim_hr = [Timesize_hr, Gridsize]
    dim_lr = [Timesize_lr, Gridsize]
    # Initialise number of tracer in list
    nlist = 0
    if rank == 0:
        # Low resolution variables
        grid.lon_lr = numpy.empty(dim_lr)
        grid.lat_lr = numpy.empty(dim_lr)
        grid.mask_lr = numpy.empty(dim_lr)
        grid.iit_lr = numpy.empty((2, dim_lr[0], dim_lr[1]))
        grid.ijt_lr = numpy.empty((2, dim_lr[0], dim_lr[1]))
        # High resolution variables
        grid.lon_hr = numpy.empty(dim_hr)
        grid.lat_hr = numpy.empty(dim_hr)
        grid.mask_hr = numpy.empty(dim_hr)
        grid.vel_v_hr = numpy.empty(dim_hr)
        grid.vel_u_hr = numpy.empty(dim_hr)
        grid.hei_hr = numpy.empty(dim_hr)
        if paro["compute_strain"] is True:
            grid.S_hr = numpy.empty(dim_hr)
        if paro["compute_rv"] is True:
            grid.RV_hr = numpy.empty(dim_hr)
        if paro["compute_ow"] is True:
            grid.OW_hr = numpy.empty(dim_hr)
        if list_tra is not None:
            nlist = len(listTr)
            for i in range(nlist):
                Tr = listTr[i]
                Tr.newvar = numpy.empty(dim_lr)
                Tr.newi = numpy.empty(dim_lr)
                Tr.newj = numpy.empty(dim_lr)
    return dim_hr, dim_lr, Gridsize, reducepart, i0, i1


def gather_data_mpi(par: dict, list_var_adv: list, listGr: list, listTr: list,
                    dim_lr: list, dim_hr: list,
                    comm, rank: int, size: int, grid_size,
                    list_tra: Optional = None, list_num: Optional[list] = None
                    ) -> dict:

    # Define local empty variables with the correct size
    if list_tra is not None:
        nlist = len(listTr)
        for i in range(nlist):
            Tr = listTr[i]
            Tr.newvarlocal = numpy.empty(dim_lr)
            Tr.newilocal = numpy.empty(dim_lr)
            Tr.newjlocal = numpy.empty(dim_lr)

    # - Gather data in processor 0 and save them
    if listTr is not None:
        reordering1dmpi(par, listTr, listGr, list_num)
    comm.barrier()
    local = {}
    for key, value in list_var_adv.items():
        local[key] = comm.gather(value, root=0)
    if list_tra is not None:
        nlist = len(listTr)
        for i in range(nlist):
            Tr = listTr[i]
            Tr.newvarlocal = comm.gather(Tr.newvarloc,  root=0)
            Tr.newilocal = comm.gather(Tr.newiloc,  root=0)
            Tr.newjlocal = comm.gather(Tr.newjloc, root=0)
    if rank == 0:
        data = {}
        if 'time_hr' in list_var_adv.keys():
            data['time_hr'] = list_var_adv['time_hr']
            del list_var_adv['time_hr']
        tadvection = (par["last_date"] - par["first_date"]).total_seconds()
        tadvection = tadvection / 86400
        tstep = tadvection / par["output_step"] / abs(tadvection)
        first_day = (par["first_date"] - par["reference"]).total_seconds()
        first_day = first_day / 86400
        tstop = first_day + tadvection + tstep
        data['time'] = numpy.arange(first_day, tstop, tstep)
        for irank in range(0, size):

            npixel = int((grid_size - 1)/size) + 1
            i0 = npixel * irank
            i1 = min(npixel*(irank + 1), grid_size)
            for key, value in list_var_adv.items():
                if irank == 0:
                    if 'lr' in key:
                        dim = dim_lr
                    else:
                        dim = dim_hr
                    data[key] = numpy.empty(dim)
                ndim = len(dim)
                if ndim == 1:
                    data[key][i0:i1] = local[key][irank][:]
                elif ndim == 2:
                    print(numpy.shape(local[key][irank]))
                    _shape_loc = numpy.shape(local[key][irank])
                    if _shape_loc[1] == numpy.shape(data[key][:, i0:i1])[1]:
                        data[key][:, i0:i1] = local[key][irank][:, :]
                    else:
                        logger.info(f'{key} has dimension {_shape_loc}')
                else:
                    logger.error(f'Wrong dimension for variable {key}: {ndim}')
            if (listTr is not None) and (len(listTr) > 0):
                nlist = len(listTr)
                for i in range(nlist):
                    Tr = listTr[i]
                    Tr.newvar[:, i0:i1] = Tr.newvarlocal[irank][:, :]
                    Tr.newi[:, i0:i1] = Tr.newilocal[irank][:, :]
                    Tr.newj[:, i0:i1] = Tr.newjlocal[irank][:, :]
        return data


def gather_data(par: dict, list_var_adv: list, listGr: list, listTr: list,
                list_num: Optional[list] = None) -> dict:
    logger.info('No parallelisation')
    if listTr is not None:
        reordering1d(par, listTr, listGr, list_num=list_num)
    data = {}
    for key, value in list_var_adv.items():
        data[key] = list_var_adv[key]
    tadvection = (par["last_date"] - par["first_date"]).total_seconds() / 86400
    tstep = tadvection / par["output_step"] / abs(tadvection)
    first_day = (par["first_date"] - par["reference"]).total_seconds() / 86400
    tstop = first_day + tadvection + tstep
    data['time'] = numpy.arange(first_day, tstop, tstep)
    return data


def make_list_particles(grid) -> None:
    grid.lon1d = grid.lon.ravel()
    grid.lat1d = grid.lat.ravel()
    grid.mask1d = grid.mask.ravel()
    grid.lon1d = grid.lon1d[~numpy.isnan(grid.mask1d)]
    grid.lat1d = grid.lat1d[~numpy.isnan(grid.mask1d)]
    grid.mask1d = grid.mask1d[~numpy.isnan(grid.mask1d)]


def reordering1d(par: dict, listTr: list, listGr: list,
                 list_num: Optional[list] = None):
    first_day = (par["first_date"] - par["reference"]).total_seconds() / 86400
    tadvection = (par["last_date"] - par["first_date"]).total_seconds() / 86400
    if listGr and (list_num is not None):
        n2, nt, npa = numpy.shape(listGr[0].newi)
        nlist = len(listTr)
        tra = numpy.zeros((nt, npa, nlist))
        for i in range(nlist):
            Tr = listTr[i]
            Gr = listGr[list_num[i]]
            for time in range(nt):
                realtime = (first_day + time * numpy.sign(tadvection))
                tratime = numpy.argmin(abs(Tr.time - realtime))
                for pa in range(npa):
                    tra[time, pa, i] = Tr.var[tratime,
                                              int(Gr.newi[0, time, pa]),
                                              int(Gr.newj[0, time, pa])]
            Tr.newvar = tra[:, :, i]
            Tr.newi = Gr.newi[0, :, :]
            Tr.newj = Gr.newj[0, :, :]
        return tra
    else:
        return None


def reordering1dmpi(par: dict, listTr: list, listGr: list,
                    list_num: Optional[list] = None):
    tra = None
    first_day = (par["first_date"] - par["reference"]).total_seconds() / 86400
    tadvection = (par["last_date"] - par["first_date"]).total_seconds() / 86400
    if listGr and (list_num is not None):
        n2, nt, npa = numpy.shape(listGr[0].newi)
        nlist = len(listTr)
        tra = numpy.zeros((nt, npa, nlist))
        for i in range(nlist):
            Tr = listTr[i]
            Gr = listGr[list_num[i]]
            for time in range(nt):
                realtime = (first_day + time * numpy.sign(tadvection))
                tratime = numpy.argmin(abs(Tr.time - realtime))
                for pa in range(npa):
                    tra[time, pa, i] = Tr.var[tratime,
                                              int(Gr.newi[0, time, pa]),
                                              int(Gr.newj[0, time, pa])]
            Tr.newvarloc = tra[:, :, i]
            Tr.newiloc = Gr.newi[0, :, :]
            Tr.newjloc = Gr.newj[0, :, :]
    return tra
