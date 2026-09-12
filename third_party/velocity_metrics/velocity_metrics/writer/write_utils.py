# @date 2023-09-01
# @author lucile.gaultier@oceandatalab.com


'''
Module to read and write data \n
Contains tracer and velocity classes to read  netcdf, cdf and hdf files
Classes are:

'''

import sys
import netCDF4
import numpy
import os
import velocity_metrics.writer.attributes as default_attributes
import velocity_metrics.utils.constant as const
import logging
from typing import Optional
logger = logging.getLogger(__name__)

# -------------------#
#      GENERAL       #
# -------------------#
# TO CHECK


def write_params(params: dict, pfile: str):
    """ Write parameters that have been selected to run diagnostics.
    Args:
        params: parameter object
        pfile: name of file to save parameters
    """
    with open(pfile, 'w') as f:
        for key in dir(params):
            if not key[0:2] == '__':
                f.write('{} = {}\n'.format(key, params.__dict__[key]))


def write_velocity(data: dict, outfile: str,
                   description: Optional[str] = 'AVISO-like data',
                   unit: Optional[dict] = default_attributes.unit,
                   long_name: Optional[dict] = default_attributes.long_name,
                   fill_value: Optional[float] = default_attributes.fill_value,
                   meta: Optional[dict] = {}, **kwargs):

    '''Save netcdf file '''
    lon = data['lon']
    lat = data['lat']

    # - Open Netcdf file in write mode
    fid = netCDF4.Dataset(outfile, 'w')
    fid.description = description

    # - Create dimensions
    ndim_lon = 'lon'
    ndim_lat = 'lat'
    isidf = False
    if 'lon_gcp' in data.keys():
        isidf = True
    if isidf:
        ndim_glon = 'lon_gcp'
        ndim_glat = 'lat_gcp'
    ndim_time = 'time'
    ndim_time1 = 'time1'
    dim_lon = len(numpy.shape(lon))
    fid.createDimension(ndim_lat, numpy.shape(lat)[0])
    if isidf:
        fid.createDimension(ndim_glat, numpy.shape(data['lat_gcp'])[0])
    if dim_lon == 1:
        fid.createDimension(ndim_lon, numpy.shape(data['lon'])[0])
        if isidf:
            fid.createDimension(ndim_glon, numpy.shape(data['lon_gcp'])[0])
    elif dim_lon == 2:
        fid.createDimension(ndim_lon, numpy.shape(lon)[1])
        if isidf:
            fid.createDimension(ndim_glon, numpy.shape(data['lon_gcp'])[1])
    else:
        logger.critical(f'Wrong number of dimension in longitude, is {dim_lon}'
                         'should be one or two')
        sys.exit(1)
    fid.createDimension(ndim_time, None)
    fid.createDimension(ndim_time1, 1)

    # - Create and write Variables
    if dim_lon == 1:
        vlon = fid.createVariable('lon', 'f4', (ndim_lon))
        vlat = fid.createVariable('lat', 'f4', (ndim_lat))
        vlon[:] = lon
        vlat[:] = lat
        if isidf:
            vglon = fid.createVariable('lon_gcp', 'f4', (ndim_glon))
            vglat = fid.createVariable('lat_gcp', 'f4', (ndim_glat))
            vilon = fid.createVariable('index_lon_gcp', 'f4', (ndim_glon))
            vilat = fid.createVariable('index_lat_gcp', 'f4', (ndim_glat))
            vglon[:] = data['lon_gcp']
            vglat[:] = data['lat_gcp']
            vilon[:] = data['index_lon_gcp']
            vilat[:] = data['index_lat_gcp']
    elif dim_lon == 2:
        vlon = fid.createVariable('lon', 'f4', (ndim_lat, ndim_lon))
        vlat = fid.createVariable('lat', 'f4', (ndim_lat, ndim_lon))
        vlon[:, :] = lon
        vlat[:, :] = lat
        if 'gcp' in data.keys():
            vglon = fid.createVariable('lon_gcp', 'f4', (ndim_glat, ndim_glon))
            vglat = fid.createVariable('lat_gcp', 'f4', (ndim_glat, ndim_glon))
            vilon = fid.createVariable('index_lon_gcp', 'f4', (ndim_glat,
                                                               ndim_glon))
            vilat = fid.createVariable('index_lat_gcp', 'f4', (ndim_glat,
                                                               ndim_glon))
            vglon[:, :] = data['lon_gcp']
            vglat[:, :] = data['lat_gcp']
            vilon[:, :] = data['index_lon_gcp']
            vilat[:, :] = data['index_lat_gcp']
    vlon.units = unit['lon']
    vlat.units = unit['lat']
    vlon.long_name = long_name['lon']
    vlat.long_name = long_name['lat']
    if isidf:
        vglon.units = unit['lon']
        vglat.units = unit['lat']
        vglon.long_name = f'ground control point {long_name["lon"]}'
        vglat.long_name = f'ground control point {long_name["lat"]}'
        _text = 'index of ground control point in'
        vilon.long_name = f'{_text} {long_name["lon"]} dimension'
        vilat.long_name = f'{_text} {long_name["lat"]} dimension'
    for key, value in kwargs.items():
        if key == 'time':
            value *= 86400
        if value.any():
            dim_value = len(value.shape)
            if isidf:
                bin_key = f'{key}_bin'
                _value = + value
                bvalue, scale, offset = pack_as_ubytes(_value, fill_value)
            if dim_value == 1:
                if key == 'time':
                    var = fid.createVariable(str(key), 'f8', (ndim_time, ),
                                             fill_value=fill_value)
                else:
                    var = fid.createVariable(str(key), 'f4', (ndim_time, ),
                                             fill_value=fill_value)
                var[:] = + value
                if isidf:
                    bvar = fid.createVariable(bin_key, 'u1', (ndim_time, ),
                                              fill_value=numpy.ubyte(255))
                    bvar[:] = bvalue

            if dim_value == 2:
                var = fid.createVariable(str(key), 'f4', (ndim_time1, ndim_lat,
                                         ndim_lon), fill_value=fill_value)
                var[0, :, :] = value
                if isidf:
                    bvar = fid.createVariable(bin_key, 'u1', (ndim_time1,
                                              ndim_lat, ndim_lon),
                                              fill_value=numpy.ubyte(255))
                    bvar[0, :, :] = bvalue
            elif dim_value == 3:
                var = fid.createVariable(str(key), 'f4', (ndim_time, ndim_lat,
                                         ndim_lon), fill_value=fill_value)
                var[:, :, :] = value
                if isidf:
                    bvar = fid.createVariable(bin_key, 'u1', (ndim_time,
                                              ndim_lat, ndim_lon),
                                              fill_value=numpy.ubyte(255))
                    bvar[:, :, :] = bvalue
            if str(key) in unit.keys():
                var.units = unit[str(key)]
                if isidf:
                    bvar.units = unit[str(key)]
            if str(key) in long_name.keys():
                var.long_name = long_name[str(key)]
                if isidf:
                    bvar.long_name = long_name[str(key)]
            if isidf:
                bvar.scale_factor = scale
                bvar.add_offset = offset
                bvar.valid_min = 0
                bvar.valid_max = 254

    for key, value in meta.items():
        setattr(fid, key, value)
    fid.close()
    return None


def write_tracer_1d(data: dict, namevar: str, outfile: str,
                    unit: Optional[dict] = default_attributes.unit,
                    long_name: Optional[dict] = default_attributes.long_name,
                    fill_value: Optional[float] = default_attributes.fill_value,
                    global_attribute: Optional[dict] = {},
                    ):
    '''Write new tracer in a cdf file. '''
    lon = data['lon']
    lat = data['lat']
    var = data['tracer']
    time = data['time']
    # - Open Netcdf file in write mode
    fid = netCDF4.Dataset(outfile, 'w')
    for name, value in global_attribute.keys():
        setattr(fid, name, value)
    if 'description' not in global_attribute.items():
        fid.description = const.glob_attribute['description']

    # - Create dimensions
    dim_part = 'obs'
    dim_time = 'time'
    fid.createDimension(dim_part, numpy.shape(lon)[1])
    fid.createDimension(dim_time, None)
    dim_time = 'time'

    # - Create and write Variables
    vtime = fid.createVariable('time', 'f4', (dim_time))
    vlon = fid.createVariable('lon', 'f4', (dim_time, dim_part))
    vlat = fid.createVariable('lat', 'f4', (dim_time, dim_part))
    vtra = fid.createVariable(namevar, 'f4', (dim_part), fill_value=fill_value)
    vtime[:] = time
    vtime.units = "days"
    vlon[:, :] = lon
    vlon.units = unit['lon']
    vlon.long_name = long_name['lon']
    vlat[:, :] = lat
    vlat.units = unit['lat']
    vlat.long_name = long_name['lat']
    vtra[:] = var
    vtra.units = unit['T']
    vtra.long_name = long_name['T']
    fid.close()
    return None


def write_listracer_1d(wfile: str, T: dict, par: dict, paro: dict, listTr: list,
                       description: Optional[str] = "Drifter advected by lagrangian tool",
                       global_attribute: Optional[dict] = {},
                       fill_value: Optional[float] = default_attributes.fill_value):
    '''Write list of tracer in a cdf file. '''
    if "fill_value" in paro.keys():
        fill_value = paro["fill_value"]
    # - Open Netcdf file in write mode
    if os.path.exists(wfile):
        os.remove(wfile)
    path, _ = os.path.split(wfile)
    os.makedirs(path, exist_ok=True)
    fid = netCDF4.Dataset(wfile, 'w')
    for name, value in global_attribute.items():
        if value is None:
            value = ''
        setattr(fid, name, value)
    if 'description' not in global_attribute.keys():
        fid.description = description
    # exportables = [k for k in dir(p) if not k.startswith('__')]

    str_params = [f'{k} = {v}' for k, v in par.items()]
    if 'comment' not in global_attribute.keys():
        fid.comment = ', '.join(str_params)
    strtime = '%Y-%m-%dT%H:%M:%SZ'
    fid.time_coverage_start = par["first_date"].strftime(strtime)
    fid.time_coverage_end = par["last_date"].strftime(strtime)

    # - Create dimensions
    dim_part = 'obs'
    dim_time = 'time'
    dim_time_hr = 'time_hr'
    fid.createDimension(dim_part, numpy.shape(T['lon_lr'])[1])
    fid.createDimension(dim_time, None)
    if paro["save_traj"] is True:
        fid.createDimension(dim_time_hr, numpy.shape(T['lon_hr'])[0])

        # - Create and write Variables
        vlon = fid.createVariable('lon_hr', 'f4', (dim_time_hr, dim_part))
        vlon[:, :] = T['lon_hr']
        vlon.units = "deg E"
        vlon.long_name = 'High temporal resolution longitude'
        vlat = fid.createVariable('lat_hr', 'f4', (dim_time_hr, dim_part))
        vlat[:, :] = T['lat_hr']
        vlat.units = "deg N"
        vlat.long_name = 'High temporal resolution latitude'
        vmask = fid.createVariable('mask_hr', 'f4', (dim_time_hr, dim_part))
        vmask[:, :] = T['mask_hr']
        vtime = fid.createVariable('time_hr', 'f4', (dim_time_hr))

        vtime[:] = T['time_hr']
        vtime.units = "days since start of advection"
        vtime.long_name = ('High temporal resolution time from the reference'
                           'time')
        vu = fid.createVariable('zonal_velocity', 'f4', (dim_time_hr,
                                dim_part), fill_value=fill_value)
        vu[:, :] = T['u_hr']
        vu.units = 'm/s'
        vu.long_name = 'High temporal resolution zonal velocity'
        vv = fid.createVariable('meridional_velocity', 'f4',
                                (dim_time_hr, dim_part),
                                fill_value=fill_value)
        vv[:, :] = T['v_hr']
        vv.units = 'm/s'
        vv.long_name = 'High temporal resolution meridional velcity'
        if paro["compute_strain"] is True:
            vS = fid.createVariable('Strain', 'f4', (dim_time_hr, dim_part),
                                    fill_value=fill_value)
            vS[:, :] = T['S_hr']
            vS.units = 's-1'
            vS.long_name = 'Strain'
        if paro["compute_rv"] is True:
            vRV = fid.createVariable('Vorticity', 'f4', (dim_time_hr,
                                     dim_part), fill_value=fill_value)
            vRV[:, :] = T['RV_hr']
            vRV.units = 's-1'
            vRV.long_name = 'Relative Vorticity'
        if paro["compute_ow"] is True:
            vOW = fid.createVariable('OW', 'f4', (dim_time_hr, dim_part),
                                     fill_value=fill_value)
            vOW[:, :] = T['OW_hr']
            vOW.units = 's-1'
            vOW.long_name = 'Okubo-Weiss'

    vlon = fid.createVariable('lon', 'f4', (dim_time, dim_part),
                              fill_value=fill_value)
    vlon.units = "deg E"
    vlon.long_name = 'longitude'
    vlat = fid.createVariable('lat', 'f4', (dim_time, dim_part),
                              fill_value=fill_value)
    vlat.units = "deg N"
    vlat.long_name = 'latitude'
    vtime = fid.createVariable('time', 'f4', (dim_time), fill_value=fill_value)
    vlon[:, :] = T['lon_lr']
    vlat[:, :] = T['lat_lr']
    vtime[:] = T['time_lr'][:]
    vtime.units = "days since start of advection"
    vmask = fid.createVariable('mask_lr', 'f4', (dim_time, dim_part))
    vmask[:, :] = T['mask_lr']
    list_tracer = None
    if list_tracer is not None:
        for i in range(len(listTr)):
            vtra = fid.createVariable(par["listtracer"][i], 'f4', (dim_time,
                                      dim_part), fill_value=fill_value)
            vtra[:, :] = listTr[i].newvar
    fid.close()
    return None


# TO CHECK
def write_aviso(wfile, VEL, t, fill_value=-1.e36):

    '''Write AVISO data, Strain, Vorticity and Okubo Weiss in a cdf file. '''
    # - Open Netcdf file in write mode
    fid = netCDF4.Dataset(wfile, 'w')
    fid.description = ('Strain, Vorticity and OW parameters from AVISO'
                       ' velocity fields')

    # - Create dimensions
    dim_time = 'time'
    dim_lon = 'lon'
    dim_lat = 'lat'
    fid.createDimension(dim_lon, numpy.shape(VEL.lon)[0])
    fid.createDimension(dim_lat, numpy.shape(VEL.lat)[0])
    fid.createDimension(dim_time, None)

    # - Create and write Variables
    vlon = fid.createVariable('lon', 'f4', (dim_lon))
    vlat = fid.createVariable('lat', 'f4', (dim_lat))
    vtime = fid.createVariable('time', 'f4', (dim_time))
    vlon[:] = VEL.lon
    vlon.units = "deg E"
    vlon.long_name = "Longitudes"
    vlat[:] = VEL.lat
    vlat.units = "deg N"
    vlat.long_name = "Latitudes"
    vtime[:] = VEL.time[t]
    vtime.units = "Julian days (CNES)"
    vH = fid.createVariable('H', 'f4', (dim_time, dim_lat, dim_lon),
                            fill_value=fill_value)
    if t is None:
        vH[0, :, :] = VEL.h[:, :]
    else:
        vH[0, :, :] = VEL.h[t, :, :]
    vH.units = 'm'
    vH.long_name = 'Sea Surface Height'
    vu = fid.createVariable('U', 'f4', (dim_time, dim_lat, dim_lon),
                            fill_value=fill_value)
    if t is None:
        vu[0, :, :] = VEL.us[:, :]
    else:
        vu[0, :, :] = VEL.us[t, :, :]
    vu.units = 'm/s'
    vu.long_name = 'zonal velocity'
    vv = fid.createVariable('V', 'f4', (dim_time, dim_lat, dim_lon),
                            fill_value=fill_value)
    if t is None:
        vv[0, :, :] = VEL.vs[:, :]
    else:
        vv[0, :, :] = VEL.vs[t, :, :]
    vv.units = 'm/s'
    vv.long_name = 'meridional velocity'
    if hasattr(VEL, 'Sn'):
        vSn = fid.createVariable('Sn', 'f4', (dim_time, dim_lat, dim_lon),
                                 fill_value=fill_value)
        vSn[0, :, :] = VEL.Sn[t, :, :]
        vSn.units = "s-1"
        vSn.long_name = "Normal Strain"
    if hasattr(VEL, 'Ss'):
        vSs = fid.createVariable('Ss', 'f4', (dim_time, dim_lat, dim_lon),
                                 fill_value=fill_value)
        vSs[0, :, :] = VEL.Ss[t, :, :]
        vSs.units = "s-1"
        vSs.long_name = "Shear Strain"
    if hasattr(VEL, 'RV'):
        vRV = fid.createVariable('Vorticity', 'f4', (dim_time, dim_lat,
                                 dim_lon), fill_value=fill_value)
        vRV[0, :, :] = VEL.RV[t, :, :]
        vRV.units = "s-1"
        vRV.long_name = 'Relative Vorticity'
    if hasattr(VEL, 'OW'):
        vOW = fid.createVariable('OW', 'f4', (dim_time, dim_lat, dim_lon),
                                 fill_value=fill_value)
        vOW[0, :, :] = VEL.OW[t, :, :]
        vOW.units = "s-1"
        vOW.long_name = 'Okubo-Weiss'
    fid.close()
    return None


def pack_as_ubytes(var, fill_value):
    vmin = numpy.nanmin(var)
    vmax = numpy.nanmax(var)
    vmin = max(vmin, -100)
    vmax = min(vmax, 400)
    nan_mask = ((numpy.ma.getmaskarray(var)) | (numpy.isnan(var))
                | (var == fill_value))

    offset, scale = vmin, (float(vmax) - float(vmin)) / 254.0
    if vmin == vmax:
        scale = 1.0

    numpy.clip(var, vmin, vmax, out=var)

    # Required to avoid runtime warnings on masked arrays wherein the
    # division of the _FillValue by the scale cannot be stored by the dtype
    # of the array
    if isinstance(var, numpy.ma.MaskedArray):
        mask = numpy.ma.getmaskarray(var).copy()
        var[numpy.where(mask)] = vmin
        _var = (numpy.ma.getdata(var) - offset) / scale
        var.mask = mask  # Restore mask to avoid side-effects
    else:
        _var = (var - offset) / scale

    result = numpy.round(_var).astype('ubyte')
    result[numpy.where(nan_mask)] = 255
    return result, scale, offset
