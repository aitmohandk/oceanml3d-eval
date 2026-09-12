import numpy
import logging
logger = logging.getLogger(__name__)

idf_fmt = '%Y-%m-%dT%H:%M:%S.%fZ'
global_idf = {'cdm_data_type': 'grid',
              'idf_version': '1.0',
              'idf_subsampling_factor': 0,
              'institution': 'OceanDataLab',
              'product_version': '1.0',
              'Metadata_Conventions': 'Unidata Dataset Discovery V1.0',
              'standard_name_vocabulary': 'NetCDF Climate and Forecast (CF) '
                                          'Metadata Convention',
              'creator_name': 'OceanDataLab',
              'creator_url': 'www.oceandatalab.com',
              'publisher_email': 'contact@oceandatalab.com',
              'file_version': '1.0',
              }


def compute_gcp(lon, lat, gcp_lat_spacing=32, gcp_lon_spacing=32):

    # Compute geotransform parameters
    # A workaround is used here to avoid numerical precision issues in
    # numpy.mean: if values are too close to 0, underflow errors may arise so
    # we multiply values by a large factor before passing them to numpy.mean,
    # then we divide the result by the same factor
    precision_factor = 10000
    lon0 = lon[0]
    _dlon = lon[1:] - lon[:-1]
    dlon = numpy.mean(precision_factor * _dlon) / precision_factor
    lat0 = lat[0]
    _dlat = lat[1:] - lat[:-1]
    dlat = numpy.mean(precision_factor * _dlat) / precision_factor
    x0, dxx, dxy, y0, dyx, dyy = [lon0, dlon, 0, lat0, 0, dlat]

    logger.debug(f'Geotransform: {x0} {dxx} {dxy} {y0} {dyx} {dyy}')

    # Compute number of GCPs (except the ones for bottom and right edge)
    # according to the requested resolution, i.e. the number of digital points
    # between two GCPs
    nlat = len(lat)
    nlon = len(lon)
    gcp_nlat = numpy.ceil(nlat / gcp_lat_spacing).astype('int')
    gcp_nlon = numpy.ceil(nlon / gcp_lon_spacing).astype('int')

    logger.debug(f'{nlat}, {nlon}, {gcp_lat_spacing}, {gcp_lon_spacing},'
                 f'{gcp_nlat}, {gcp_nlon}')

    # Compute matrix indices for the GCPs
    gcp_lin = numpy.arange(gcp_nlat) * gcp_lat_spacing
    gcp_pix = numpy.arange(gcp_nlon) * gcp_lon_spacing

    # Add an extra line and column to geolocate the bottom and right edges of
    # the data matrix
    gcp_lin = numpy.concatenate((gcp_lin, [nlat]))
    gcp_pix = numpy.concatenate((gcp_pix, [nlon]))

    # GCP pixels are located at the edge of data pixels, meaning that the
    # center of the first data pixel is located at (0.5, 0.5) in the GCPs
    # matrix.
    x0 = x0 - 0.5 * dxx
    y0 = y0 - 0.5 * dyy

    # Compute GCP geographical coordinates expressed in lat/lon
    _gcp_lin = gcp_lin[:, numpy.newaxis]
    _gcp_pix = gcp_pix[numpy.newaxis, :]
    gcp_lat = y0 + dyx * _gcp_pix + dyy * _gcp_lin
    gcp_lon = x0 + dxx * _gcp_pix + dxy * _gcp_lin
    return gcp_lon[0, :], gcp_lat[:, 0], gcp_lin, gcp_pix
