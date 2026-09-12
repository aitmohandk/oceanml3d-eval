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
Tools to compute power spectrum, cospectrum, coherence between two signals
"""

import numpy
from scipy import signal
import math
from typing import Tuple, Optional
import logging

logger = logging.getLogger()
handler = logging.StreamHandler()
logger.addHandler(handler)


def norma(mat: numpy.ndarray):
    """
    Normalize array on ubyte
    Args:
        mat (numpy.ndarray): float matrix
    Returns
        ubyte matrix
    """
    mat1 = mat.real
    mat1 -= mat1.min()
    mat1 *= 255. / mat1.max()
    return mat1


def make_taper(tap, N: int) -> numpy.ndarray:
    """
    Make taper to handle edges
    Args:
        tap:
        N (int): size of taper
    Returns:
        taper
    """
    ntaper = int(tap * N + 0.5)
    taper = numpy.ones((N))
    taper[: ntaper] = numpy.cos(numpy.arange(0, ntaper) / float(ntaper - 1)
                                * numpy.pi/2. + 3.*numpy.pi/2.)
    taper[N - ntaper: N] = numpy.cos(numpy.arange(0, ntaper)
                                     / float(ntaper - 1) * numpy.pi/2.)
    return taper


def spec_1d(L: float, dx: float, H: numpy.ndarray
            ) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Make 1d spectrum
    Args:
        L (float): length
        dx (float): resolution
        H (numpy.ndarray): 1d array
    Returns:
        Power spectrum, frequency
    """
    Lx = L * dx
    coeff = Lx / (2. * numpy.pi)
    PS = numpy.fft.fft(H)
    ff = numpy.hstack([range(int(L/2)), 0, range(-int(L/2) + 1, 0, 1)])/coeff
    PP = numpy.real(PS * numpy.conj(PS))
    dff = ff[1] - ff[0]
    PP = PP / float(L) / dff
    return PP, ff


def spec_1d_withtaper(ll: int, H: numpy.ndarray, tap: float
                      ) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Make 1d spectrum
    Args:
        L (float): length
        dx (float): resolution
        H (numpy.ndarray): 1d array
    Returns:
        Power spectrum, frequency
    """
    N = numpy.shape(ll)[0]
    L = N * (ll[1] - ll[0])
    ny = numpy.shape(H)[0]
    if N != ny:
        logger.critical(f'Error in 1d spectrum {N} != {ny}')
        exit(1)
    taper = make_taper(tap, N)
    H = H - numpy.mean(H)
    H = signal.detrend(H)
    H = H * taper
    H2 = numpy.zeros(2*ny)
    H2[: ny] = + H
    H2[ny:] = H[::-1]
    PS = numpy.fft.fft(H2)
    ff = numpy.arange(1, N + 1)/float(2. * L)
    PP = (0.5/float(ff[-1])*1.0 / float(N)
          * (numpy.sqrt(numpy.real(PS[2: int(N + 2)]
                        * numpy.conj(PS[2:int(N + 2)]))))**2)
    return PP, ff


def get_kxky(L: int, M: int, dx: float, dy: float, dim: Optional[bool] = True
             ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray,
                        numpy.ndarray, numpy.ndarray]:
    """
    Compute wave number in 2d at a given resolution
    Args:
        L (int): first dimension
        M (int): second dimension
        dx (float): resolution in first dimension
        dy (float): resolution in second dimension
        dim(bool): true for dimensionless wave number
    Returns:
        1d wave number in x direction, y direction, 2d wave number in x
        direction, y direction, wave number norm
    """
    if dim is True:
        Lx = L * dx
        Ly = M * dy
    else:
        Lx = numpy.pi  # Domain length regarding x and y in the physical space
        Ly = Lx * M / L * dy / dx  # Take into account anisotropic grid

    coefx = Lx / (2 * math.pi)
    coefy = Ly / (2 * math.pi)
    kx = numpy.hstack([numpy.arange(0, (L/2)), 0,
                       numpy.arange(-(L/2) + 1, 0, 1)]) / coefx
    ky = numpy.hstack([numpy.arange(0, (M/2)), 0,
                       numpy.arange(-(M/2) + 1, 0, 1)]) / coefy
    [kky, kkx] = numpy.meshgrid(ky, kx)
    kk = numpy.sqrt(kkx*kkx + kky*kky)
    return kx, ky, kkx, kky, kk


def detrend_harmonic(phi: numpy.ndarray) -> numpy.ndarray:
    """
    Detrend signal
    Args:
        phi (numpy.ndarray): 2D Signal
    Returns:
        Detrended signal
    """
    M, L = numpy.shape(phi)
    cache = numpy.zeros((M, L))
    hann = numpy.zeros((M))
    hann = signal.windows.hamming(M)
    for i in range(L):
        cache[:, i] = hann
    phi = phi * cache
    return phi


def spec_2d(dx: float, dy: float, phi: numpy.ndarray
            ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    Compute spectrum from 2d data
    Args:
        dx (float): resolution in first dimension
        dy (float): resolution in second dimension
        phi (numpy.ndarray): 2d signal
    Returns
        Phi spectrum, phi phase, Wave number
    """

    # Compute FFT
    M, L = numpy.shape(phi)
    phi = detrend_harmonic(phi)
    [kx, ky, kkx, kky, kk] = get_kxky(M, L, dx, dy)
    hat_phi = numpy.fft.fft2(phi)
    hat_phi_abs = numpy.real(hat_phi * numpy.conj(hat_phi))
    hat_phi_phase = numpy.angle(hat_phi)
    if M < L:
        k = kx[0: int(M / 2)]
    else:
        k = ky[0: int(L / 2)]
    dk = k[1] - k[0]
    epsilon = dk / 2
    # values centered on intervalles
    k = k + dk

    # Integration on the spectral band
    spec_phi = numpy.zeros(numpy.shape(k))
    for ii in range(numpy.shape(k)[0]):
        _ind = numpy.where((kk >= (k[ii] - epsilon)) & (kk < (k[ii] + epsilon)))
        spec_phi[ii] = numpy.mean(hat_phi_abs[_ind])

    # Danioux 2011 normalisation
    spec_phi = spec_phi / (L * L * M * M) / dk
    return spec_phi, hat_phi_phase, k


def co_spec_uv(dx: float, dy: float, phi1: numpy.ndarray, phi2: numpy.ndarray
               ) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Cospectrum between two signals of the same dimension
    Args:
        dx (float): resolution in first dimension
        dy (float): resolution in second dimension
        phi1 (numpy.ndarray): first signal
        phi2 (numpy.ndarray): second signal
    Returns:
        Cospectrum and wave number
    """
    M, L = numpy.shape(phi1)
    [kx, ky, kkx, kky, kk] = get_kxky(M, L, dx, dy, dim=True)
    phi1 = detrend_harmonic(phi1)
    phi2 = detrend_harmonic(phi2)

# # COMPUTE FFT
# #---------------
    hat_phi1 = numpy.fft.fft2(phi1)
    hat_phi2 = numpy.fft.fft2(phi2)
    hat_cospec1 = numpy.real(hat_phi1*numpy.conj(hat_phi1))
    hat_cospec2 = numpy.real(hat_phi2*numpy.conj(hat_phi2))
    hat_cospec = hat_cospec1 + hat_cospec2
    if M < L:
        k = kx[0:int(M/2)]
    else:
        k = ky[0:int(L/2)]
    dk = k[1] - k[0]
    epsilon = dk/2
    # # values centered on intervalles
    k = k + dk

# # Integration on the spectral band
    cospec_uv = numpy.zeros(numpy.shape(k))
    for ii in range(numpy.shape(k)[0]):
        _ind = numpy.where((kk >= (k[ii]-epsilon)) & (kk < (k[ii]+epsilon)))
        cospec_uv[ii] = numpy.sum(hat_cospec[_ind])

# # Danioux 2011 normalisation
    cospec_uv = cospec_uv/(L*L*M*M)/dk
    return cospec_uv, k


def coherence(dx: float, dy: float, phi: numpy.ndarray, phi2: numpy.ndarray
              ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray,
                         numpy.ndarray]:
    """
    Compute Coherence between two signals
    Args:
        dx (float): resolution in first dimension
        dy (float): resolution in second dimension
        phi (numpy.ndarray): first signal
        phi2 (numpy.ndarray): second signal
    Returns:
        First signal spectrum, second signal spectrum, Coherence and
        wave number
    """

    M, L = numpy.shape(phi)

    # Compute FFT
    [kx, ky, kkx, kky, kk] = get_kxky(M, L, dx, dy)
    hat_phi = numpy.fft.fft2(phi)
    hat_phi2 = numpy.fft.fft2(phi2)
    hat_phi_abs = numpy.real(hat_phi * numpy.conj(hat_phi))
    hat_phi_abs2 = numpy.real(hat_phi2 * numpy.conj(hat_phi2))

    hat_phi_phase = numpy.angle(hat_phi)
    hat_phi_phase2 = numpy.angle(hat_phi2)
    coherence = (numpy.cos((hat_phi_phase2) - (hat_phi_phase)))
    if M < L:
        k = kx[0: int(M/2)]
    else:
        k = ky[0: int(L/2)]
    dk = k[1] - k[0]
    epsilon = dk / 2

    # Integration on the spectral band
    spec_phi = numpy.zeros(k)
    spec_phi2 = numpy.zeros(k)
    spec_coherence = numpy.zeros(k)
    ones = numpy.ones(numpy.shape(coherence))
    for ii in range(numpy.shape(k)[0]):
        _ind = numpy.where((kk >= (k[ii] - epsilon))
                           & (kk < (k[ii] + epsilon)))
        spec_coherence[ii] = numpy.sum(coherence[_ind])/numpy.sum(ones[_ind])
        spec_phi2[ii] = numpy.sum(hat_phi_abs2[_ind])
        spec_phi[ii] = numpy.sum(hat_phi_abs[_ind])

    # Danioux 2011 normalisation
    spec_phi = spec_phi / (L * L * M * M) / dk
    spec_phi2 = spec_phi2 / (L * L * M * M) / dk

    return spec_phi, spec_phi2, spec_coherence, k


def gen_noise(x: numpy.ndarray, y: numpy.ndarray, L: list, E: list,
              nwave: int) -> numpy.ndarray:
    """
    Generate spectral noise
    Args:
        x: longitude coordinate
        y: latitude coordinate
        L: Tuple
        E:
        nwave (int): number of wave number to consider
    Returns:
        Random noise following
    """
    import math

# # Prepare Coordinate
    nx = numpy.shape(x)[0]
    ny = numpy.shape(y)[0]
    Y, X = numpy.meshgrid(y, x)
    H = numpy.zeros((nx, ny))

# # Compute Random vecteur in an annular (radius (k1, k2))
    dir = 2 * math.pi * numpy.random.random(nwave)
    logk1 = math.log10(2 * math.pi / L[0])
    logk2 = math.log10(2 * math.pi / L[1])
    log_ensk = (logk2 - logk1) * numpy.random.random(nwave) + logk1
    ensk = 10**log_ensk[:]
    enskx = ensk * numpy.cos(dir[:])
    ensky = ensk * numpy.sin(dir[:])

# # Compute random phase
    phi = 2 * math.pi * numpy.random.random(nwave)
# # Compute noise proportional to random noise
    logPower1D = (numpy.log10(E[0]) + (log_ensk - logk1) / (logk2 - logk1)
                  * (math.log10(E[1]) - math.log10(E[0])))
    Power2D = 10**(logPower1D) / (2 * math.pi * ensk)
    A = numpy.sqrt(Power2D * ensk**2 * math.pi * (math.pi * 2)**2
                   * ((1 / L[1])**2 - (1 / L[0])**2)
                   / sum(ensk**2) / numpy.sqrt(2))

# # Sum noise in different directions
    for wave in range(0, nwave):
        H = H + A[wave]*numpy.cos(enskx[wave]*X + ensky[wave]*Y + phi[wave])
    return H
