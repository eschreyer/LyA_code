import numpy as np
import constants_new as const
import scipy.special as sp_special


def doppler_shift(w0, velocity):
    return (1 + velocity / const.c)*w0


def voigt_profile(w, w0, gauss_sigma, lorentz_HWHM):
    """Return

    Parameters
    --------------------
    w:

    w0:

    gauss_sigma: standard variation of

    lorentz_HWHM:

    """

    return sp_special.voigt_profile(w-w0, gauss_sigma, lorentz_HWHM)


def voigt_xsection(w, w0, f, Gamma, T, mmw):
    """
    Compute the absoprtion cross section using the voigt profile for a line

    Parameters
    ------------------------
    w:

    w0: Line center wavelength (may be doppler shifted)

    f:

    Gamma:

    T:

    mass:

    Returns
    --------------------------
    """

    lorentz_HWHM = Gamma / (4*np.pi)
    #fixed from last commit
    gauss_sigma = np.sqrt(const.k_b*T/(mmw * const.c**2))*w0
    xsection = np.pi * const.e**2 / (const.m_e * const.c) * f * voigt_profile(w, w0, gauss_sigma, lorentz_HWHM)
    return xsection


def LyA_xsection(w, absorber_v, T):
    absorber_w0 = doppler_shift(const.LyA_linecenter_w, absorber_v) #in the frame of the object emitting light
    f_LyA = 4.1641e-1
    Gamma_LyA = 6.2649e8 #s^-1
    LyA_xsection = voigt_xsection(w, absorber_w0, f_LyA, Gamma_LyA, T, const.m_proton)
    return LyA_xsection


def d_tau(mass_density, xsection, mmw, ds):
    d_tau = mass_density / mmw * xsection * ds
    return d_tau
