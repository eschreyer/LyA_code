import density_Gaussian2D_test_new as density
import constants_new as const
import numpy as np


"""
Configuration File:

This configuration file configures

- Stellar Wind Structure

- Density Structure of the Tail

- Photoionization Rate

- Sampled Parameters

- Priors

--------------------------------------------------------------------------------------------------------------------
Stellar Wind

The velocity structure of the stellar wind. We assume the velocity has reached its asymptotic value at the position of the planet
"""

def get_u_stellar_wind(r, v_at_planet):

    return v_at_planet

def get_T_stellar_wind(r, T_at_planet):

    return T_at_planet



class StellarWind():

    def __init__(self, mdot, gamma, get_u, get_T):

        self.mdot = mdot
        self.gamma = gamma
        self.get_u = get_u
        self.get_T = get_T

    def rho(self, r):

        return self.mdot / (4 * np.pi * r**2 * self.get_u(r))

    def a(self, r):

        return np.sqrt(2 * self.gamma * const.k_b * self.get_T(r) / const.m_proton)


    def M(self, r):

        return self.a(r) / self.get_u(r)


    def ram_pressure(self,r):

        return (self.mdot * self.get_u(r)) / (4 * np.pi * r**2)

    def post_shock_rho(self, r, cos_angle):

        M_n = self.M(r) * cos_angle

        return self.rho(r) * (((self.gamma + 1) * M_n**2) / ((self.gamma - 1) * M_n**2 + 2))

    def post_shock_pressure(self, r, cos_angle):

        M_n = np.where(self.M(r) * cos_angle < 1.1, self.M(r), 1.1) #why did i put a cutoff at 1.1

        post_shock_P = self.rho(r) * self.a(r)**2 * (1 + (2 * self.gamma / (self.gamma + 1)) * (M_n**2 - 1))

        return post_shock_P

""""
--------------------------------------------------------------------------------------------------------------------
Photionization Rate
"""

def photoionization_rate(r, L_EUV):

    """The optically thin photoionization rate. This is calculated by putting all the flux into photons with a representative energy of 20ev"""

    E_phot = 20 * 1.60218e-12

    F_phot = L_EUV / (4 * np.pi * r**2 * E_phot)

    xsection = const.HI_crosssection * (13.6 / 20)**3

    photoionization_rate = F_phot * xsection

    return photoionization_rate


"""
---------------------------------------------------------------------------------------------------------------------
Density Structure
"""

get_D_dimensionless = density.Gaussian2D.make_D_interpolant(is_zeta_zero = True)

def make_rho_struc(parameters):

    c_s = 1e6 #parameters['c_s_planet']
    mass_s = parameters['mass_s']
    a = parameters['semimajoraxis']
    SW = StellarWind(parameters['mdot_star'], 1, lambda r : get_u_stellar_wind(r, parameters['v_stellar_wind']), lambda r : get_T_stellar_wind(r, parameters['T_stellar_wind']))

    def get_alpha(position, velocity):

        r = np.sqrt(np.sum(position**2, axis = 1))

        return np.sqrt(c_s**2 * a**3 / (const.G * mass_s))

    def get_beta(position, velocity):
        r = np.sqrt(np.sum(position**2, axis = 1))
        return np.sqrt(2 * c_s**2 * r**3 / (const.G * mass_s))

    def get_zeta(position, velocity):

        return 0

    def get_PswD(position, velocity):

        r = np.sqrt(np.sum(position**2, axis = 1))

        u = np.sqrt(np.sum(velocity**2, axis = 1))

        cos_ang = np.sqrt(np.sum(np.cross(position, velocity)**2, axis = -1)) / (r * u)

        return SW.post_shock_pressure(r, cos_ang)


    rho_struc = density.Gaussian2D(parameters['mdot_planet'], c_s, get_alpha, get_beta, get_zeta, get_PswD, get_D_dimensionless)

    return rho_struc

"""
---------------------------------------------------------------------------------------------------------------------
Sampled Parameters
"""







"""
--------------------------------------------------------------------------------------------------------------------
Priors
"""



"""
--------------------------------------------------------------------------------------------------------------------
Energetic Neutral Atoms
"""

class ENA():
#ENA class:


    def __init__(self, SW, u_ENA, L_mix):

        self.SW = SW
        self.u_ENA = u_ENA
        self.L_mix = L_mix


    def get_rho(self, position, velocity):

        r = np.sqrt(np.sum(position**2, axis = 1))

        u = np.sqrt(np.sum(velocity**2, axis = 1))

        cos_ang = np.sqrt(np.sum(np.cross(position, velocity)**2, axis = -1)) / (r * u)

        return self.SW.post_shock_rho(r, cos_ang)


def make_ENA(parameters):

    SW = StellarWind(parameters['mdot_star'], 1, lambda r : get_u_stellar_wind(r, parameters['v_stellar_wind']), lambda r : get_T_stellar_wind(r, parameters['T_stellar_wind']))
    u_ENA = parameters['u_ENA']
    L_mix = parameters['L_mix']

    ENA_c = ENA(SW, u_ENA, L_mix)

    return ENA_c
