import density_Gaussian2D_test_new as density
import constants_new as const
import numpy as np


"""
Configuration File:

This configuration file configures

- Stellar Wind Structure

- Density Structure of the Tail

- Photoionization Rate

- ENA

- Sampled Parameters

- Priors

Quick description of particular configuration file:

ENA -off
HillSphere -off
TailTemp: 10^4K
SWTemp: 0.5 * 10^6K
Height Vary Fac: 0.3
Same as config but different random seeds

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

        return self.get_u(r) / self.a(r)


    def ram_pressure(self,r):

        return (self.mdot * self.get_u(r)) / (4 * np.pi * r**2)

    def post_shock_rho(self, r, cos_angle):

        M_n = np.where(self.M(r) * cos_angle < 1, 1, self.M(r) * cos_angle)

        return self.rho(r) * (((self.gamma + 1) * M_n**2) / ((self.gamma - 1) * M_n**2 + 2))

    def post_shock_pressure(self, r, cos_angle):

        M_n = np.where(self.M(r) * cos_angle < 1, 1, self.M(r) * cos_angle) #make min M_n equal 1

        post_shock_P = self.rho(r) * self.a(r)**2 * (1 + (2 * self.gamma / (self.gamma + 1)) * (M_n**2 - 1))

        return post_shock_P

def make_stellar_wind(parameters):

    SW = StellarWind(parameters['mdot_star'], 1, lambda r : get_u_stellar_wind(r, parameters['v_stellar_wind']), lambda r : get_T_stellar_wind(r, parameters['T_stellar_wind']))

    return SW

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

def make_photoionization_rate(parameters):

    def photoionization_rate1(r):

        return photoionization_rate(r, parameters['L_EUV'])

    return photoionization_rate1


"""
---------------------------------------------------------------------------------------------------------------------
Density Structure
"""

get_D_dimensionless = density.Gaussian2D.make_D_interpolant(is_zeta_zero = True)

def make_rho_struc(parameters):

    c_s = np.sqrt(2) * np.sqrt(const.k_b * 10**4 / const.m_proton)   #parameters['c_s_planet']
    mass_s = parameters['mass_s']
    a = parameters['semimajoraxis']
    SW = StellarWind(parameters['mdot_star'], 1, lambda r : get_u_stellar_wind(r, parameters['v_stellar_wind']), lambda r : get_T_stellar_wind(r, parameters['T_stellar_wind']))

    def get_alpha(position, velocity):

        r = np.sqrt(np.sum(position**2, axis = 1))

        return np.sqrt(c_s**2 * r**3 / (const.G * mass_s))

    def get_beta(position, velocity):
        r = np.sqrt(np.sum(position**2, axis = 1))
        return np.sqrt(2 * c_s**2 * r**3 / (const.G * mass_s))

    def get_zeta(position, velocity):

        return 0

    def get_PswD(position, velocity):

        r = np.sqrt(np.sum(position**2, axis = 1))

        u = np.sqrt(np.sum(velocity**2, axis = 1))

        cos_ang = np.sqrt(np.sum(np.cross(position, velocity)**2, axis = -1)) / (r * u)

        vary_fac = 0.3

        return vary_fac * SW.post_shock_pressure(r, cos_ang)


    rho_struc = density.Gaussian2D(parameters['mdot_planet'], c_s, get_alpha, get_beta, get_zeta, get_PswD, get_D_dimensionless)

    return rho_struc

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

is_ENA_on = False
"""
---------------------------------------------------------------------------------------------------------------------
Sampled Parameters
"""

"""table of parameters : {'mass_s', 'radius_s' STAR PARAM
                          'mass_p', 'radius_p', 'semimajoraxis', 'inclination' PLANET PARAM
                          'c_s_planet', 'mdot_planet' , v_stellar_wind', 'mdot_star', 'T_stellar_wind', 'L_EUV', 'angle' MODEL PARAM
                          'u_ENA', 'L_mix'} ENA param """

constant_parameters = {'mass_s' : 0.45*const.m_sun, 'radius_s' : 0.425*const.r_sun,
                       'mass_p' : 0.07*const.m_jupiter, 'radius_p' : 0.35 * const.r_jupiter, 'semimajoraxis' : 4.35e11,
                       'T_stellar_wind' : 0.5e6}

sampled_parameters = ['c_s_planet', 'mdot_planet', 'v_stellar_wind', 'mdot_star', 'L_EUV', 'angle', 'inclination']
sampled_parameter_guess = np.array([6.3, 8.5, 7, 11.5, 27.7, (2.5/4)*np.pi, 1.53])

#assert that dimensions make sense

if is_ENA_on:
    assert len(constant_parameters) + len(sampled_parameters) == 15
else:
    assert len(constant_parameters) + len(sampled_parameters) == 13


"""
--------------------------------------------------------------------------------------------------------------------
Priors
"""

def evaluate_log_prior(lp, constant_parameters):
    """
    Parameters
    --------------------
    lp : log_sampled_parameters

    constant parameters :
    Returns
    --------------------
    """
    #calculate energy limited mass loss rate

    F_XUV = 10**lp['L_EUV'] / (4 * np.pi * constant_parameters['semimajoraxis']**2)
    energy_limited_mlr = np.pi * F_XUV * constant_parameters['radius_p']**3 / (const.G * constant_parameters['mass_p'])
    #first check and calculate prior

    #uniform(and log uniform priors)
    if 5.2 <= lp['c_s_planet'] <= 6.5\
    and 8 <= lp['mdot_planet'] <= np.log10(energy_limited_mlr)\
    and 6.5 <= lp['v_stellar_wind'] <= 8.5\
    and 10 <= lp['mdot_star'] <= 13\
    and 26 <= lp['L_EUV'] <= 29\
    and np.pi/2 <= lp['angle'] <= np.pi:
    #and 0.01 <= lp['L_mix'] <= 0.1\
    #and 6.4 <= lp['u_ENA'] <= 8:

        #gaussian priors
        mu = 1.51
        sigma = 0.02
        lp_val = - 0.5 * ((lp['inclination'] - mu)**2 / sigma **2 + np.log(2 * np.pi * sigma**2))

        return lp_val

    else:

        return -np.inf

"""
transit range
------------------------------------------------------------------------------------------------------------------------
"""

transit_rng = (1.3, 31.4)
tgrid = np.concatenate((np.linspace(0.8, 10, 23), np.linspace(25.2, 31.5, 18)))

"""
random seeds
----------------------------------------------------------------------------------------------------------------------------
"""

random_seed_init_guess = 209189

random_seed_chain = 62073003


"""
-----------------------------------------------------------------------------------------------------------------------------------

Here we list a number of parameters which we fix throughout the simulation. These are split into physical parameters that control the tail and simulation parameters
that control the

Tail Temp :

SW Temp :

Height:

Star Cells : 15

Z cells size : 0.1 min(height, depth)

Number of ENA cells split: 2 * (max height / min height) // 0.1 + 1
"""
