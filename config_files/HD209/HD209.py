import density_Gaussian2D_test_new as density
import constants_new as const
import numpy as np
import HD209_Obs_Package.chi2pkg.chi2 as c2
import HD209_Obs_Package.chi2pkg.stis as stis
from astropy import table




"""
Configuration File: HD209 System

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
configuration parameters list
--------------------------------------------------------------------------------------------------------------------
"""

configuration_parameters = {'make_rho_struc' : make_rho_struc, 'make_stellar_wind' : make_stellar_wind, 'make_photoionization_rate' : make_photoionization_rate}
configuration_parameters = [configuration_parameters]


"""
---------------------------------------------------------------------------------------------------------------------
Sampled Parameters
"""

"""table of parameters : {'mass_s', 'radius_s' STAR PARAM
                          'mass_p', 'radius_p', 'semimajoraxis', 'inclination' PLANET  PARAM
                          'c_s_planet', 'mdot_planet', v_stellar_wind', 'mdot_star', 'T_stellar_wind', 'L_EUV', 'angle' MODEL PARAM
                          'u_ENA', 'L_mix'} ENA param """

constant_parameters_star = {'mass_s' : 1.07*const.m_sun, 'radius_s' : 1.2*const.r_sun, 'T_stellar_wind' : 0.5e6}
constant_parameters_planet = {'mass_p' : 0.73*const.m_jupiter, 'radius_p' : 1.4*const.r_jupiter, 'semimajoraxis' : 0.046*1.5e13, 'inclination' : 1.513}
constant_parameters_planet = [constant_parameters_planet]


sampled_parameters = ['c_s_planet', 'mdot_planet', 'v_stellar_wind', 'mdot_star', 'L_EUV', 'angle']
sampled_parameter_guess = np.array([6, 10, 7.4, 12, 28, (3/4)*np.pi])


planet_key_list = ['c_s_planet', 'mdot_planet', 'v_stellar_wind', 'mdot_star', 'L_EUV', 'angle']
key_list = ['c_s_planet', 'mdot_planet', 'v_stellar_wind', 'mdot_star', 'L_EUV', 'angle']
mcmc_parameters_key_list = [[planet_key_list, key_list]]

is_mlr_ratio = False

#assert that dimensions make sense

"""if is_ENA_on:
    assert len(constant_parameters) + len(sampled_parameters) == 21
else:
    assert len(constant_parameters) + len(sampled_parameters) == 19"""


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
    and 6.5 <= lp['v_stellar_wind'] <= 8\
    and 10.3 <= lp['mdot_star'] <= 13\
    and 26 <= lp['L_EUV'] <= 30\
    and np.pi/2 <= lp['angle'] <= np.pi:

        return 0

    else:

        return -np.inf


evaluate_log_prior = [evaluate_log_prior]

"""
observational fitting functions
----------------------------------------------------------------------------------------------------------------------------
"""

#planet b

vgrid = np.concatenate((np.arange(-2e8, -7e7, 1e7), np.arange(-4e7, 4e7, 1e5), np.arange(7e7, 2.1e8, 1e7)))
wavgrid = (1 + np.asarray(vgrid) / const.c) * 1215.67
tgrid = np.linspace(-4.5, 8, 25)

path_aggregated_data = 'HD209_Obs_Package/chi2pkg/g140m_aggregated_data_in_system_frame.fits'
data = table.Table.read(path_aggregated_data)
path_oot_line_profile = 'HD209_Obs_Package/chi2pkg/vidal-madjar_lya_profile.ecsv'
oot_model = table.Table.read(path_oot_line_profile)
fine_wavgrid = (1 + np.asarray(np.arange(-2e8, 2e8, 1e5)) / const.c) * 1215.67
oot_profile = np.interp(fine_wavgrid, oot_model['w'], oot_model['y'])
w_line = fine_wavgrid
y_line = oot_profile
line = table.Table((w_line, y_line), names=['w', 'y'])
spec = stis.g140m
g140m_lya = c2.FitPackage(wavgrid, tgrid, spec, data, line, wrest=1215.67, normalize=((-350,-115),(120,350)))


fit_package = [g140m_lya]
logL_fnct = [g140m_lya.compute_logL_with_spectrum]

"""
random seeds
----------------------------------------------------------------------------------------------------------------------------
"""

random_seed_init_guess = 176

random_seed_chain = 250697


"""
-----------------------------------------------------------------------------------------------------------------------------------
"""

transit_parametersb = {'n_star_cells' : 15, 'n_z_cells' : None}


transit_parameters = [transit_parametersb]

"""
------------------------------------------------------------------------------------------------------------------------------------
Here we list a number of parameters which we fix throughout the simulation. These are split into physical parameters that control the tail and simulation parameters
that control the

Tail Temp :

SW Temp :

Height:

Star Cells : 15

Z cells size : 0.1 min(height, depth)

Number of ENA cells split: 2 * (max height / min height) // 0.1 + 1
"""
