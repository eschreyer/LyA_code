import density_Gaussian2D_test_new as density
import constants_new as const
import numpy as np
import TOI_776_Obs_Package4.chi2 as c2
import TOI_776_Obs_Package4.stis as stis
from astropy import table
import functools




"""
Configuration File: TOI766 System

This configuration file configures

- Stellar Wind Structure

- Density Structure of the Tail

- Photoionization Rate

- ENA

- Sampled Parameters

- Priors

We include uncertainty in mass of the planet

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

configuration_parametersb = {'make_rho_struc' : make_rho_struc, 'make_stellar_wind' : make_stellar_wind, 'make_photoionization_rate' : make_photoionization_rate}
configuration_parametersc = {'make_rho_struc' : make_rho_struc, 'make_stellar_wind' : make_stellar_wind, 'make_photoionization_rate' : make_photoionization_rate}
configuration_parameters = [configuration_parametersb, configuration_parametersc]


"""
---------------------------------------------------------------------------------------------------------------------
Sampled Parameters
"""

"""table of parameters : {'mass_s', 'radius_s' STAR PARAM
                          'mass_pb', 'radius_pb', 'semimajoraxisb', 'inclinationb' PLANET b PARAM
                          'mass_pc', 'radius_pc', 'semimajoraxisc', 'inclinationc' PLANET c PARAM
                          'c_s_planetb', 'mdot_planetb', 'c_s_planetc', 'mdot_planetc', v_stellar_wind', 'mdot_star', 'T_stellar_wind', 'L_EUV', 'angle' MODEL PARAM
                          'u_ENA', 'L_mix' ENA param
                          'scalefacsb (2,), scalefacsc (1,)} Lya line param'"""

constant_parameters_star = {'mass_s' : 0.544*const.m_sun, 'radius_s' : 0.538*const.r_sun, 'T_stellar_wind' : 0.5e6}
constant_parameters_planetb = {'radius_p' : 1.85*const.r_earth, 'semimajoraxis' : 0.0652*1.5e13, 'inclination' : 1.565}
constant_parameters_planetc = {'radius_p' : 2.02*const.r_earth, 'semimajoraxis' : 0.10*1.5e13, 'inclination' : 1.563}
constant_parameters_planet = [constant_parameters_planetb, constant_parameters_planetc]


sampled_parameters = ['c_s_planetb', 'mdot_planetb', 'c_s_planetc', 'mdot_planetc', 'v_stellar_wind', 'mdot_star', 'L_EUV', 'angleb', 'anglec', 'mass_pb', 'mass_pc', 'scalefacsb1', 'scalefacsb2', 'scalefacsc']
sampled_parameter_guess = np.array([5.8, 8.8, 6.1, 8.8, 7, 11.5, 28.5, (3/5)*np.pi, (3/5)*np.pi, 4 * const.m_earth, 5.3 * const.m_earth, 1.0, 1.0, 1.0])

planetb_key_list = ['c_s_planetb', 'mdot_planetb', 'v_stellar_wind', 'mdot_star', 'L_EUV', 'angleb', 'mass_pb', ('scalefacsb1', 'scalefacsb2')]
planetc_key_list = ['c_s_planetc', 'mdot_planetc', 'v_stellar_wind', 'mdot_star', 'L_EUV', 'anglec', 'mass_pc', ('scalefacsc',)]
key_list = ['c_s_planet', 'mdot_planet', 'v_stellar_wind', 'mdot_star', 'L_EUV', 'angle', 'mass_p', 'scalefacs']
mcmc_parameters_key_list = [[planetb_key_list, key_list], [planetc_key_list, key_list]]

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

def evaluate_log_priorb(lp, constant_parameters):
    """
    Parameters
    --------------------
    lp : log_sampled_parameters

    constant parameters :
    Returns
    --------------------
    """
    #calculate energy limited mass loss rate
    if const.m_earth >= lp['mass_p']:
        return -np.inf

    F_XUV = 10**lp['L_EUV'] / (4 * np.pi * constant_parameters['semimajoraxis']**2)
    energy_limited_mlr = np.pi * F_XUV * constant_parameters['radius_p']**3 / (const.G * lp['mass_p'])
    #first check and calculate prior

    #uniform(and log uniform priors)
    if 5 <= lp['c_s_planet'] <= 6.35\
    and 7 <= lp['mdot_planet'] <= (np.log10(energy_limited_mlr) + 1)\
    and 6.5 <= lp['v_stellar_wind'] <= 8\
    and 10.3 <= lp['mdot_star'] <= 13\
    and 26 <= lp['L_EUV'] <= 29\
    and np.pi/2 <= lp['angle'] <= np.pi\
    and (0, 0) < lp['scalefacs'] <= (2, 2):

        #gaussian priors for inclination
        mu = np.array([4 * const.m_earth])
        sigma = np.array([0.9 * const.m_earth])
        lp_val = - 0.5 * ((np.array([lp['mass_p']]) - mu)**2 / sigma **2 + np.log(2 * np.pi * sigma**2))
        return np.sum(lp_val)

    else:

        return -np.inf


def evaluate_log_priorc(lp, constant_parameters):
    """
    Parameters
    --------------------
    lp : log_sampled_parameters

    constant parameters :
    Returns
    --------------------
    """
    #calculate energy limited mass loss rate
    if const.m_earth >= lp['mass_p']:
        return -np.inf

    F_XUV = 10**lp['L_EUV'] / (4 * np.pi * constant_parameters['semimajoraxis']**2)
    energy_limited_mlr = np.pi * F_XUV * constant_parameters['radius_p']**3 / (const.G * lp['mass_p'])
    #first check and calculate prior

    #uniform(and log uniform priors)
    if 5 <= lp['c_s_planet'] <= 6.35\
    and 7 <= lp['mdot_planet'] <= (np.log10(energy_limited_mlr) + 1)\
    and 6.5 <= lp['v_stellar_wind'] <= 8\
    and 10.3 <= lp['mdot_star'] <= 13\
    and 26 <= lp['L_EUV'] <= 29\
    and np.pi/2 <= lp['angle'] <= np.pi\
    and (0,) < lp['scalefacs'] < (2,):

        #gaussian priors for inclination
        mu = np.array([5.3 * const.m_earth])
        sigma = np.array([1.8 * const.m_earth])
        lp_val = - 0.5 * ((np.array([lp['mass_p']]) - mu)**2 / sigma **2 + np.log(2 * np.pi * sigma**2))

        return np.sum(lp_val)

    else:

        return -np.inf

evaluate_log_prior = [evaluate_log_priorb, evaluate_log_priorc]

"""
observational fitting functions
----------------------------------------------------------------------------------------------------------------------------
"""

#planet b

vgridb = np.arange(-2e7, 2e7, 2e5)
wavgridb = (1 + np.asarray(vgridb) / const.c) * 1215.67
tgridb = np.concatenate((np.linspace(-41, -40, 3), np.linspace(-1.8, 1.3, 10)))

path_aggregated_datab = 'TOI_776_Obs_Package4/pubdata/g140m_aggregated_data_in_system_frame.fits'
datab = table.Table.read(path_aggregated_datab)
datab['pha'] = datab['pha_b']
datab['phb'] = datab['phb_b']
datab['ph'] = datab['ph_b']
datab.sort('ph')
path_oot_line_profileb = 'TOI_776_Obs_Package4/pubdata/lya_fit_g140m.ecsv'
lineb = table.Table.read(path_oot_line_profileb)
specb = stis.g140m
g140m_lya = c2.FitPackage(wavgridb, tgridb, specb, datab, lineb, wrest=1215.67, epoch_divisions=(2459550,))

#planet c

vgridc = np.concatenate((np.arange(-1.5e8, -4e7, 1e7), np.arange(-4e7, 4e7, 1e5), np.arange(4e7, 1.6e8, 1e7)))
wavgridc = (1 + np.asarray(vgridc) / const.c) * 1215.67
tgridc = np.concatenate((np.linspace(-43.5, -41, 8), np.linspace(-2.3, 2, 12)))

path_aggregated_datac = 'TOI_776_Obs_Package4/pubdata/g140l_aggregated_data_in_system_frame.fits'
datac = table.Table.read(path_aggregated_datac)
datac['pha'] = datac['pha_c']
datac['phb'] = datac['phb_c']
datac['ph'] = datac['ph_c']
datac.sort('ph')

#we only want data from the last epoch of observations, which happened starting on MJD 2459933
keep = datac['t'] > 2459933
datac = datac[keep]

path_oot_line_profilec = 'TOI_776_Obs_Package4/pubdata/lya_fit_rescaled_to_g140l_epoch.ecsv'
linec = table.Table.read(path_oot_line_profilec)
specc = stis.g140l
g140l_lya = c2.FitPackage(wavgridc, tgridc, specc, datac, linec, wrest=1215.67, epoch_divisions=None)

fit_package = [g140m_lya, g140l_lya]
logL_fnct = [functools.partial(g140m_lya.compute_logL_with_partial_band_lightcurve, band = [-37, 69]), g140l_lya.compute_logL_with_full_band_lightcurve]
#note band

"""
random seeds
----------------------------------------------------------------------------------------------------------------------------
"""

random_seed_init_guess = 209189

random_seed_chain = 62073003



"""
-----------------------------------------------------------------------------------------------------------------------------------
"""

transit_parametersb = {'n_star_cells' : 15, 'n_z_cells' : None}
transit_parametersc = {'n_star_cells' : 15, 'n_z_cells' : None}

transit_parameters = [transit_parametersb, transit_parametersc]

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
