import trajectory_tail_cartesian_new as ttc
import do_transit_new as dt
import numpy as np
import constants_new as const
import LyA_transit_datatypes_new as LyA
import matplotlib.pyplot as plt
import Observation_new_v2.chi2_gj436 as obs
import xsection_new as xs
import constants_new as const
import logging
import do_transit_hill as dth
import Parker_wind_planet_new as pw
import scipy.interpolate as sp_int


def convert_to_linspace(dic):
    new_dict = {}
    for key in dic:
        if key == 'angle' or key == 'inclination':
            new_dict[key] = dic[key]
        else:
            new_dict[key] = 10**dic[key]
    return new_dict


def make_log_posterior_fn(constant_parameters, evaluate_log_prior, configuration_parameters, tgrid, transit_rng, hill_sphere = False, only_blue = False, weight_fluxes = False):
    """
    Parameters
    ------------------
    constant_parameters :

    config:

    Returns
    ------------------
    evaluate_posterior : the
    """
    #make comparison to obs fnctn
    vgrid = np.concatenate((np.arange(-1e8, -4e7, 4e6), np.arange(-4e7, 4e7, 1e5), np.arange(4e7, 1.04e8, 4e6)))  #(-1000 km/s, 1000 km/s)                                               #-1000km/s to 1000km/s
    wavgrid = (1 + np.asarray(vgrid) / const.c) * 1215.67                           #IN ANGSTROMS! (Not CGS)
    wgrid = (1 - np.asarray(vgrid) / const.c) * const.LyA_linecenter_w              #IN CGS


    phasegrid = tgrid * np.sqrt(constant_parameters['mass_s'] * const.G / constant_parameters['semimajoraxis']**3) * 3600 + np.pi/2

    oot_profile, oot_data, transit_data, simulate_spectra, get_lightcurves, compute_chi2, compute_logL = obs.make_transit_chi2_tools(wavgrid, tgrid, transit_rng)

    #make transit fnct
    do_transit, _ = dt.make_transit_tools(constant_parameters['radius_s'], 15)

    if hill_sphere == True:

        do_transit_hill = dth.make_transit_tools_hill_and_ena(constant_parameters['radius_s'], 15)


    def evaluate_posterior(mcmc_log_parameters):

        ###
        parameters = {**constant_parameters, **convert_to_linspace(mcmc_log_parameters)}
        star = LyA.Star(mass = parameters['mass_s'], radius = parameters['radius_s'])
        planet = LyA.Planet(mass = parameters['mass_p'], radius = parameters['radius_p'], semimajoraxis = parameters['semimajoraxis'], inclination = parameters['inclination'])
        model_parameters = LyA.ModelParameters(c_s_planet = parameters['c_s_planet'], mdot_planet = parameters['mdot_planet'], v_stellar_wind = parameters['v_stellar_wind'], mdot_star = parameters['mdot_star'], T_stellar_wind = parameters['T_stellar_wind'], L_EUV = parameters['L_EUV'], angle = parameters['angle'])

        #make_density_structure
        rho_struc = configuration_parameters['make_rho_struc'](parameters)

        #make_stellar_wind
        SW = configuration_parameters['make_stellar_wind'](parameters)

        #make_photoionisation_rate
        photoionization_rate = configuration_parameters['make_photoionization_rate'](parameters)

        #get planet keplerian velocity
        omega_p = np.sqrt(const.G * parameters['mass_s'] / parameters['semimajoraxis']**3)

        #make_ENA_structure
        if 'make_ENA' in configuration_parameters:
            ENA = configuration_parameters['make_ENA'](parameters)
        else:
            ENA = None

        if hill_sphere == True:
            def density(z, y_c):
                r = np.sqrt(z**2 + y_c**2)
                return pw.density_planetary_wind(r, star, planet, model_parameters)

            def velocity(z, y_c):
                r = np.sqrt(z**2 + y_c**2)
                return pw.velocity_planetary_wind(r, star, planet, model_parameters) * z / r

            neutral_frac = pw.neutral_frac_planetary_wind(star, planet, model_parameters, photoionization_rate, tau = True)
            neutral_frac_interpolant = sp_int.InterpolatedUnivariateSpline(neutral_frac.t, neutral_frac.y, ext = 3)
            def neutral_fraction(z, y_c):
                r = np.sqrt(z**2 + y_c**2)
                return neutral_frac_interpolant(r)

            pw_functions = {'density' : density, 'neutral_fraction' : neutral_fraction, 'z_velocity' : velocity}

        #evaluate prior
        logP = evaluate_log_prior(mcmc_log_parameters, constant_parameters)



        if logP == -np.inf:
            return -np.inf

        else:
            try:
                tail_solution_cartesian = ttc.trajectory_solution_cartesian(star, planet, model_parameters, rho_struc, SW, photoionization_rate)
            except (ValueError, RuntimeWarning):
                logging.exception(f"the parameters are {str(model_parameters)}")
                return -np.inf
            else:
                if tail_solution_cartesian.success == False: #if solution no found assume parameters unreasonable
                    return -np.inf
                elif tail_solution_cartesian.t_events[1].size: #check if stopped by epicycle event
                    return -np.inf
                else:
                    tail = ttc.trajectory_solution_polar(star, planet, model_parameters, rho_struc, SW, photoionization_rate)
                    phase, model_intensity = do_transit(tail, phasegrid, wgrid, rho_struc, omega_p, parameters['inclination'], ENA = ENA)
                    if hill_sphere == True:
                        phse_hill, model_intensity_hill = do_transit_hill(parameters, pw_functions, phasegrid, wgrid, parameters['inclination'], ENA = ENA)
                        model_intensity *= model_intensity_hill
                    logL = compute_logL(1 - model_intensity, only_blue = only_blue, weight_fluxes = weight_fluxes)
                    logPosterior = logL + logP
                    return logPosterior


    return evaluate_posterior
