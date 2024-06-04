import numpy as np
import config
import LyA_transit_datatypes_new as LyA
import constants_new as const
import do_transit_new as dt
import trajectory_tail_cartesian_new as ttc
import Observation_new.chi2_gj436 as obs
import logging


def evaluate_log_prior(lp):
    """
    Parameters
    --------------------


    Returns
    --------------------
    """
    #first check and calculate prior

    #uniform(and log uniform priors)
    if 4.7 <= lp['c_s_planet'] <= 7\
    and 8 <= lp['mdot_planet'] <= 13\
    and 25 <= lp['L_EUV'] <= 29:
        return 0 #lp_val

    else:

        return -np.inf


def convert_to_linspace(dic):
    new_dict = {}
    for key in dic:
        if key == 'angle' or key == 'inclination':
            new_dict[key] = dic[key]
        else:
            new_dict[key] = 10**dic[key]
    return new_dict



def make_log_posterior_fn_test(constant_parameters):

    datafile = np.load('test_generation1.npz')
    data_intensity = datafile['noisy_sim_data']
    data_err = datafile['err']

    vgrid = np.concatenate((np.arange(-1e8, -4e7, 4e6), np.arange(-4e7, 4e7, 1e5), np.arange(4e7, 1.04e8, 4e6)))         #(-1000km/s, 1000km/s)
    wavgrid = (1 + np.asarray(vgrid) / const.c) * 1215.67                           #IN ANGSTROMS! (Not CGS)
    wgrid = (1 - np.asarray(vgrid) / const.c) * const.LyA_linecenter_w              #IN


    tgrid = np.concatenate((np.linspace(0.8, 10, 23), np.linspace(25.2, 31.5, 18))) #IN HOURS
    phasegrid = tgrid * np.sqrt(constant_parameters['mass_s'] * const.G / constant_parameters['semimajoraxis']**3) * 3600 + np.pi/2

    oot_profile, oot_data, transit_data, simulate_spectra, get_lightcurves, compute_chi2, compute_logL = obs.make_transit_chi2_tools(wavgrid, tgrid)
    do_transit = dt.make_transit_tools(constant_parameters['radius_s'])


    def evaluate_posterior(mcmc_log_parameters):

        ###
        parameters = {**constant_parameters, **convert_to_linspace(mcmc_log_parameters)}
        star = LyA.Star(mass = parameters['mass_s'], radius = parameters['radius_s'])
        planet = LyA.Planet(mass = parameters['mass_p'], radius = parameters['radius_p'], semimajoraxis = parameters['semimajoraxis'], inclination = parameters['inclination'])
        model_parameters = LyA.ModelParameters(c_s_planet = parameters['c_s_planet'], mdot_planet = parameters['mdot_planet'], v_stellar_wind = parameters['v_stellar_wind'], mdot_star = parameters['mdot_star'], T_stellar_wind = parameters['T_stellar_wind'], L_EUV = parameters['L_EUV'], angle = parameters['angle'])

        #make_density_structure
        rho_struc = config.make_rho_struc(parameters)

        #evaluate prior
        logP = evaluate_log_prior(mcmc_log_parameters)

        if logP == -np.inf:
            return -np.inf

        else:
            try:
                tail_solution_cartesian = ttc.trajectory_solution_cartesian(star, planet, model_parameters, rho_struc)
            except ValueError:
                logging.exception(f"the parameters are {str(model_parameters)}")
                return -np.inf
            else:
                if tail_solution_cartesian.success == False: #if solution no found assume parameters unreasonable
                    logging.warning(f"failed ODE: the parameters are {str(model_parameters)}")
                    return -np.inf
                elif tail_solution_cartesian.t_events[1].size: #check if stopped by epicycle event
                    logging.warning(f"failed epicycle the parameters are {str(model_parameters)}")
                    return -np.inf
                else:
                    tail = ttc.trajectory_solution_polar(star, planet, model_parameters, rho_struc)
                    phase, intensity = do_transit(tail, phasegrid, wgrid, rho_struc, parameters['inclination'])
                    t, _, transit_data['w'], transit_data['we'], scaled_data, scaled_err, sim_data = simulate_spectra(1 - intensity)
                    logL = np.sum(-(1/2) * ((data_intensity - sim_data)**2 / data_err**2 + np.log(2 * np.pi * data_err**2)))
                    logPosterior = logL + logP

                    return logPosterior


    return evaluate_posterior
