import trajectory_tail_cartesian_new as ttc
import do_transit_new as dt
import numpy as np
import constants_new as const
import LyA_transit_datatypes_new as LyA
import matplotlib.pyplot as plt
import xsection_new as xs
import constants_new as const
import logging

class PosteriorMaker():
    def __init__(self, constant_parameters_star, constant_parameters_planet, evaluate_log_prior, configuration_parameters, transit_parameters, fit_package, logL_fnct):
        """

        Parameters
        ----------
        constant_parameters_star : star_parameters
        constant_parameters_planet : planet_parameters (list of planets)
        evaluate_prior: list of evaluate prior fncts, 1 for each planet
        configuration_parameters : list of dictionaries, 1 for each planet
        do_transit : list of do_transit_fncts, 1 for each planet
        fit_package : list of packages used to fit data
        logL_fnct : list of log_likelihood_fncts

        Returns
        -------
        evaluate posterior : evaluate posterior based on
        """

        self.constant_parameters_star = constant_parameters_star
        self.constant_parameters_planet = constant_parameters_planet
        self.evaluate_log_prior = evaluate_log_prior
        self.configuration_parameters = configuration_parameters
        self.fit_package = fit_package
        self.logL_fnct = logL_fnct
        self.do_transit = [dt.make_transit_tools(constant_parameters_star['radius_s'], n_star_cells = tp['n_star_cells'], n_z_cells = tp['n_z_cells']) for tp in transit_parameters]

    def partition_mcmc_log_parameters(self, mcmc_log_parameters):

        planetb_key_list = ['c_s_planetb', 'mdot_planetb', 'v_stellar_wind', 'mdot_star', 'L_EUV', 'angleb', 'inclinationb']
        planetc_key_list = ['c_s_planetc', 'mdot_planetc', 'v_stellar_wind', 'mdot_star', 'L_EUV', 'anglec', 'inclinationc']
        key_list = ['c_s_planet', 'mdot_planet', 'v_stellar_wind', 'mdot_star', 'L_EUV', 'angle', 'inclination']

        log_parameters_planetb = {k1:mcmc_log_parameters[k] for k1,k in zip(key_list,planetb_key_list)}
        log_parameters_planetc = {k1:mcmc_log_parameters[k] for k1,k in zip(key_list,planetc_key_list)}

        log_parameters = [log_parameters_planetb, log_parameters_planetc]

        return log_parameters

    def convert_to_linspace(self, dic):
        new_dict = {}
        for key in dic:
            if key == 'angle' or key == 'inclination':
                new_dict[key] = dic[key]
            else:
                new_dict[key] = 10**dic[key]
        return new_dict


    def evaluate_logL(self, log_parameters, constant_parameters, configuration_parameters, do_transit, fit_package, logL_fnct):

        phasegrid = fit_package.tgrid * np.sqrt(constant_parameters['mass_s'] * const.G / constant_parameters['semimajoraxis']**3) * 3600 + np.pi/2
        wgrid = const.c / fit_package.wgrid #note fit package wgrid is wavelength by mine is frequency


        parameters = {**constant_parameters, **self.convert_to_linspace(log_parameters)}
        star = LyA.Star(mass = parameters['mass_s'], radius = parameters['radius_s'])
        planet = LyA.Planet(mass = parameters['mass_p'], radius = parameters['radius_p'], semimajoraxis = parameters['semimajoraxis'], inclination = parameters['inclination'])
        model_parameters = LyA.ModelParameters(c_s_planet = parameters['c_s_planet'], mdot_planet = parameters['mdot_planet'], v_stellar_wind = parameters['v_stellar_wind'], mdot_star = parameters['mdot_star'], T_stellar_wind = parameters['T_stellar_wind'], L_EUV = parameters['L_EUV'], angle = parameters['angle'])

        #make_density, stellar wind and photoionization rate
        rho_struc = configuration_parameters['make_rho_struc'](parameters)
        SW = configuration_parameters['make_stellar_wind'](parameters)
        photoionization_rate = configuration_parameters['make_photoionization_rate'](parameters)

        #make_ENA_structure if included
        if 'make_ENA' in configuration_parameters:
            ENA = configuration_parameters['make_ENA'](parameters)
        else:
            ENA = None

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
                phase, model_intensity = do_transit(tail, phasegrid, wgrid, rho_struc, parameters['inclination'], ENA = ENA)
                logL = logL_fnct(1 - model_intensity)
                return logL


        return evaluate_logL


    def evaluate_posterior(self, mcmc_log_parameters):

        #first partition mcmc log parameters

        log_parameters = self.partition_mcmc_log_parameters(mcmc_log_parameters)

        #now check priors

        log_prior = [elp(lp, {**self.constant_parameters_star, **cpp}) for elp, lp, cpp in zip(self.evaluate_log_prior, log_parameters, self.constant_parameters_planet)]

        #check if either prior is infinity

        if -np.inf in log_prior:
            return -np.inf

        #evaluate likelihood

        logL = [self.evaluate_logL(lp, {**self.constant_parameters_star, **cpp}, cfp, dt, fp, logL_f) for lp, cpp, cfp, dt, fp, logL_f in zip(log_parameters, self.constant_parameters_planet, self.configuration_parameters, self.do_transit, self.fit_package, self.logL_fnct)]

        return np.sum(np.array(log_prior) + np.array(logL))
