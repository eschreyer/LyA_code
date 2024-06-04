import trajectory_tail_cartesian_new as ttc
import do_transit_new as dt
import do_transit_hill as dth
import numpy as np
import constants_new as const
import LyA_transit_datatypes_new as LyA
import logging
import Parker_wind_planet_new as pw
import scipy.interpolate as sp_int

#make logger

logger = logging.getLogger(__name__)



class PosteriorMaker():
    def __init__(self, constant_parameters_star, constant_parameters_planet, mcmc_parameters_key_list, evaluate_log_prior, configuration_parameters, transit_parameters, fit_package, logL_fnct, is_mlr_ratio = False):
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
        self.mcmc_parameters_key_list = mcmc_parameters_key_list
        self.evaluate_log_prior = evaluate_log_prior
        self.configuration_parameters = configuration_parameters
        self.fit_package = fit_package
        self.logL_fnct = logL_fnct
        self.do_transit = [dt.make_transit_tools(constant_parameters_star['radius_s'], n_star_cells = tp['n_star_cells'], n_z_cells = tp['n_z_cells'])[0] for tp in transit_parameters]
        self.do_transit_hill = [dth.make_transit_tools_hill_and_ena(constant_parameters_star['radius_s'], n_star_cells = 15) for tp in transit_parameters]
        self.is_mlr_ratio = is_mlr_ratio

    def partition_mcmc_log_parameters(self, mcmc_log_parameters):

        log_parameters = []
        for key_list in self.mcmc_parameters_key_list:
            log_parameters_single = {}
            for k, k1 in zip(key_list[0], key_list[1]):
                if isinstance(k, tuple):
                    log_parameters_single[k1] = tuple(mcmc_log_parameters[kk] for kk in k)
                else:
                    log_parameters_single[k1] = mcmc_log_parameters[k]
            log_parameters.append(log_parameters_single)
        #log_parameters = [{k1:mcmc_log_parameters[k] for k,k1 in zip(key_list[0], key_list[1])} for key_list in self.mcmc_parameters_key_list]

        if self.is_mlr_ratio:

            mlr_ratio = (self.constant_parameters_planet[0]['semimajoraxis']**2 * log_parameters[0]['mass_p'] * self.constant_parameters_planet[1]['radius_p']**3) / (self.constant_parameters_planet[1]['semimajoraxis']**2 * log_parameters[1]['mass_p'] * self.constant_parameters_planet[0]['radius_p']**3)

            if mlr_ratio < 0:
                logger.info(f"the parameters are {str(mcmc_log_parameters)}")

            log_parameters[1]['mdot_planet'] += np.log10(mlr_ratio)


        return log_parameters

    def convert_to_linspace(self, dic):
        new_dict = {}
        for key in dic:
            if key == 'angle' or key == 'inclination' or key == 'mass_p' or key == 'o_frac' or key == 'scalefacs':
                new_dict[key] = dic[key]
            else:
                new_dict[key] = 10**dic[key]
        return new_dict


    def evaluate_logL(self, log_parameters, constant_parameters, configuration_parameters, do_transit, do_transit_hill, fit_package, logL_fnct):

        phasegrid = fit_package.tgrid * np.sqrt(constant_parameters['mass_s'] * const.G / constant_parameters['semimajoraxis']**3) * 3600 + np.pi/2
        wgrid = const.c / (fit_package.wgrid * 1e-8) #note fit package wgrid is in wavelength in angstroms and i want to convert to cgs

        parameters = {**constant_parameters, **self.convert_to_linspace(log_parameters)}
        star = LyA.Star(mass = parameters['mass_s'], radius = parameters['radius_s'])
        planet = LyA.Planet(mass = parameters['mass_p'], radius = parameters['radius_p'], semimajoraxis = parameters['semimajoraxis'], inclination = parameters['inclination'])
        model_parameters = LyA.ModelParameters(c_s_planet = parameters['c_s_planet'], mdot_planet = parameters['mdot_planet'], v_stellar_wind = parameters['v_stellar_wind'], mdot_star = parameters['mdot_star'], T_stellar_wind = parameters['T_stellar_wind'], L_EUV = parameters['L_EUV'], angle = parameters['angle'])

        #make_density, stellar wind and photoionization rate
        rho_struc = configuration_parameters['make_rho_struc'](parameters)
        SW = configuration_parameters['make_stellar_wind'](parameters)
        photoionization_rate = configuration_parameters['make_photoionization_rate'](parameters)

        #make planetary wind functions for hill sphere

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

        #get planet keplerian velocity
        omega_p = np.sqrt(const.G * parameters['mass_s'] / parameters['semimajoraxis']**3)


        #make_ENA_structure if included
        if 'make_ENA' in configuration_parameters:
            ENA = configuration_parameters['make_ENA'](parameters)
        else:
            ENA = None

        try:
            tail_solution_cartesian = ttc.trajectory_solution_cartesian(star, planet, model_parameters, rho_struc, SW, photoionization_rate)
        except (ValueError, RuntimeWarning):
            print('hi')
            logger.exception(f"the parameters are {str(model_parameters)}")
            return -np.inf
        else:
            if not tail_solution_cartesian: #if solution no found assume parameters unreasonable
                return -np.inf
            elif tail_solution_cartesian.t_events[1].size: #check if stopped by epicycle event
                return -np.inf
            else:
                tail = ttc.trajectory_solution_polar(star, planet, model_parameters, rho_struc, SW, photoionization_rate)
                phase, model_intensity = do_transit(tail, phasegrid, wgrid, rho_struc, omega_p, parameters['inclination'], ENA = ENA)
                phase, model_intensity_hill = do_transit_hill(parameters, pw_functions, phasegrid, wgrid, parameters['inclination'], ENA = ENA)
                logL = logL_fnct(1 - model_intensity * model_intensity_hill, scalefacs = parameters['scalefacs'])
                if np.isnan(logL):
                    logger.error(f"the parameters are {str(model_parameters)}")
                    raise ValueError("this aint right")
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

        logL = [self.evaluate_logL(lp, {**self.constant_parameters_star, **cpp}, cfp, dt, dth, fp, logL_f) for lp, cpp, cfp, dt, dth, fp, logL_f in zip(log_parameters, self.constant_parameters_planet, self.configuration_parameters, self.do_transit, self.do_transit_hill, self.fit_package, self.logL_fnct)]

        return np.sum(np.array(log_prior) + np.array(logL))

    #stuff just for plotting

    def make_single_lightcurve(self, log_parameters, constant_parameters, configuration_parameters, do_transit, do_transit_hill, fit_package, logL_fnct, tgrid = None):

        if type(tgrid) == type(None):
            tgrid = fit_package.tgrid

        phasegrid = tgrid * np.sqrt(constant_parameters['mass_s'] * const.G / constant_parameters['semimajoraxis']**3) * 3600 + np.pi/2
        wgrid = const.c / (fit_package.wgrid * 1e-8) #note fit package wgrid is in wavelength in angstroms and i want to convert to cgs

        parameters = {**constant_parameters, **self.convert_to_linspace(log_parameters)}
        star = LyA.Star(mass = parameters['mass_s'], radius = parameters['radius_s'])
        planet = LyA.Planet(mass = parameters['mass_p'], radius = parameters['radius_p'], semimajoraxis = parameters['semimajoraxis'], inclination = parameters['inclination'])
        model_parameters = LyA.ModelParameters(c_s_planet = parameters['c_s_planet'], mdot_planet = parameters['mdot_planet'], v_stellar_wind = parameters['v_stellar_wind'], mdot_star = parameters['mdot_star'], T_stellar_wind = parameters['T_stellar_wind'], L_EUV = parameters['L_EUV'], angle = parameters['angle'])

        #make_density, stellar wind and photoionization rate
        rho_struc = configuration_parameters['make_rho_struc'](parameters)
        SW = configuration_parameters['make_stellar_wind'](parameters)
        photoionization_rate = configuration_parameters['make_photoionization_rate'](parameters)

        #make planetary wind functions for hill sphere

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

        #get planet keplerian velocity
        omega_p = np.sqrt(const.G * parameters['mass_s'] / parameters['semimajoraxis']**3)


        #make_ENA_structure if included
        if 'make_ENA' in configuration_parameters:
            ENA = configuration_parameters['make_ENA'](parameters)
        else:
            ENA = None


        tail = ttc.trajectory_solution_polar(star, planet, model_parameters, rho_struc, SW, photoionization_rate)
        phase, model_intensity = do_transit(tail, phasegrid, wgrid, rho_struc, omega_p, parameters['inclination'], ENA = ENA)
        phase, model_intensity_hill = do_transit_hill(parameters, pw_functions, phasegrid, wgrid, parameters['inclination'], ENA = ENA)
        transit = {'tgrid' : tgrid, 'model_intensity_hill' : model_intensity_hill, 'model_intensity' : model_intensity}

        return transit



    def make_lightcurves(self, mcmc_log_parameters, tgrids = [None, None]):

        log_parameters = self.partition_mcmc_log_parameters(mcmc_log_parameters)

        transits = [self.make_single_lightcurve(lp, {**self.constant_parameters_star, **cpp}, cfp, dt, dth, fp, logL_f, tg) for lp, cpp, cfp, dt, dth, fp, logL_f, tg in zip(log_parameters, self.constant_parameters_planet, self.configuration_parameters, self.do_transit, self.do_transit_hill, self.fit_package, self.logL_fnct, tgrids)]

        return transits
