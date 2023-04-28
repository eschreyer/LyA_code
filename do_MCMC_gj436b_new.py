import os
import logging
import sys
import emcee
import numpy as np
import multiprocess as mp
import posterior_gj436b_new as p
import constants_new as const
import argparse
import importlib


os.environ["OMP_NUM_THREADS"] = "1"

#parse command line arguments

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--backend", help="file that chain outputs will be written to")
parser.add_argument("-c", "--config", help="configuration file")
parser.add_argument("-r", "--restart", action="store_true", help="restart chain that has already been run")
parser.add_argument("-f", "--fitting", help="what type of fitting to perform, choose from: b: blue wing, f: flux weighted, bf: blue wing and flux weighted, if no input will fit full spectrum")
parser.add_argument("--ena", action="store_true", help="whether to include ENAs")
parser.add_argument("--hill_sphere", action="store_true", help="whether to include hill_sphere")
args = parser.parse_args()

#set up logging

if args.backend:
    logging.basicConfig(filename = args.backend[:-2] + 'log')


#import configuration file
config = importlib.import_module(args.config)


#table of parameters

"""table of parameters : {'mass_s', 'radius_s' STAR PARAM
                          'mass_p', 'radius_p', 'semimajoraxis', 'inclination' PLANET PARAM
                          'c_s_planet', 'mdot_planet' , v_stellar_wind', 'mdot_star', 'T_stellar_wind', 'L_EUV', 'angle' MODEL PARAM
                          'u_ENA', 'L_mix'} ENA param """


#constant parameters

constant_parameters = config.constant_parameters

assert config.is_ENA_on == args.ena

if args.ena == True:
    configuration_parameters = {'make_rho_struc' : config.make_rho_struc, 'make_stellar_wind' : config.make_stellar_wind, 'make_photoionization_rate' : config.make_photoionization_rate, 'make_ENA' : config.make_ENA}
else:
    configuration_parameters = {'make_rho_struc' : config.make_rho_struc, 'make_stellar_wind' : config.make_stellar_wind, 'make_photoionization_rate' : config.make_photoionization_rate}



evaluate_log_prior = config.evaluate_log_prior


def main(target_file, restart = False, only_blue = False, weight_fluxes = False):

    #make posterior function
    evaluate_posterior = p.make_log_posterior_fn(constant_parameters, evaluate_log_prior, configuration_parameters, config.tgrid, config.transit_rng, hill_sphere = args.hill_sphere, only_blue = only_blue, weight_fluxes = weight_fluxes)

    #chain params
    n_walkers = 100
    n_iterations = 20000

    #sampled parameters and initial values
    sampled_parameters = config.sampled_parameters #['c_s_planet', 'mdot_planet', 'v_stellar_wind', 'mdot_star', 'L_EUV', 'angle', 'inclination']
    n_dim = len(sampled_parameters)

    if restart == True:
    #if chain already run and we just want to restart it

        backend = emcee.backends.HDFBackend(target_file)

        with mp.Pool() as pool:
            sampler = emcee.EnsembleSampler(n_walkers, n_dim, evaluate_posterior, pool = pool, backend = backend, parameter_names = sampled_parameters) # initialise sampler object
            sampler.run_mcmc(None, n_iterations, progress=True)  # start the chain!

    else:
    #new chain

        sampled_parameters_guess = config.sampled_parameter_guess  #np.array([6, 9.3, 7.4, 12, 27.2, (3/4)*np.pi, 1.51])

        #set random seed so can reproduce initial values
        rng = np.random.default_rng(seed = config.random_seed_init_guess)
        sampled_parameters_init = np.tile(sampled_parameters_guess, (n_walkers, 1)) * (1 + rng.uniform(-1e-2, 1e-2, (n_walkers, n_dim)))

        #backend initilization

        backend = emcee.backends.HDFBackend(target_file)
        backend.reset(n_walkers, n_dim)


        #sampler initilazation and run
        with mp.Pool() as pool:
            sampler = emcee.EnsembleSampler(n_walkers, n_dim, evaluate_posterior, pool = pool, backend = backend, parameter_names = sampled_parameters) # initialise sampler object
            random_seed = config.random_seed_chain
            sampler._random.seed(random_seed)  #fix chain random seed so that it can be reproduced
            sampler.run_mcmc(sampled_parameters_init, n_iterations, progress=True)  # start the chain!


if __name__ == "__main__":
    if args.fitting == None:
        main(args.backend, restart = args.restart)
    elif args.fitting == 'b':
        main(args.backend, restart = args.restart, only_blue = True)
    elif args.fitting == 'f':
        main(args.backend, restart = args.restart, weight_fluxes = True)
    elif args.fitting == 'bf':
        main(args.backend, restart = args.restart, only_blue = True, weight_fluxes = True)
    else:
        logging.exception(f"{args.fitting} is not an allowed argument")
