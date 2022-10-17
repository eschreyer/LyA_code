import os
import logging
import sys
import emcee
import numpy as np
import multiprocess as mp
import posterior_gj436b_new as p
import constants_new as const

os.environ["OMP_NUM_THREADS"] = "1"

#table of parameters

"""table of parameters : {'mass_s', 'radius_s' STAR PARAM
                          'mass_p', 'radius_p', 'semimajoraxis', 'inclination' PLANET PARAM
                          'c_s_planet', 'mdot_planet' , v_stellar_wind', 'mdot_star', 'T_stellar_wind', 'L_EUV', 'angle'} MODEL PARAM"""


#constant parameters

constant_parameters = {'mass_s' : 0.45*const.m_sun, 'radius_s' : 0.415*const.r_sun,
                       'mass_p' : 0.07*const.m_jupiter, 'radius_p' : 0.35 * const.r_jupiter, 'semimajoraxis' : 4.35e11, 'inclination' : 1.51,
                       'T_stellar_wind' : 1e6}

#make posterior function

evaluate_posterior = p.make_log_posterior_fn(constant_parameters)

def main(target_file):


    #chain params
    n_walkers = 100
    n_iterations = 10000

    #sampled parameters and initial values
    n_dim = 6
    sampled_parameters = ['c_s_planet', 'mdot_planet', 'v_stellar_wind', 'mdot_star', 'L_EUV', 'angle']

    sampled_parameters_guess = np.array([6, 9.7, 7.4, 12, 26.9, (3/4)*np.pi])

    #set random seed so can reproduce initial values
    rng = np.random.default_rng(seed = 2)
    sampled_parameters_init = np.tile(sampled_parameters_guess, (n_walkers, 1)) * (1 + rng.uniform(-1e-2, 1e-2, (n_walkers, n_dim)))

    #backend initilization

    backend = emcee.backends.HDFBackend(target_file)
    backend.reset(n_walkers, n_dim)


    #sampler initilazation and run
    with mp.Pool() as pool:
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, evaluate_posterior, pool = pool, backend = backend, parameter_names = sampled_parameters) # initialise sampler object
        random_seed = 1
        sampler._random.seed(random_seed)  #fix chain random seed so that it can be reproduced
        sampler.run_mcmc(sampled_parameters_init, n_iterations, progress=True)  # start the chain!

if __name__ == '__main__':
    main(sys.argv[1])
