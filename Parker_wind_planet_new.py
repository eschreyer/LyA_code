import numpy as np
import scipy.special as sp_special
from functools import partial
import scipy.integrate as sp_int
import constants_new as const
import logging

logger = logging.getLogger(__name__)


"""
Cubic Root solver
-------------------------------------------
"""


def cardano_formula(p,q):
    """
    returns the real root of the cubic equation of the form
    x^3 + px + q = 0 for the case where the discriminant
    4p^3 + 27q^2 > 0

    returns float
    """
    C = (-q/2 + (q**2/4 + p**3/27)**(1/2))
    if C >= 0:
        return C**(1/3) - p/(3 * C**(1/3))
    elif C < 0:
        pos_C = np.abs(C)
        return -pos_C**(1/3) - p/(3 * -pos_C**(1/3))

"""
Velocity and Density of Outflow
----------------------------------------------
"""

def velocity_planetary_wind(r, star, planet, model_parameters):
    """
    calculates the velocity of the planetary wind using the Lambert W function at distance r from planet centre
    ### add test to see if critical point is less than planet radius
    returns float
    """

    R_L = planet.semimajoraxis * (planet.mass / (3 * star.mass))**(1/3)    #roche radius
    r_a = const.G * planet.mass / (2 * model_parameters.c_s_planet**2)    #critical point if stellar gravity ignored
    r_c = cardano_formula(R_L**3/r_a, -R_L**3)                            #critical point
    D = (r/r_c)**-4 * np.exp(4 * r_a * (1 / r_c - 1 / r) + (2 * r_a / R_L**3) * (r_c**2 - r**2) - 1)

    u = np.where(r < r_c, np.sqrt(-model_parameters.c_s_planet**2 * np.real(sp_special.lambertw(-D,0))), np.sqrt(-model_parameters.c_s_planet**2 * np.real(sp_special.lambertw(-D,-1))))

    if np.isnan(u).any():
        logger.warning('watch out nan velocity, here are the parameters' + f"the parameters are {str(model_parameters)}")

    return np.where(np.isnan(u), model_parameters.c_s_planet, u)


    """if r < r_c:
        return np.sqrt(-model_parameters.c_s_planet**2 * np.real(sp_special.lambertw(-D,0)))
    elif r >= r_c:
        return np.sqrt(-model_parameters.c_s_planet**2 * np.real(sp_special.lambertw(-D,-1)))"""


def density_planetary_wind(r, star, planet, model_parameters):
    """
    returns float
    """
    density_wind = model_parameters.mdot_planet / (4 * np.pi * r**2 * velocity_planetary_wind(r, star, planet, model_parameters))
    return density_wind

"""
Ionisation and Temperature of Outflow (Requires Velocity and Masslossrate)
--------------------------------------------------------------------------
"""


def ionisation_eq_w_tau(r, N, star, planet, model_parameters, photoionization_rate):

    #calculate tau

    u = velocity_planetary_wind(r, star, planet, model_parameters) #velocity of outflow at r

    hill_sphere_radius = planet.semimajoraxis * (planet.mass / (3 * star.mass))**(1/3)

    def dtau(r_1, star, planet, model_parameters):

        return (density_planetary_wind(r_1, star, planet, model_parameters) / const.m_proton) * const.HI_crosssection * (13.6 / 20)**3

    tau = sp_int.quad(dtau, r, hill_sphere_radius, args=(star, planet, model_parameters))

    return -(N * photoionization_rate(planet.semimajoraxis - r) * np.exp(-tau[0])  / u) + (model_parameters.mdot_planet * (1 - N)**2 * const.recombination_rate_caseA)/(4 * np.pi * r**2 * u**2)


def neutral_frac_planetary_wind(star, planet, model_parameters, photoionization_rate, tau = False):

    hill_sphere_radius = planet.semimajoraxis * (planet.mass / (3 * star.mass))**(1/3)
    r_a = const.G * planet.mass / (2 * model_parameters.c_s_planet**2)    #critical point if stellar gravity ignored
    r_c = cardano_formula(hill_sphere_radius**3/r_a, -hill_sphere_radius**3)

    N_init = 1 #initial neutral fraction set to 1 at planet radius

    if tau == True:

        ionisation_eq_eval = partial(ionisation_eq_w_tau, star = star, planet = planet, model_parameters = model_parameters, photoionization_rate = photoionization_rate)

    else:

        ionisation_eq_eval = partial(ionisation_eq, star = star, planet = planet, model_parameters = model_parameters, photoionization_rate = photoionization_rate)

    if 1.1 * r_c < hill_sphere_radius:

        sol = sp_int.solve_ivp(ionisation_eq_eval, [1.1*r_c, hill_sphere_radius], [N_init], dense_output = 'False', rtol = 1e-13, atol = 1e-13, method = 'LSODA')

    else:

        sol = sp_int.solve_ivp(ionisation_eq_eval, [hill_sphere_radius / 4, hill_sphere_radius], [N_init], dense_output = 'False', rtol = 1e-13, atol = 1e-13, method = 'LSODA')

    return sol

def temperature(N, model_parameters):
    """
    Parameters
    ------------
    N : neutral fraction

    model_parameters : c_s(sound speed)

    """
    mmw = const.m_proton / (2 - N)
    return (model_parameters.c_s_planet**2 * mmw) / const.k_b


"""
Outputs of 1D hill sphere model to tail model
--------------------------------------------
"""

def planetary_wind(star, planet, model_parameters, photoionization_rate):
    """
    returns: velocity (float), neutral fraction (float) and temperature (float) at hill radius
    """

    hill_sphere_radius = planet.semimajoraxis * (planet.mass / (3 * star.mass))**(1/3)

    velocity_at_hill_radius = velocity_planetary_wind(hill_sphere_radius, star, planet, model_parameters)

    neutral_fraction_at_hill_radius = neutral_frac_planetary_wind(star, planet, model_parameters, photoionization_rate, tau = True).y[0][-1]

    temperature_at_hill_radius = temperature(neutral_fraction_at_hill_radius, model_parameters)

    return velocity_at_hill_radius, neutral_fraction_at_hill_radius, temperature_at_hill_radius
