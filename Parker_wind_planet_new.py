import numpy as np
import scipy.special as sp_special
from functools import partial
import scipy.integrate as sp_int
import constants_new as const
import config as config

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

    return np.where(r < r_c, np.sqrt(-model_parameters.c_s_planet**2 * np.real(sp_special.lambertw(-D,0))), np.sqrt(-model_parameters.c_s_planet**2 * np.real(sp_special.lambertw(-D,-1))))


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


def ionisation_eq(r, N, star, planet, model_parameters):
    #RHS of ionization equation

    u = velocity_planetary_wind(r, star, planet, model_parameters) #velocity of outflow at r

    if u < 5e4: #gas moving below that velocity will probably be far below the tau = 1 surface, so we assume it is not ionised at all
        return 0
    else:
        return -(N * config.photoionization_rate(planet.semimajoraxis - r, model_parameters.L_EUV)  / u) + (model_parameters.mdot_planet * (1 - N)**2 * const.recombination_rate_caseA)/(4 * np.pi * r**2 * u**2)



def neutral_frac_planetary_wind(star, planet, model_parameters):

    hill_sphere_radius = planet.semimajoraxis * (planet.mass / (3 * star.mass))**(1/3)

    N_init = 1 #initial neutral fraction set to 1 at planet radius

    ionisation_eq_eval = partial(ionisation_eq, star = star, planet = planet, model_parameters = model_parameters)

    sol = sp_int.solve_ivp(ionisation_eq_eval, [planet.radius , hill_sphere_radius], [N_init], dense_output = 'True', rtol = 1e-13, atol = 1e-13, method = 'LSODA')

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

def planetary_wind(star, planet, model_parameters):
    """
    returns: velocity (float), neutral fraction (float) and temperature (float) at hill radius
    """

    hill_sphere_radius = planet.semimajoraxis * (planet.mass / (3 * star.mass))**(1/3)

    velocity_at_hill_radius = velocity_planetary_wind(hill_sphere_radius, star, planet, model_parameters)

    neutral_fraction_at_hill_radius = neutral_frac_planetary_wind(star, planet, model_parameters).y[0][-1]

    temperature_at_hill_radius = temperature(neutral_fraction_at_hill_radius, model_parameters)

    return velocity_at_hill_radius, neutral_fraction_at_hill_radius, temperature_at_hill_radius
