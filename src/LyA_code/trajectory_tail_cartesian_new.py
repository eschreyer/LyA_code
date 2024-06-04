import numpy as np
import scipy.integrate as sp_int
import constants_new as const
import tail_object_holders_new as to
import Parker_wind_planet_new as pw
from functools import partial


"""
Forces acting on the tail
------------------------------------------
"""

def G_Force(r, star):
    """
    returns float
    """
    return const.G * star.mass / r**2

def Centrifugal_Force(r, star, planet):
    """
    returns float
    """

    omega = np.sqrt(const.G*star.mass/planet.semimajoraxis**3)
    return omega**2 * r

def Ram_Pressure_Force(u_x, u_y, x, y, star, planet, model_parameters, height, SW):
    """
    approximate force through the tail due to the stellar wind ram pressure

    returns float
    """
    r = np.sqrt(x**2 + y**2)

    phi = np.arctan2(y,x)

    u_s = np.sqrt(u_x**2 + u_y**2)

    ram_pressure_stellar_wind = SW.ram_pressure(r)

    if (u_x*np.cos(phi) + u_y*np.sin(phi)) < model_parameters.v_stellar_wind:

        return (2*height*u_s*ram_pressure_stellar_wind/(model_parameters.mdot_planet))*((model_parameters.v_stellar_wind - (u_x*np.cos(phi) + u_y*np.sin(phi)))**2/model_parameters.v_stellar_wind**2)*np.abs((-u_y * np.cos(phi) + u_x * np.sin(phi)))/u_s

    else:

        return 0

"""
Solving the tail equations
--------------------------------------------------
"""
def trajectory_equations(s, w, star, planet, model_parameters, rho_struc, SW, photoionization_rate):
    """
    Outputs the equations that solve the geometry (position and velocity) and ionization fraction of the tail as a
    function of the distance down the streamline

    Parameters
    ----------------------------
    s : the distance down the streamline

    w : array of [u_x, u_y, x, y, neutral], where

    planet:

    model_parameters:

    Returns
    ----------------------------
    A system of first order ODE's contained in an array which can be passed to solve_ivp

    """
    #w = [u_x, u_y, x, y, neutral_fraction]
    r = np.sqrt(w[2]**2 + w[3]**2)
    phi = np.arctan2(w[3] , w[2])
    u_s = np.sqrt(w[0]**2 + w[1]**2)
    [height1], [depth1] = rho_struc.get_height_and_depth(np.array([[w[2], w[3], 0]]), np.array([[w[0], w[1], 0]]))
    height  = min(height1, planet.semimajoraxis*(planet.mass/(3*star.mass))**(1/3) + s) #smooth height at start
    depth = min(depth1, planet.semimajoraxis*(planet.mass/(3*star.mass))**(1/3) + s) #smooth depth at start



    u_x_eq = (1/u_s)*((Ram_Pressure_Force(w[0], w[1], w[2], w[3], star, planet, model_parameters, height, SW) - G_Force(r, star) + Centrifugal_Force(r, star, planet))*np.cos(phi) + 2*np.sqrt(const.G*star.mass/planet.semimajoraxis**3)*w[1])
    u_y_eq = (1/u_s)*((Ram_Pressure_Force(w[0], w[1], w[2], w[3], star, planet, model_parameters, height, SW) - G_Force(r, star) + Centrifugal_Force(r, star, planet))*np.sin(phi) - 2*np.sqrt(const.G*star.mass/planet.semimajoraxis**3)*w[0])
    x_eq = w[0]/u_s
    y_eq = w[1]/u_s
    neutral_fraction_eq = - w[4] * photoionization_rate(r) / u_s + (model_parameters.mdot_planet / (np.pi * height * depth * const.m_proton * u_s**2)) * (1 - w[4])**2 * const.recombination_rate_caseA

    return [u_x_eq, u_y_eq, x_eq, y_eq, neutral_fraction_eq]



def trajectory_solution_cartesian(star, planet, model_parameters, rho_struc, SW, photoionization_rate):

    """
    Uses

    Solves the geometry (position and velocity) in cartesian coordinates and ionization fraction of
    the tail as a function of the distance down the streamline

    Returns
    ------------------

    sol.t: s

    sol.y:

    """

    #initial velocity, neutral fraction and temperature conditions in tail
    #maybe remove temperature --- not neccessary

    velocity_init, neutral_frac_init, temperature_init = pw.planetary_wind(star, planet, model_parameters, photoionization_rate)
    wind_angle = model_parameters.angle
    hill_sphere_radius = planet.semimajoraxis*(planet.mass/(3*star.mass))**(1/3)


    trajectory_eq = partial(trajectory_equations, star = star, planet = planet, model_parameters = model_parameters, rho_struc = rho_struc, SW = SW, photoionization_rate = photoionization_rate)

    #for now just stop the tail if it moves above y = 0
    def stop_tail(s, w):
        return np.arctan2(w[3], w[2])
    stop_tail.terminal = True
    stop_tail.direction = 1

    #stop tail if the angular velocity is positive (this normally happens in the case that there are epicycles)
    def stop_tail_w(s, w):
        return -w[0]*w[3] + w[1]*w[2]
    stop_tail_w.terminal = True
    stop_tail_w.direction = 1


    #stop tail once it becomes unnecessary for observations
    def stop_tail_o(s, w):
        theta = np.arctan2(w[3], w[2])
        t_length = 34 * 3600 #seconds
        omega = np.sqrt(const.G * star.mass / planet.semimajoraxis**3)
        return omega * t_length + theta
    stop_tail_o.terminal = True



    #solver
    sol = sp_int.solve_ivp(trajectory_eq, [0, 10*planet.semimajoraxis], [velocity_init*np.sin(wind_angle), velocity_init*np.cos(wind_angle), planet.semimajoraxis + hill_sphere_radius*np.sin(wind_angle), hill_sphere_radius*np.cos(wind_angle), neutral_frac_init], dense_output = 'True', events = [stop_tail, stop_tail_w, stop_tail_o], rtol = 1e-13, atol = 1e-14, method = 'BDF')

    return sol

def trajectory_solution_polar(star, planet, model_parameters, rho_struc, SW, photoionization_rate):

    """
    Convert the into polar coordinates in the orbital plane. We take theta = 0 to be inline with the
    postive x-axis and measure it anticlockwise
    """

    sol_cartesian = trajectory_solution_cartesian(star, planet, model_parameters, rho_struc, SW, photoionization_rate)

    r_sol = np.sqrt(sol_cartesian.y[2]**2 + sol_cartesian.y[3]**2)

    phi_sol = np.arctan2(sol_cartesian.y[3], sol_cartesian.y[2])

    U_r_sol = sol_cartesian.y[0]*np.cos(phi_sol) + sol_cartesian.y[1]*np.sin(phi_sol)

    U_phi_sol = (-sol_cartesian.y[0]*np.sin(phi_sol) + sol_cartesian.y[1]*np.cos(phi_sol))

    return to.TailOrbitalPlanePolarArray(t = sol_cartesian.t, y = [U_r_sol, r_sol, U_phi_sol, phi_sol, sol_cartesian.y[4]])
