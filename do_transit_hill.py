import Parker_wind_planet_new as pw
import numpy as np
import scipy.integrate as sp_int
import xsection_new as xs
import constants_new as const
import LyA_transit_datatypes_new as LyA
import star_grid_new as sg
import change_coords_of_tail_trajectory_new as cc
import do_transit_hill_ena as dthe
import xsection_new as xs

#In this file we do the transit of the hll sphere


def tau_los(y_c, planet_velocity, pw_functions, planet_radius, hill_sphere_radius, wgrid, atomic_xsection = xs.LyA_xsection):
    """
    This function calculates the optical depth of a ray emanating from the star and going through the hill sphere of the planet
    -----------------------------------------------------
    y_c : the closest distance of the ray from planet: since our rays are at (x_0, y_0, z) and planet is at (x_1, y_1, z_1), the closest point is
    at (x_0, y_0, z_1)

    pw_functions: functions of the density, velocity, neutral fraction along a ray emanating from the star with closest point of approach y_c. z is
    the coordinate down the ray. z = 0 at closest approach to centre of planet.

    wgrid: frequency grid

    """

    if y_c > hill_sphere_radius:
    #doesn't intersect with Hill sphere
        return np.zeros(len(wgrid))

    elif y_c < planet_radius:
    #intersects with planet
        return np.inf * np.ones(len(wgrid))

    else:

        z_grid = np.linspace(-np.sqrt(hill_sphere_radius**2 - y_c**2), np.sqrt(hill_sphere_radius**2 - y_c**2), 15)

        density_grid = pw_functions['density'](z_grid, y_c)

        neutral_fraction_grid = pw_functions['neutral_fraction'](z_grid, y_c)

        z_velocity_grid = pw_functions['z_velocity'](z_grid, y_c) + planet_velocity[2]

        w_mesh, z_velocity_mesh = np.meshgrid(wgrid, z_velocity_grid)
        #write as o_grid to speed up but beware indices are swapped

        #remember everything is not at 10**4K
        xsection_grid = atomic_xsection(w_mesh, z_velocity_mesh, 10**4) #shape (k1, l) , #where l is the number of wavelength points

        dtau_grid = xs.d_tau(np.reshape(density_grid * neutral_fraction_grid, (len(density_grid), 1)), xsection_grid, const.m_proton, z_grid[1] - z_grid[0])

        return np.sum(dtau_grid, axis = 0)


def vectorized_tau_los(y_c, planet_velocity, pw_functions, planet_radius, hill_sphere_radius, wgrid, atomic_xsection = xs.LyA_xsection):

    array = np.zeros((len(y_c), len(wgrid)))
    y_c = np.asarray(y_c)

    for i,y in enumerate(y_c):

        tau = tau_los(y, planet_velocity, pw_functions, planet_radius, hill_sphere_radius, wgrid, atomic_xsection)
        array[i, :] = tau

    return array


def get_tau_at_phase_hill(star_grid, planet_position, planet_velocity, pw_functions, planet_radius, hill_sphere_radius, wgrid, atomic_xsection = xs.LyA_xsection):

    if planet_position[2] < 0:

        return np.zeros((len(star_grid), len(wgrid)))

    else:

        y_c = np.sqrt((planet_position[0] - star_grid[:, 0])**2 + (planet_position[1] - star_grid[:, 1])**2)

        tau_grid = vectorized_tau_los(y_c, planet_velocity, pw_functions, planet_radius, hill_sphere_radius, wgrid, atomic_xsection)

        return tau_grid


def make_transit_tools_hill(star_radius, n_star_cells, n_z_cells = None):

    star_grid, areas_array = sg.make_grid_cartesian2(star_radius, n_star_cells)

    def do_transit_hill(parameters, pw_functions, phase_grid, wgrid, inclination, atomic_xsection = xs.LyA_xsection):

        intensity_array = np.empty((len(phase_grid), len(wgrid)))
        hill_sphere_radius = parameters['semimajoraxis']*(parameters['mass_p']/(3*parameters['mass_s']))**(1/3)

        for index, phase in enumerate(phase_grid):
            planet_position = cc.convert_point_on_orbitalplane_to_transitcoords(parameters['semimajoraxis'], phase, inclination)
            planet_velocity = cc.convert_vector_on_orbitalplane_to_transitcoords(0, 0, parameters['semimajoraxis'], phase, inclination)
            print(planet_velocity)
            tau_grid = get_tau_at_phase_hill(star_grid, planet_position, planet_velocity, pw_functions, parameters['radius_p'], hill_sphere_radius, wgrid, atomic_xsection)
            intensity = np.einsum('i, ij -> j', areas_array, np.exp(-tau_grid)) / (np.pi * star_radius**2)
            intensity_array[index, :] = intensity

        return phase_grid, intensity_array

    def do_transit_hill_tau(parameters, pw_functions, phase_grid, wgrid, inclination):

        tau_array = np.empty((len(phase_grid), len(star_grid), len(wgrid)))
        hill_sphere_radius = parameters['semimajoraxis']*(parameters['mass_p']/(3*parameters['mass_s']))**(1/3)

        for index, phase in enumerate(phase_grid):
            planet_position = cc.convert_point_on_orbitalplane_to_transitcoords(parameters['semimajoraxis'], phase, inclination)
            planet_velocity = cc.convert_vector_on_orbitalplane_to_transitcoords(0, 0, parameters['semimajoraxis'], phase, inclination)
            tau_grid = get_tau_at_phase_hill(star_grid, planet_position, planet_velocity, pw_functions, parameters['radius_p'], hill_sphere_radius, wgrid)
            tau_array[index, :, :] = tau_grid

        return phase_grid, tau_array

    return do_transit_hill, do_transit_hill_tau


def make_transit_tools_hill_and_ena(star_radius, n_star_cells, n_z_cells = None):

    do_transit_hill, _ = make_transit_tools_hill(star_radius, n_star_cells, n_z_cells = None)

    do_transit_hill_ena, _ = dthe.make_transit_tools_hill_ena(star_radius, n_star_cells, n_z_cells = None)

    def do_transit_hill_and_ena(parameters, pw_functions, phase_grid, wgrid, inclination, atomic_xsection = xs.LyA_xsection, ENA = None):

        phase, intensity_hill = do_transit_hill(parameters, pw_functions, phase_grid, wgrid, inclination, atomic_xsection)

        if ENA:

            phase, intensity_ena = do_transit_hill_ena(parameters, pw_functions, phase_grid, wgrid, inclination, ENA)

            return phase, intensity_hill * intensity_ena

        else:

            return phase, intensity_hill

    return do_transit_hill_and_ena
