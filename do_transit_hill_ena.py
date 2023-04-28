import Parker_wind_planet_new as pw
import numpy as np
import scipy.integrate as sp_int
import xsection_new as xs
import constants_new as const
from star_grid_new import *
import matplotlib.pyplot as plt
import LyA_transit_datatypes_new as LyA
import star_grid_new as sg
import change_coords_of_tail_trajectory_new as cc

#in this file we do the ena transit of the hill sphere

def tau_los_ena(y_c, planet_velocity, pw_at_hill_sphere, ENA_at_hill_sphere, hill_sphere_radius, wgrid):

    ENA_radius = hill_sphere_radius * (1 + ENA_at_hill_sphere['L_mix'])

    if y_c > ENA_radius:

        return np.zeros(len(wgrid))

    else:

        density_ENA = ENA_at_hill_sphere['rho']

        velocity_ENA = planet_velocity[2] + ENA_at_hill_sphere['velocity']

        neutral_fraction_ENA = pw_at_hill_sphere['neutral_fraction']

        xsection_grid_ENA = xs.LyA_xsection(wgrid, velocity_ENA, ENA_at_hill_sphere['temperature'])

        if y_c > hill_sphere_radius:

            intersection_length = 2 * np.sqrt((hill_sphere_radius * (1 + ENA_at_hill_sphere['L_mix']))**2 - y_c**2)

        else:

            intersection_length = 2 * np.sqrt((hill_sphere_radius * (1 + ENA_at_hill_sphere['L_mix']))**2 - y_c**2) - 2 * np.sqrt((hill_sphere_radius**2 - y_c**2))

        dtau_grid_ENA = xs.d_tau(density_ENA * neutral_fraction_ENA, xsection_grid_ENA, const.m_proton, intersection_length)

        return dtau_grid_ENA



def vectorized_tau_los_ena(y_c, planet_velocity, pw_at_hill_sphere, ENA_at_hill_sphere, hill_sphere_radius, wgrid):

    array = np.zeros((len(y_c), len(wgrid)))
    y_c = np.asarray(y_c)

    for i,y in enumerate(y_c):

        tau = tau_los_ena(y, planet_velocity, pw_at_hill_sphere, ENA_at_hill_sphere, hill_sphere_radius, wgrid)
        array[i, :] = tau

    return array


def get_tau_at_phase_hill_ena(star_grid, planet_position, planet_velocity, pw_at_hill_sphere, ENA_at_hill_sphere, hill_sphere_radius, wgrid):

    if planet_position[2] < 0:

        return np.zeros((len(star_grid), len(wgrid)))

    else:

        y_c = np.sqrt((planet_position[0] - star_grid[:, 0])**2 + (planet_position[1] - star_grid[:, 1])**2)

        tau_grid = vectorized_tau_los_ena(y_c, planet_velocity, pw_at_hill_sphere, ENA_at_hill_sphere, hill_sphere_radius, wgrid)

        return tau_grid


def make_transit_tools_hill_ena(star_radius, n_star_cells, n_z_cells = None):

    star_grid, areas_array = sg.make_grid_cartesian2(star_radius, n_star_cells)

    def do_transit_hill_ena(parameters, pw_functions, phase_grid, wgrid, inclination, ENA):


        intensity_array = np.empty((len(phase_grid), len(wgrid)))
        hill_sphere_radius = parameters['semimajoraxis']*(parameters['mass_p']/(3*parameters['mass_s']))**(1/3)

        pw_at_hill_sphere = {'neutral_fraction' : pw_functions['neutral_fraction'](hill_sphere_radius, 0)}

        ENA_at_hill_sphere = {'rho' : ENA.SW.post_shock_rho(parameters['semimajoraxis'], 1), 'velocity' : ENA.u_ENA, 'temperature' : ENA.SW.get_T(parameters['semimajoraxis']), 'L_mix': ENA.L_mix}

        for index, phase in enumerate(phase_grid):
            planet_position = cc.convert_point_on_orbitalplane_to_transitcoords(parameters['semimajoraxis'], phase, inclination)
            planet_velocity = cc.convert_vector_on_orbitalplane_to_transitcoords(0, 0, parameters['semimajoraxis'], phase, inclination)
            tau_grid = get_tau_at_phase_hill_ena(star_grid, planet_position, planet_velocity, pw_at_hill_sphere, ENA_at_hill_sphere, hill_sphere_radius, wgrid)
            intensity = np.einsum('i, ij -> j', areas_array, np.exp(-tau_grid)) / (np.pi * star_radius**2)
            intensity_array[index, :] = intensity

        return phase_grid, intensity_array

    def do_transit_hill_ena_tau(parameters, pw_functions, phase_grid, wgrid, inclination, ENA):


        tau_array = np.empty((len(phase_grid), len(star_grid), len(wgrid)))
        hill_sphere_radius = parameters['semimajoraxis']*(parameters['mass_p']/(3*parameters['mass_s']))**(1/3)

        pw_at_hill_sphere = {'neutral_fraction' : pw_functions['neutral_fraction'](hill_sphere_radius, 0)}

        ENA_at_hill_sphere = {'rho' : ENA.SW.post_shock_rho(parameters['semimajoraxis'], 1), 'velocity' : ENA.u_ENA, 'temperature' : ENA.SW.get_T(parameters['semimajoraxis']), 'L_mix': ENA.L_mix}

        for index, phase in enumerate(phase_grid):
            planet_position = cc.convert_point_on_orbitalplane_to_transitcoords(parameters['semimajoraxis'], phase, inclination)
            planet_velocity = cc.convert_vector_on_orbitalplane_to_transitcoords(0, 0, parameters['semimajoraxis'], phase, inclination)
            tau_grid = get_tau_at_phase_hill_ena(star_grid, planet_position, planet_velocity, pw_at_hill_sphere, ENA_at_hill_sphere, hill_sphere_radius, wgrid)
            tau_array[index, :] = tau_grid

        return phase_grid, tau_array

    return do_transit_hill_ena, do_transit_hill_ena_tau
