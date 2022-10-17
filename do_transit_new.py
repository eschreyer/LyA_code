import numpy as np
import scipy.integrate as sp_int
import scipy.interpolate as sp_intpl
from numba import jit
import xsection_new as xs
import change_coords_of_tail_trajectory_new as cc
import star_grid_new as sg
import constants_new as const
import tail_object_holders_new as toh

"""This Module is responsible for doing the ray tracing   . For more detail look at 'do_transit.ipynb' """

@jit(nopython = True)
def find_s_grid(xy_grid, z_grid, cartesian_solution):

    s_grid = np.zeros((np.shape(xy_grid)[0], np.shape(z_grid)[0]))
    xyz_grid = np.zeros((np.shape(xy_grid)[0], np.shape(z_grid)[0], 3))

    for i_xy, xy in enumerate(xy_grid):
        for i_z, z in enumerate(z_grid):

            #want to find the the s values in which the tangent of the curve and line from curve to grid point are normal

            d_prod = xy[0] * cartesian_solution.velocity_x + xy[1] * cartesian_solution.velocity_y + z * cartesian_solution.velocity_z - cartesian_solution.x * cartesian_solution.velocity_x - cartesian_solution.y * cartesian_solution.velocity_y - cartesian_solution.z * cartesian_solution.velocity_z

            d_prod_roots = np.nonzero(np.sign(d_prod[:-1]) != np.sign(d_prod[1:]))[0] + 1

            #in the case of multiple roots, choose s value that is closest to the grid point #need to check this

            dist_to_grid = np.inf #initialise at infinity so first possible root takes over value

            s = 0

            for root in d_prod_roots:

                dist = (xy[0] - cartesian_solution.x[root])**2 + (xy[1] - cartesian_solution.y[root])**2 + (z - cartesian_solution.z[root])**2

                if dist < dist_to_grid:

                    dist_to_grid = dist

                    s = cartesian_solution.s[root - 1] - (d_prod[root - 1] / ((d_prod[root] - d_prod[root - 1]) / (cartesian_solution.s[root] - cartesian_solution.s[root - 1])))

            s_grid[i_xy, i_z] = s
            xyz_grid[i_xy, i_z] = np.array([xy[0], xy[1], z])


    flat_sgrid = np.ravel(s_grid)
    flat_xyz_grid = np.reshape(xyz_grid, (len(flat_sgrid), 3))

    flat_nz_s_grid_ix = np.flatnonzero(s_grid)
    flat_nz_s_grid = np.zeros(len(flat_nz_s_grid_ix))
    flat_nz_xyz_grid = np.zeros((len(flat_nz_s_grid_ix), 3))


    for i in range(len(flat_nz_s_grid)):
        flat_nz_s_grid[i] = flat_sgrid[flat_nz_s_grid_ix[i]]
        flat_nz_xyz_grid[i] = flat_xyz_grid[flat_nz_s_grid_ix[i]]


    #maybe figure out a way to only interpolate non masked values
    #return x,y,z grid
    x_grid = np.interp(flat_nz_s_grid, cartesian_solution.s, cartesian_solution.x)
    y_grid = np.interp(flat_nz_s_grid, cartesian_solution.s, cartesian_solution.y)
    z_grid = np.interp(flat_nz_s_grid, cartesian_solution.s, cartesian_solution.z)
    s_position_grid = np.stack((x_grid, y_grid, z_grid), axis = -1)


    #return u_x, u_y, u_z grid
    u_x_grid = np.interp(flat_nz_s_grid, cartesian_solution.s, cartesian_solution.velocity_x)
    u_y_grid = np.interp(flat_nz_s_grid, cartesian_solution.s, cartesian_solution.velocity_y)
    u_z_grid = np.interp(flat_nz_s_grid, cartesian_solution.s, cartesian_solution.velocity_z)
    s_velocity_grid = np.stack((u_x_grid, u_y_grid, u_z_grid), axis = -1)

    return flat_nz_xyz_grid, flat_nz_s_grid, s_position_grid, s_velocity_grid, flat_nz_s_grid_ix

"""--------------------------------------------------------------------------------------------------------------------"""

def change_components_from_cartesian_to_ellipse_matrix(s_velocity_grid, i):
    """

    """
    U_x, U_y, U_z = s_velocity_grid[:, 0], s_velocity_grid[:, 1], s_velocity_grid[:, 2]

    U = np.reshape(np.sqrt(U_x**2 + U_y**2 + U_z**2), (len(s_velocity_grid), 1))

    row1 = s_velocity_grid / U

    row2 = np.stack((- U_y * np.cos(i) + U_z * np.sin(i), -U_x * np.cos(i), -U_x * np.sin(i)), axis = -1) / U

    row3 = np.tile(np.array([0, -np.sin(i), np.cos(i)]), (len(s_velocity_grid), 1))

    return np.stack((row1, row2, row3), axis = 1)

def change_components_from_cartesian_to_ellipse(grid_point, point_on_tail_coordinates, point_on_tail_velocity, i):
    """

    """
    change_to_ellipse_coords_matrix = change_components_from_cartesian_to_ellipse_matrix(point_on_tail_velocity, i)
    position_in_ellipse_coordinates = np.einsum('ijk, ik -> ij', change_to_ellipse_coords_matrix, grid_point - point_on_tail_coordinates) #[s, n , z']
    return position_in_ellipse_coordinates

def is_point_in_ellipse1(position_in_ellipse_coordinates, ellipse_height, ellipse_depth):
    """

    """
    return (position_in_ellipse_coordinates[:, 1]**2)/(ellipse_depth**2) + (position_in_ellipse_coordinates[:, 2]**2)/(ellipse_height**2) < 1

"""--------------------------------------------------------------------------------------------------------------------------------"""
def get_tau_at_phase(xy_grid, z_grid, cartesian_solution, w, rho_struc, u_los = None):

    #this gets the positions on the tail (if there are two it returns the one which is the minimum distance)
    flat_nz_xyz_grid, flat_nz_s_grid, s_position_grid, s_velocity_grid, flat_nz_s_grid_ix = find_s_grid(xy_grid, z_grid, cartesian_solution) #shape (k), shape (k,3), shape (k,3)

    #height and depth
    height_grid, depth_grid = rho_struc.get_height_and_depth(s_position_grid, s_velocity_grid)


    #change points grid into ellipse coordinates
    point_grid_ellipse_coords = change_components_from_cartesian_to_ellipse(flat_nz_xyz_grid, s_position_grid, s_velocity_grid, np.pi/2)  #(k,3)

    #check that s coordinate is zero (or close to zero at least)
    #print(point_grid_ellipse_coords)

    #check if point is in ellipse
    is_point_in_ellipse = is_point_in_ellipse1(point_grid_ellipse_coords, height_grid, depth_grid) #might be able to reduce this to a generic coordinate change

    #reduce grid, only keep the points in the ellipse

    flat_nz_s_grid2 = flat_nz_s_grid[is_point_in_ellipse != 0] #shape (k1)

    if flat_nz_s_grid2.size == 0:
        return np.zeros((xy_grid.shape[0], w.shape[0]))

    flat_nz_s_grid_ix2 = flat_nz_s_grid_ix[is_point_in_ellipse != 0] #shape(k1)

    depth_grid2 = depth_grid[is_point_in_ellipse != 0]

    #tau calculation

    neutral_fraction_grid = np.interp(flat_nz_s_grid2, cartesian_solution.s, cartesian_solution.neutral_fraction) #shape (k)

    density_grid = rho_struc.get_density(point_grid_ellipse_coords[:, 1][is_point_in_ellipse != 0], point_grid_ellipse_coords[:, 2][is_point_in_ellipse != 0], s_position_grid[is_point_in_ellipse != 0], s_velocity_grid[is_point_in_ellipse != 0], depth_grid2) #shape (k1)

    if u_los == None:

        z_velocity = (s_velocity_grid[:,2])[is_point_in_ellipse != 0]  #shape (k1)

    else:

        z_velocity = u_los(flat_nz_s_grid2)

    w_mesh, z_velocity_mesh = np.meshgrid(w, z_velocity)  #write as o_grid to speed up but beware indices are swapped

    x_section_grid = xs.LyA_xsection(w_mesh, z_velocity_mesh, 10**4)  #shape (k1, l) , #where l is the number of wavelength points

    dtau_grid_flat = xs.d_tau(np.reshape(density_grid * neutral_fraction_grid, (len(density_grid), 1)), x_section_grid, const.m_proton, z_grid[1] - z_grid[0]) #shape (k, l)

    tau_grid = np.zeros((len(xy_grid), len(w)))

    for index, value in zip(flat_nz_s_grid_ix2, dtau_grid_flat):

        tau_grid[index // len(z_grid)] += value

    return tau_grid

"""----------------------------------------------------------------------------------------------------------------------------------------------------"""

def make_transit_tools(star_radius):

    #specify star grid in which to perform ray tracing
    star_grid, areas_array = sg.make_grid_cartesian2(star_radius, 20)

    #specify z_grid
    def get_z_grid(tail_transitcoords_array, rho_struc, number_of_z_intervals):
        x_masks = np.array([tail_transitcoords_array.x < 1.5 * star_radius, tail_transitcoords_array.x > - 1.5 * star_radius])
        total_x_masks = np.all(x_masks, axis = 0)
        if np.any(total_x_masks) == False:
            return np.empty(0), tail_transitcoords_array
        else:
            max_height = np.max(rho_struc.get_height_and_depth(np.stack((tail_transitcoords_array.x[total_x_masks], tail_transitcoords_array.y[total_x_masks], tail_transitcoords_array.z[total_x_masks]), axis = 1), np.stack((tail_transitcoords_array.velocity_x[total_x_masks], tail_transitcoords_array.velocity_y[total_x_masks], tail_transitcoords_array.velocity_z[total_x_masks]), axis = 1)))
            yz_masks = np.array([tail_transitcoords_array.y > - star_radius - max_height, tail_transitcoords_array.y < star_radius + max_height, tail_transitcoords_array.z > 0])
            total_masks = np.all(np.concatenate((np.array([total_x_masks]), yz_masks)), axis = 0)
            masked_z = np.copy(tail_transitcoords_array.z)[total_masks]
            new_transitcoords_array = toh.TailTransitCoordArray(s = tail_transitcoords_array.s[total_masks], x = tail_transitcoords_array.x[total_masks], y = tail_transitcoords_array.y[total_masks], z = tail_transitcoords_array.z[total_masks], velocity_x = tail_transitcoords_array.velocity_x[total_masks], velocity_y = tail_transitcoords_array.velocity_y[total_masks], velocity_z = tail_transitcoords_array.velocity_z[total_masks], neutral_fraction = tail_transitcoords_array.neutral_fraction[total_masks])
            if np.size(masked_z) == 0:
                return np.empty(0), tail_transitcoords_array
            else:
                max_z = np.max(masked_z)
                min_z = np.min(masked_z)
                return np.linspace(min_z - 2 * max_height, max_z + 2 * max_height, number_of_z_intervals), new_transitcoords_array


    def do_transit(tail_polar, phase, w, rho_struc, inclination, u_los = None):

        intensity_array = np.empty((len(phase), len(w)))

        for index, p in enumerate(phase):
            tail_transitcoords_array = cc.change_tail_trajectory_from_orbitalplane_to_transitcoords(tail_polar, p, inclination)
            z_grid, new_transitcoords_array = get_z_grid(tail_transitcoords_array, rho_struc, 30)
            tau_grid = get_tau_at_phase(star_grid, z_grid, new_transitcoords_array, w, rho_struc, u_los)
            intensity = np.einsum('i, ij -> j', areas_array, np.exp(-tau_grid)) / (np.pi * star_radius**2)
            intensity_array[index, :] = intensity

        return phase, intensity_array

    return do_transit
