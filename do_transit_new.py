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

@jit(nopython = True)
def find_s_grid1(xyz_grid, cartesian_solution):
    #xyz grid should have a shape (n, 3)

    #initialise s_grid

    s_grid = np.zeros(np.shape(xyz_grid)[0])


    ##write a loop implementation because vectorization seems confusing. Also will use numba so performance difference may be minimal
    for i_xyz, xyz in enumerate(xyz_grid):

            #want to find the the s values in which the tangent of the curve and line from curve to grid point are normal

            d_prod = xyz[0] * cartesian_solution.velocity_x + xyz[1] * cartesian_solution.velocity_y + xyz[2] * cartesian_solution.velocity_z - cartesian_solution.x * cartesian_solution.velocity_x - cartesian_solution.y * cartesian_solution.velocity_y - cartesian_solution.z * cartesian_solution.velocity_z

            d_prod_roots = np.nonzero(np.sign(d_prod[:-1]) != np.sign(d_prod[1:]))[0] + 1

            #in the case of multiple roots, choose s value that is closest to the grid point #need to check this

            dist_to_grid = np.inf #initialise at infinity so first possible root takes over value

            s = 0

            for root in d_prod_roots:

                dist = (xyz[0] - cartesian_solution.x[root])**2 + (xyz[1] - cartesian_solution.y[root])**2 + (xyz[2] - cartesian_solution.z[root])**2

                #print(dist)

                if dist < dist_to_grid:

                    dist_to_grid = dist

                    s = cartesian_solution.s[root - 1] - (d_prod[root - 1] / ((d_prod[root] - d_prod[root - 1]) / (cartesian_solution.s[root] - cartesian_solution.s[root - 1])))

            s_grid[i_xyz] = s


    flat_nz_s_grid_ix = np.flatnonzero(s_grid)
    flat_nz_s_grid = np.zeros(len(flat_nz_s_grid_ix))
    flat_nz_xyz_grid = np.zeros((len(flat_nz_s_grid_ix), 3))


    for i in range(len(flat_nz_s_grid)):
        flat_nz_s_grid[i] = s_grid[flat_nz_s_grid_ix[i]]
        flat_nz_xyz_grid[i] = xyz_grid[flat_nz_s_grid_ix[i]]




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


"""
--------------------------------------------------------------------------------------------------------------------------
"""
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

def find_is_point_in_ellipse_ENA(position_in_ellipse_coordinates, ellipse_height, ellipse_depth, ENA):
    """

    """

    return ((position_in_ellipse_coordinates[:, 1]**2)/(ellipse_depth**2) + (position_in_ellipse_coordinates[:, 2]**2)/(ellipse_height**2) > 1) & ((position_in_ellipse_coordinates[:, 1]**2)/((ellipse_depth + ENA.L_mix * ellipse_depth)**2) + (position_in_ellipse_coordinates[:, 2]**2)/((ellipse_height + ENA.L_mix * ellipse_height)**2) < 1)

"""--------------------------------------------------------------------------------------------------------------------------------"""
def get_tau_at_phase(xy_grid, z_grid, cartesian_solution, w, rho_struc, inclination, u_los = None):
    #just for calculating pw part

    #this gets the positions on the tail (if there are two it returns the one which is the minimum distance)
    flat_nz_xyz_grid, flat_nz_s_grid, s_position_grid, s_velocity_grid, flat_nz_s_grid_ix = find_s_grid(xy_grid, z_grid, cartesian_solution) #shape (k), shape (k,3), shape (k,3)

    #height and depth
    height_grid, depth_grid = rho_struc.get_height_and_depth(s_position_grid, s_velocity_grid)


    #change points grid into ellipse coordinates
    point_grid_ellipse_coords = change_components_from_cartesian_to_ellipse(flat_nz_xyz_grid, s_position_grid, s_velocity_grid, inclination)  #(k,3)

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

    x_section_grid = xs.LyA_xsection(w_mesh, z_velocity_mesh, rho_struc.c_s**2 * const.m_proton / (2 * const.k_b))  #shape (k1, l) , #where l is the number of wavelength points

    dtau_grid_flat = xs.d_tau(np.reshape(density_grid * neutral_fraction_grid, (len(density_grid), 1)), x_section_grid, const.m_proton, z_grid[1] - z_grid[0]) #shape (k, l)

    tau_grid = np.zeros((len(xy_grid), len(w)))

    for index, value in zip(flat_nz_s_grid_ix2, dtau_grid_flat):

        tau_grid[index // len(z_grid)] += value

    return tau_grid

"""
------------------------------------------------------------------------------------------------------------------------------------------------------
#tau grid calculation including ENA
"""
def get_tau_at_phase_w_ENA(xy_grid, z_grid, cartesian_solution, w, rho_struc, inclination, ENA = None, u_los = None):

    #this gets the positions on the tail (if there are two it returns the one which is the minimum distance)
    flat_nz_xyz_grid, flat_nz_s_grid, s_position_grid, s_velocity_grid, flat_nz_s_grid_ix = find_s_grid(xy_grid, z_grid, cartesian_solution) #shape (k), shape (k,3), shape (k,3)

    if flat_nz_s_grid.size == 0:

        return np.zeros((len(xy_grid), len(w)))

    #height and depth
    height_grid, depth_grid = rho_struc.get_height_and_depth(s_position_grid, s_velocity_grid)


    #change points grid into ellipse coordinates
    point_grid_ellipse_coords = change_components_from_cartesian_to_ellipse(flat_nz_xyz_grid, s_position_grid, s_velocity_grid, inclination)  #(k,3)

    #check that s coordinate is zero (or close to zero at least)
    #print(point_grid_ellipse_coords)

    #check if point is in ellipse pw, elliptical disk ENA
    is_point_in_ellipse_pw = is_point_in_ellipse1(point_grid_ellipse_coords, height_grid, depth_grid) #might be able to reduce this to a generic coordinate change

    #planetary wind part

    #reduce grid, only keep the points in the ellipse
    flat_nz_s_grid_pw = flat_nz_s_grid[is_point_in_ellipse_pw != 0] #shape (k1)

    if flat_nz_s_grid_pw.size == 0:

        tau_grid_pw = np.zeros((xy_grid.shape[0], w.shape[0]))

        return tau_grid_pw

    else:

        flat_nz_s_grid_ix_pw = flat_nz_s_grid_ix[is_point_in_ellipse_pw != 0] #shape(k1)

        depth_grid2 = depth_grid[is_point_in_ellipse_pw != 0]

        #tau calculation

        neutral_fraction_grid_pw = np.interp(flat_nz_s_grid_pw, cartesian_solution.s, cartesian_solution.neutral_fraction) #shape (k)

        density_grid_pw = rho_struc.get_density(point_grid_ellipse_coords[:, 1][is_point_in_ellipse_pw != 0], point_grid_ellipse_coords[:, 2][is_point_in_ellipse_pw != 0], s_position_grid[is_point_in_ellipse_pw != 0], s_velocity_grid[is_point_in_ellipse_pw != 0], depth_grid2) #shape (k1)

        if u_los == None:

            z_velocity_pw = (s_velocity_grid[:,2])[is_point_in_ellipse_pw != 0]  #shape (k1)

        else:

            z_velocity_pw = u_los(flat_nz_s_grid_pw)

        w_mesh_pw, z_velocity_mesh_pw = np.meshgrid(w, z_velocity_pw)  #write as o_grid to speed up but beware indices are swapped

        x_section_grid_pw = xs.LyA_xsection(w_mesh_pw, z_velocity_mesh_pw, rho_struc.c_s**2 * const.m_proton / (2 * const.k_b))  #shape (k1, l) , #where l is the number of wavelength points

        dtau_grid_flat_pw = xs.d_tau(np.reshape(density_grid_pw * neutral_fraction_grid_pw, (len(density_grid_pw), 1)), x_section_grid_pw, const.m_proton, z_grid[1] - z_grid[0]) #shape (k, l)

        tau_grid_pw = np.zeros((len(xy_grid), len(w)))

        for index, value in zip(flat_nz_s_grid_ix_pw, dtau_grid_flat_pw):

            tau_grid_pw[index // len(z_grid)] += value

        if ENA == None:

            return tau_grid_pw

        n_ENA_cells = int(np.max([0.1 // ENA.L_mix, 4]))

        max_to_min_h_ratio = np.max((depth_grid, height_grid)) / np.min((depth_grid, height_grid))

        n_ex_cells = int((2 * max_to_min_h_ratio * ENA.L_mix) // 0.1 + 1)

        xyz_grid_ENA, xy_ix = get_ENA_grid(flat_nz_s_grid_ix_pw, flat_nz_xyz_grid, flat_nz_s_grid_ix, n_ENA_cells, n_ex_cells, z_grid)

        tau_grid_ENA = get_tau_at_phase_ENA(xyz_grid_ENA, xy_ix, len(xy_grid), cartesian_solution, w, rho_struc, inclination, (z_grid[1] - z_grid[0]) / n_ENA_cells, ENA, u_los = None)

        return tau_grid_pw + tau_grid_ENA



    """##ENA part

    ##reduce grid, only keep points in ENA elliptical disc
    is_point_in_ellipse_ENA = find_is_point_in_ellipse_ENA(point_grid_ellipse_coords, height_grid, depth_grid, ENA)

    flat_nz_s_grid_ENA = flat_nz_s_grid[is_point_in_ellipse_ENA != 0] #shape (k1)

    if flat_nz_s_grid_ENA.size == 0:

        tau_grid_ENA = np.zeros((xy_grid.shape[0], w.shape[0]))

    else:

        flat_nz_s_grid_ix_ENA = flat_nz_s_grid_ix[is_point_in_ellipse_ENA != 0] #shape(k1)

        #depth_grid2 = depth_grid[is_point_in_ellipse != 0]

        #tau calculation

        neutral_fraction_grid_ENA = np.interp(flat_nz_s_grid_ENA, cartesian_solution.s, cartesian_solution.neutral_fraction) #shape (k)

        density_grid_ENA = ENA.get_rho(s_position_grid[is_point_in_ellipse_ENA != 0], s_velocity_grid[is_point_in_ellipse_ENA != 0]) #shape (k1)

        z_velocity_ENA = (s_velocity_grid[:,2])[is_point_in_ellipse_ENA != 0] + ENA.u_ENA  #shape (k1)

        w_mesh_ENA, z_velocity_mesh_ENA = np.meshgrid(w, z_velocity_ENA)  #write as o_grid to speed up but beware indices are swapped

        x_section_grid_ENA = xs.LyA_xsection(w_mesh_ENA, z_velocity_mesh_ENA, 10**5)  #shape (k1, l) , #where l is the number of wavelength points

        dtau_grid_flat_ENA = xs.d_tau(np.reshape(density_grid_ENA * neutral_fraction_grid_ENA, (len(density_grid_ENA), 1)), x_section_grid_ENA, const.m_proton, z_grid[1] - z_grid[0]) #shape (k, l)

        tau_grid_ENA = np.zeros((len(xy_grid), len(w)))

        for index, value in zip(flat_nz_s_grid_ix_ENA, dtau_grid_flat_ENA):

            tau_grid_ENA[index // len(z_grid)] += value

    return tau_grid_pw + tau_grid_ENA"""
"""----------------------------------------------------------------------------------------------------------------------------------------------------
Getting ENA grid
"""

def get_ENA_grid(flat_nz_s_grid_ix2, flat_nz_xyz_grid, flat_nz_s_grid_ix, n_split_cells, n_ex_cells, z_grid):

    i = np.flatnonzero((flat_nz_s_grid_ix2[1:] - flat_nz_s_grid_ix2[:-1]) - 1)

    flat_nz_s_grid_ix_ENA = np.unique(np.concatenate((np.concatenate([flat_nz_s_grid_ix2[i] + j for j in range(n_ex_cells)]), np.concatenate([flat_nz_s_grid_ix2[i+1] - j for j in range(n_ex_cells)]))))

    is_possible_ENA = np.isin(flat_nz_s_grid_ix, flat_nz_s_grid_ix_ENA)

    xyz_grid = flat_nz_xyz_grid[is_possible_ENA]

    flat_nz_s_grid_ix_ENA = flat_nz_s_grid_ix[is_possible_ENA]

    #make grid fine

    dz = z_grid[1] - z_grid[0]

    z_grid_a = np.zeros((n_split_cells, 3))

    z_grid_a[:, 2] = np.linspace(- dz / 2 + dz / (2 * (n_split_cells + 1)), - dz / 2 + dz * ((2 * n_split_cells + 1) / (2 * (n_split_cells + 1))), n_split_cells)

    xyz_grid_ENA = np.repeat(xyz_grid, n_split_cells, axis = 0) + np.tile(z_grid_a, [np.shape(xyz_grid)[0], 1])

    xy_ix = np.repeat(flat_nz_s_grid_ix_ENA // len(z_grid), n_split_cells, axis = 0)

    return xyz_grid_ENA, xy_ix

def get_tau_at_phase_ENA(xyz_grid, xy_ix, xy_grid_shape, cartesian_solution, w, rho_struc, inclination, dz_ENA, ENA, u_los = None):

    flat_nz_xyz_grid, flat_nz_s_grid, s_position_grid, s_velocity_grid, flat_nz_s_grid_ix = find_s_grid1(xyz_grid, cartesian_solution)

    nz_xy_ix = xy_ix[flat_nz_s_grid_ix]

    height_grid, depth_grid = rho_struc.get_height_and_depth(s_position_grid, s_velocity_grid)

    point_grid_ellipse_coords = change_components_from_cartesian_to_ellipse(flat_nz_xyz_grid, s_position_grid, s_velocity_grid, inclination)

    is_point_in_ellipse_ENA = find_is_point_in_ellipse_ENA(point_grid_ellipse_coords, height_grid, depth_grid, ENA)

    neutral_fraction_grid_ENA = np.interp(flat_nz_s_grid[is_point_in_ellipse_ENA != 0], cartesian_solution.s, cartesian_solution.neutral_fraction) #shape (k)

    density_grid_ENA = ENA.get_rho(s_position_grid[is_point_in_ellipse_ENA != 0], s_velocity_grid[is_point_in_ellipse_ENA != 0])

    z_velocity_ENA = (s_velocity_grid[:,2])[is_point_in_ellipse_ENA != 0] + ENA.u_ENA  #shape (k1)

    w_mesh, z_velocity_mesh_ENA = np.meshgrid(w, z_velocity_ENA)  #write as o_grid to speed up but beware indices are swapped

    x_section_grid_ENA = xs.LyA_xsection(w_mesh, z_velocity_mesh_ENA, ENA.SW.get_T(0)) #shape (k1, l) #set at zero because #where l is the number of wavelength points

    dtau_grid_flat_ENA = xs.d_tau(np.reshape(density_grid_ENA * neutral_fraction_grid_ENA, (len(density_grid_ENA), 1)), x_section_grid_ENA, const.m_proton, dz_ENA) #shape (k, l)

    tau_grid_ENA = np.zeros((xy_grid_shape, len(w)))

    for index, value in zip(nz_xy_ix[is_point_in_ellipse_ENA], dtau_grid_flat_ENA):

        tau_grid_ENA[index] += value

    return tau_grid_ENA


"""
-------------------------------------------------------------------------------------------------------------------------------------------------------
"""

def make_transit_tools(star_radius, n_star_cells, n_z_cells = None):

    #specify star grid in which to perform ray tracing
    star_grid, areas_array = sg.make_grid_cartesian2(star_radius, n_star_cells)

    #specify z_grid
    def get_z_grid(tail_transitcoords_array, rho_struc, ENA = None, n_z_cells = None):
        x_masks = np.array([tail_transitcoords_array.x < 1.5 * star_radius, tail_transitcoords_array.x > - 1.5 * star_radius])
        total_x_masks = np.all(x_masks, axis = 0)
        if np.any(total_x_masks) == False:
            return np.empty(0), tail_transitcoords_array
        else:
            min_height = np.min(rho_struc.get_height_and_depth(np.stack((tail_transitcoords_array.x[total_x_masks], tail_transitcoords_array.y[total_x_masks], tail_transitcoords_array.z[total_x_masks]), axis = 1), np.stack((tail_transitcoords_array.velocity_x[total_x_masks], tail_transitcoords_array.velocity_y[total_x_masks], tail_transitcoords_array.velocity_z[total_x_masks]), axis = 1)))
            max_height = np.max(rho_struc.get_height_and_depth(np.stack((tail_transitcoords_array.x[total_x_masks], tail_transitcoords_array.y[total_x_masks], tail_transitcoords_array.z[total_x_masks]), axis = 1), np.stack((tail_transitcoords_array.velocity_x[total_x_masks], tail_transitcoords_array.velocity_y[total_x_masks], tail_transitcoords_array.velocity_z[total_x_masks]), axis = 1)))
            yz_masks = np.array([tail_transitcoords_array.y > - star_radius - max_height * 1.1, tail_transitcoords_array.y < star_radius + max_height * 1.1, tail_transitcoords_array.z > 0])
            total_masks = np.all(np.concatenate((np.array([total_x_masks]), yz_masks)), axis = 0)
            masked_z = np.copy(tail_transitcoords_array.z)[total_masks]
            new_transitcoords_array = toh.TailTransitCoordArray(s = tail_transitcoords_array.s[total_masks], x = tail_transitcoords_array.x[total_masks], y = tail_transitcoords_array.y[total_masks], z = tail_transitcoords_array.z[total_masks], velocity_x = tail_transitcoords_array.velocity_x[total_masks], velocity_y = tail_transitcoords_array.velocity_y[total_masks], velocity_z = tail_transitcoords_array.velocity_z[total_masks], neutral_fraction = tail_transitcoords_array.neutral_fraction[total_masks])
            if np.size(masked_z) == 0:
                return np.empty(0), tail_transitcoords_array
            else:
                max_z = np.max(masked_z)
                min_z = np.min(masked_z)
                if n_z_cells:
                    return np.linspace(min_z - max_height, max_z + max_height, n_z_cells), new_transitcoords_array
                else:
                    if ENA:
                        n_z_cells = int((max_z - min_z + 2 * max_height) // (0.1 * min_height))
                        return np.linspace(min_z - max_height * (1 + ENA.L_mix), max_z + max_height * (1 + ENA.L_mix), n_z_cells), new_transitcoords_array
                    else:
                        n_z_cells = int((max_z - min_z + 2 * max_height) // (0.1 * min_height))
                        return np.linspace(min_z - max_height, max_z + max_height, n_z_cells), new_transitcoords_array



    def do_transit(tail_polar, phase, w, rho_struc, omega_p, inclination, ENA = None, u_los = None):

        intensity_array = np.empty((len(phase), len(w)))

        for index, p in enumerate(phase):
            tail_transitcoords_array = cc.change_tail_trajectory_from_orbitalplane_to_transitcoords(tail_polar, p, omega_p, inclination)
            z_grid, new_transitcoords_array = get_z_grid(tail_transitcoords_array, rho_struc, ENA, n_z_cells)
            tau_grid = get_tau_at_phase_w_ENA(star_grid, z_grid, new_transitcoords_array, w, rho_struc, inclination, ENA, u_los)
            intensity = np.einsum('i, ij -> j', areas_array, np.exp(-tau_grid)) / (np.pi * star_radius**2)
            intensity_array[index, :] = intensity

        return phase, intensity_array

    return do_transit
