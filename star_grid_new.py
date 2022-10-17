"""This module makes a polar grid. This is made such that each grid box is roughly square (in a polar sense).
The two required parameters to make this grid are the radius of the polar grid and the number of different radius levels.
The main function returns the points at the centre of each grid box and the areas the corresponding box"""

import numpy as np

def grid_points(radius_disc, number_of_r_intervals):
    r_positions1 = np.linspace(radius_disc/number_of_r_intervals,radius_disc, number_of_r_intervals)
    r_positions = np.delete(r_positions1,-1) + radius_disc/(2*number_of_r_intervals)
    number_of_r_points = (2*np.pi/(radius_disc/number_of_r_intervals))*r_positions
    number_of_r_points1 = np.rint(number_of_r_points).astype(int)
    r_points = np.repeat(r_positions,number_of_r_points1)
    theta_points = make_theta_points(number_of_r_points1)
    areas_array = make_areas(r_positions1, number_of_r_points1)
    return r_points, theta_points, areas_array

def make_areas(r_boundaries, number_of_r_points1):
    upper_boundaries = np.delete(r_boundaries,0)
    lower_boundaries = np.delete(r_boundaries,-1)
    areas = (upper_boundaries**2 - lower_boundaries**2)*(np.pi/number_of_r_points1)
    areas_array = np.repeat(areas, number_of_r_points1)
    areas_array1 = np.append(areas_array, np.pi*r_boundaries[0]**2)
    return areas_array1

def make_theta_points(number_of_points_per_r_position):
    theta_points = np.array([])
    for i in number_of_points_per_r_position:
        theta_points = np.append(theta_points,np.linspace(0,2*np.pi,i, endpoint = False))
    return theta_points


def make_grid_cartesian(r_points,theta_points):
    x_points = r_points*np.cos(theta_points)
    y_points = r_points*np.sin(theta_points)
    cartesian_grid = np.column_stack((x_points,y_points))
    cartesian_grid1 = np.append(cartesian_grid,[[0.0,0.0]], axis = 0)
    return cartesian_grid1

def make_grid_cartesian2(radius_disc, number_of_r_intervals):
    r_points, theta_points, areas_array1 = grid_points(radius_disc, number_of_r_intervals)
    cartesian_grid = make_grid_cartesian(r_points, theta_points)
    return cartesian_grid, areas_array1


def show_grid_plot(radius_disc, numberof_r_intervals):
    r_points, theta_points = grid_points(radius_disc, numberof_r_intervals)
    cartesian_grid = make_grid_cartesian(r_points,theta_points)
    x_points,y_points = zip(*cartesian_grid)
    plt.polar(theta_points, r_points, 'r.')
    plt.show()
    plt.plot(x_points,y_points, 'r.')
    plt.show()
