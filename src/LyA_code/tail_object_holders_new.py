from typing import NamedTuple
import numpy as np
import scipy.interpolate as sp_itpl
"""
This document creates named tuple objects containing the attributes of the tail
in various different coordinate and systems and different datatypes
"""



class TailOrbitalPlanePolarArray(NamedTuple):
    #change to s once this is all working
    #maybe change the whole datatype to something more clear like below
    t: np.ndarray
    y: np.ndarray


class TailTransitCoordArray(NamedTuple):
    s : np.ndarray
    x : np.ndarray
    y : np.ndarray
    z : np.ndarray
    velocity_x : np.ndarray
    velocity_y : np.ndarray
    velocity_z : np.ndarray
    neutral_fraction : np.ndarray

class TailTransitCoordInterpolate(NamedTuple):
    x : sp_itpl.InterpolatedUnivariateSpline
    y : sp_itpl.InterpolatedUnivariateSpline
    z : sp_itpl.InterpolatedUnivariateSpline
    velocity_x : sp_itpl.InterpolatedUnivariateSpline
    velocity_y : sp_itpl.InterpolatedUnivariateSpline
    velocity_z : sp_itpl.InterpolatedUnivariateSpline
    neutral_fraction : sp_itpl.InterpolatedUnivariateSpline

class EllipseParameters(NamedTuple):
    center_position: np.ndarray #in transit coordinates
    center_velocity: np.ndarray #in transit coordinates
    r : float
    height : float
    depth : float
    

"""
"""

def get_position_in_transit_coords(point_on_tail, tail_transit_coord_interpolant):
    return np.array([tail_transit_coord_interpolant.x(point_on_tail), tail_transit_coord_interpolant.y(point_on_tail), tail_transit_coord_interpolant.z(point_on_tail)])

def get_velocity_in_transit_coords(point_on_tail, tail_transit_coord_interpolant):
    return np.array([tail_transit_coord_interpolant.velocity_x(point_on_tail), tail_transit_coord_interpolant.velocity_y(point_on_tail), tail_transit_coord_interpolant.velocity_z(point_on_tail)])
