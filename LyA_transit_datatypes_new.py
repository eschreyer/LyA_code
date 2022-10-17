from typing import NamedTuple

"""This module creates the """

class Planet(NamedTuple):
    mass : float
    radius : float
    semimajoraxis : float
    inclination : float

class Star(NamedTuple):
    mass : float
    radius : float

class ModelParameters(NamedTuple):
    """We test over these parameters"""

    # planetary wind parameters: sound speed and mass loss rate
    c_s_planet : float
    mdot_planet : float

    # stellar wind parameters: velocity (function of r?), mass loss rate and temperature
    v_stellar_wind : float
    mdot_star : float
    T_stellar_wind : float

    # stellar radiation parameters: EUV
    L_EUV : float

    # angle of launch of wind in comparison .
    #angle measured in radians, 0 degrees in line with planet instanteous velocity and measured clockwise.
    #prior assume only angles between pi/2 and pi are possible
    #May transfer into angular momentum defecit

    angle : float
