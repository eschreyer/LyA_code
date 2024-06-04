import os.path

from matplotlib import pyplot as plt
from astropy import time
from astropy import coordinates as coord
from astropy import units as u
from astropy import table
import numpy as np

import gj436b

earth_center = coord.EarthLocation.from_geocentric(0, 0, 0, unit='m')
path_phasetbl = 'gj436b_phases.fits'

def _check_time(t):
    if not isinstance(t, time.Time):
        raise ValueError('Input must be an astropy.time object.')

def bary_offset(t):
    # add this to a geocentric time and you get the barycentric time
    _check_time(t)
    return t.light_travel_time(gj436b.position, location=earth_center)


def nearest_transit(t_bary):
    _check_time(t_bary)
    Nperiods = (t_bary - gj436b.Tmid) / gj436b.period
    Nperiods = np.round(Nperiods)
    transit_time = gj436b.Tmid + gj436b.period*Nperiods

    transit_time_error = np.sqrt(gj436b.Tmid_err.sec ** 2 + (Nperiods.value * gj436b.period_err.sec) ** 2)
    transit_time_error = time.TimeDelta(transit_time_error, format='sec')

    return transit_time, transit_time_error


def transit_offset(t_bary):
    _check_time(t_bary)
    transit_time, transit_time_error = nearest_transit(t_bary)
    offsets = t_bary - transit_time
    return offsets, transit_time_error


def get_x1d_ext_timeinfo(ext):
    hdr = ext.header
    ta, tb = hdr['expstart'], hdr['expend']
    t = (ta + tb) / 2.

    ts = time.Time((ta, t, tb), format='mjd')
    dt = bary_offset(ts)
    tbary = ts + dt

    p, _ = transit_offset(tbary)

    return tbary.jd, p.to('h')


