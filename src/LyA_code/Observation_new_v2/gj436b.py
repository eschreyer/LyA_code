from astropy import units as u
from astropy import constants as const
from astropy import coordinates as coord
from astropy import time
from math import pi

radec = (175.546225, 26.706569)

position = coord.SkyCoord(*radec, unit='deg')

# STELLAR PROPERTIES
rv = 9.61 *u.km/u.s #nidever02
Rstar = 0.455 * u.Rsun

# PLANET PHYSICAL PROPERTIES
Mp = 25.4 * const.M_earth  # lanotte14
Rp = 4.1 * const.R_earth # lanotte14
Tatm = 583 * u.K # line14
RpRs = 0.08311 # knutson11

# PLANET ORBITAL PROPERTIES
periastron_longitude = 340.0 * pi/180.
eccentricity = 0.1616
a = 0.0308 * u.AU  # AU

# EPHEMERIS
# from bourrier+ 2018 https://ui.adsabs.harvard.edu/abs/2018Natur.553..477B
Tmid = time.Time(2454865.084034, format='jd')
Tmid_err = time.TimeDelta(0.000035*u.d)
period =  time.TimeDelta(2.64389803 * u.d)
period_err =  time.TimeDelta(0.00000026*u.d)
