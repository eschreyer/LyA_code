from astropy import units as u

HI_crosssection = 6.3e-18 #* u.cm**2

recombination_rate_caseA = 4.18e-13 #* u.cm**3/u.s

k_b = 1.3807e-16 #u.erg/u.K

m_proton = 1.67e-24 #*u.g

G = 6.67e-8 #*

c = 2.9979e10

LyA_linecenter_wav = 1.21567e-5

LyA_linecenter_w = c / LyA_linecenter_wav

hbar = 1.0546e-27

e = 4.803e-10

m_e = 9.11e-28

"""celestial body constants"""

m_earth = 5.972e27
r_earth = 6.3e8
m_sun = 1.989e33
r_sun = 6.99e10
m_jupiter = 1.89e30
r_jupiter = 6.99e9
mdot_sun = (2e-14 * m_sun) / (365 * 24 * 60 * 60)
