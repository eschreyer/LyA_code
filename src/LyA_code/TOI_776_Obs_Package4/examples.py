import warnings

import astropy.units
import numpy as np
from matplotlib import pyplot as plt
from astropy import table

import stis
import chi2 as c2

#plt.ion()
warnings.simplefilter('ignore', astropy.units.UnitsWarning)

wrest_Lya = 1215.67
c = 3e5
rv_star = 49.34

###########################
# region G140M
path_aggregated_data = 'pubdata/g140m_aggregated_data_in_system_frame.fits'
data = table.Table.read(path_aggregated_data)

# because there are two planets in this system, I included phase info for both,
# but for fitting you should pick one by setting its info to the pha, phb, and ph columns
# the g140m data cover planet b, g140l cover planet c
data['pha'] = data['pha_b']
data['phb'] = data['phb_b']
data['ph'] = data['ph_b']
data.sort('ph')


# region fake a simulated transit
## region transit profile
max_depth = 0.25
tgrid = np.linspace(-50, 10, 1000)
sig_pre = 2.
sig_post = 3
time_depth = np.zeros_like(tgrid)
pre = tgrid < 0
post = tgrid > 0
time_depth[pre] = max_depth * np.exp(-tgrid[pre]**2/2/sig_pre**2)
time_depth[post] = max_depth * np.exp(-tgrid[post]**2/2/sig_post**2)

plt.figure()
plt.plot(tgrid, 1 - time_depth)
plt.title('transit time profile for code test')
plt.ylim(-0.1, 1.1)
## endregion


## region velocity profile
# NOTE grid should either extend until line profile is at 1e-3 of max on each side
# or 85 pixels beyond the data being fit for G140M data
# in this case I'll go to where the line profile is below 1e-3 of max, which is 1500 km/s
# fitting tools will raise an error otherwise
vgrid = np.arange(-1500, 1500, step=1)
sig = 50
offset = 0
wave_depth_factor = np.exp(-(vgrid - offset)**2/2/sig**2)
wgrid = (vgrid/3e5 + 1)*1215.67

plt.figure()
plt.plot(vgrid, wave_depth_factor)
plt.title('transit multiplication factor v wavelength for code test')
## endregion


## region multiply into a series of depths with axes time x wavelength
depths = time_depth[:,None] * wave_depth_factor[None,:]

plt.figure()
step = len(depths) // 20
sparse_depths = depths[::step, :]
n = len(sparse_depths)
colors = np.array([np.linspace(0, 1, n), np.zeros(n), 1 - np.linspace(0, 1, n)]).T
for depth, color in zip(sparse_depths, colors):
    plt.plot(vgrid, 1 - depth, color=color, alpha=0.5)
plt.title('transit depth v wavelength (x-axis) vs time (blue->red) for code test')
## endregion


## region check model transit spectra
oot_model = table.Table.read('pubdata/lya_fit_g140m.ecsv')
oot_profile = np.interp(wgrid, oot_model['w'], oot_model['y'])
model_spectra = oot_profile[None,:] * (1 - depths)
sparse_spectra = model_spectra[::step, :]
plt.figure()
for spec, color in zip(sparse_spectra, colors):
    plt.plot(wgrid, spec, color=color, alpha=0.5)
plt.title('model spectra vs time (blue->red) for code test')
## endregion
# endregion


# region compute chi2 for the fake transit model and the G140M Lya data

## region generate the goods for fitting
path_oot_line_profile = 'pubdata/lya_fit_g140m.ecsv'
line = table.Table.read(path_oot_line_profile)
spec = stis.g140m
fitpkgM = c2.FitPackage(wgrid, tgrid, spec, data, line, wrest=1215.67, epoch_divisions=(2459550,))
## endregion


## region simulate observations of spectra and show compared to data
sim_data = fitpkgM.simulate_spectra(depths, scalefacs=(10., 0.1)) # using large scalefacs to check that they work
v = (np.array(fitpkgM.w) - wrest_Lya) / wrest_Lya * c
def plot_simspec(i):
    label = 't = {:.1f} h'.format(fitpkgM.ph[i])
    plt.figure()
    ln, = plt.step(v[i], sim_data[i], where='mid', label=label + ' sim')
    ln, = plt.step(v[i], fitpkgM.f[i], where='mid', label=label + ' data')
    plt.step(v[i], fitpkgM.e[i], alpha=0.5, color=ln.get_color(), where='mid', label=label + ' data err')
    plt.legend()
plot_simspec(0)
plot_simspec(1)
plot_simspec(10)
plot_simspec(-1)
## endregion


## region lightcurves
bin_edges = [-37, 69, 100, 250]
"""
# path_ref_spec = '../../../data/toi776/toi776_2021-06-04_stis_g140m_oei103010_full_x1d.fits'
path_ref_spec = '../../data/toi776/toi776_2021-06-04_stis_g140m_oei103010_full_x1d.fits'
ref_spec = table.Table.read(path_ref_spec)
plt.figure()
vdata = c2.w2v(ref_spec['WAVELENGTH'][0], wrest_Lya) - rv_star
plt.step(vdata, ref_spec['FLUX'][0], color='k', where='mid')
plt.plot(vgrid, oot_profile)
for v in bin_edges:
    plt.axvline(v, color='0.5', lw=0.5)
plt.title('lightcurve bins')
plt.xlim(bin_edges[0] - 20, bin_edges[-1] + 20)
# note how this makes it clear that the instrument LSF makes it tricky to say what part of the line the absorption
# is actually happening in. The code integrates the model absorption after convolving with the instrument LSF
# to make the comparison as apples to apples as possible.
"""
# the aperture used varies from epoch to epoch
# this changes how much flux we expect to fall within a given velocity range in the data
# so the get_lightcurve function computes expected *observed* out of transit fluxes
# based on the assumed line profile for each point
# these are what we should normalize by to account for aperture differences

data_lc, data_lc_err, sim_lc, oot_fluxes = fitpkgM.get_lightcurves(depths, bin_edges, scalefacs=(1.0, 1.0))
t = np.copy(fitpkgM.ph) # ph is planet "phase", actually offset from mid transit in hours
oot = t < -5
t[oot] += 35 # for nicer plotting
dt = fitpkgM.dt
fig, axs = plt.subplots(2,2)
fig.suptitle('lightcurves')
for i in range(len(bin_edges) - 1):
    ax = axs.ravel()[i]
    Fnorm = oot_fluxes[:, i]
    ratio_data = data_lc[:, i] / Fnorm
    ratio_err = data_lc_err[:, i] / Fnorm
    ratio_sim = sim_lc[:, i] / Fnorm
    ratio_out = np.mean(ratio_data[oot])
    norm_data = ratio_data / ratio_out
    norm_err = ratio_err / ratio_out # I'm ignoring the error on the OoT flux
    norm_sim = ratio_sim / ratio_out
    ax.errorbar(t, norm_data, yerr=norm_err, xerr=dt / 2, fmt='ko', ecolor='0.5', label='data')
    ax.plot(t, norm_sim, 'o', label='model')
    label = '({:.0f}, {:.0f}) km s-1'.format(bin_edges[i], bin_edges[i+1])
    ax.annotate(label, xy=(0.5, 0.95), xycoords='axes fraction', ha='center', va='top')
## endregion

## region lightcurve-based chi2
band = [-37, 69]
chi2 = fitpkgM.compute_chi2_with_partial_band_lightcurve(depths, band, scalefacs=(1.0, 1.0))
# endregion

# endregion

# endregion G140M


###########################
# region G140L

# region preliminaries
# load data
path_aggregated_data = 'pubdata/g140l_aggregated_data_in_system_frame.fits'
data = table.Table.read(path_aggregated_data)

# make planet c's phases the ones that get used
data['pha'] = data['pha_c']
data['phb'] = data['phb_c']
data['ph'] = data['ph_c']
data.sort('ph')

# we only want data from the last epoch of observations, which happened starting on MJD 2459933
keep = data['t'] > 2459933
data = data[keep]

# making a wavelength grid to cover as much of the lya line as possible while avoiding Si III at 1206 Å
vgrid = np.arange(-1300, 2000+1, 1)
wgrid = c2.v2w(vgrid, wrest_Lya)

# I'll use the same time grid and simulated data as before, but I'll need to compute new depths
# for the new wavelength grid
wave_depth_factor = np.exp(-(vgrid - offset)**2/2/sig**2)
depths = time_depth[:,None] * wave_depth_factor[None,:]

# endregion

# region compute chi2 for the fake transit model and the G140L Lya data

## region generate the goods for fitting
path_oot_line_profile = 'pubdata/lya_fit_rescaled_to_g140l_epoch.ecsv'
line = table.Table.read(path_oot_line_profile)
spec = stis.g140l
fitpkgL = c2.FitPackage(wgrid, tgrid, spec, data, line, wrest=1215.67, epoch_divisions=None)
# Note that we cannot normalize these spectra since they're unresolved -- the absorption signal will get spread out
# so there is no "safe" range. Thankfully the way the observations were taken mostly mitigates the systematic we
# want to avoid.
## endregion

print(fitpkgL.wgrid)

## region simulate observations of spectra and show compared to data
sim_data = fitpkgL.simulate_spectra(depths, scalefacs=(1.0,))
v = (np.array(fitpkgL.w) - wrest_Lya)/wrest_Lya * c
def plot_simspec(i):
    label = 't = {:.1f} h'.format(fitpkgL.ph[i])
    plt.figure()
    ln, = plt.step(v[i], sim_data[i], where='mid', label=label + ' sim')
    ln, = plt.step(v[i], fitpkgL.f[i], where='mid', label=label + ' data')
    plt.step(v[i], fitpkgL.e[i], alpha=0.5, color=ln.get_color(), where='mid', label=label + ' data err')
    plt.legend()
plot_simspec(3)
plot_simspec(12)
plot_simspec(-1)

# I suspect the wavelength solution for the G140L data is off by one pixel,
# but it's not a concern since I'm integrating the full line

## endregion


## region show what flux will be binned in the lightcurve
# in this case, the functions we will use will integrate across the full wavelength grid
bin_edges = c2.w2v(wgrid[[0,-1]], wrest_Lya)
"""
# path_ref_spec = '../../../data/toi776/toi776_2022-12-21_stis_g140l_oeoo39010_full_x1d.fits'
path_ref_spec = '../../data/toi776/toi776_2022-12-21_stis_g140l_oeoo39010_full_x1d.fits'
ref_spec = table.Table.read(path_ref_spec)
plt.figure()
vdata = c2.w2v(ref_spec['WAVELENGTH'][0], wrest_Lya) - rv_star
plt.step(vdata, ref_spec['FLUX'][0], color='k', where='mid')
oot_profile = np.interp(wgrid, line['w'], line['y'])
plt.plot(vgrid, oot_profile)
for v in bin_edges:
    plt.axvline(v, color='0.5', lw=0.5)
plt.title('lightcurve bins')
plt.xlim(bin_edges[0] - 1000, bin_edges[-1] + 1000)
## endregion
"""

## region show a comparison lightcurve
# in this case, this is exactly what the chi2 will be computed from
data_lc, data_lc_err, sim_lc = fitpkgL.get_lightcurve_full_band(depths, scalefacs=(1.0,))
plt.figure()
t = fitpkgL.ph
dt = fitpkgL.dt
plt.errorbar(t, data_lc, data_lc_err, dt/2, fmt='o', label='data')
plt.plot(t, sim_lc, 'o', label='model')
## endregion

## getting the actual chi2 value is easy
chi2 = fitpkgL.compute_chi2_with_full_band_lightcurve(depths, scalefacs=(1.0,))

## endregion

#plt.show()


# endregion G140L
