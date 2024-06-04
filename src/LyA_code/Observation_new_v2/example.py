import numpy as np
from matplotlib import pyplot as plt
from astropy import table

import chi2_gj436 as c2

wrest_Lya = 1215.67
c = 3e5

# region fake a simulated transit
## region transit profile
max_depth = 0.5
tgrid = np.linspace(-10, 50, 1000)
sig_pre = 2.
sig_post = 10
time_depth = np.zeros_like(tgrid)
pre = tgrid < 0
post = tgrid > 0
time_depth[pre] = max_depth * np.exp(-tgrid[pre]**2/2/sig_pre**2)
time_depth[post] = max_depth * np.exp(-tgrid[post]**2/2/sig_post**2)

plt.figure()
plt.plot(tgrid, 1 - time_depth)
plt.title('transit time profile')
plt.ylim(-0.1, 1.1)
## endregion


## region velocity profile
# NOTE grid should either extend until 0 absorption on each side or bout 500 km/s beyond the data being fit
# the "aggregated_data" file just includes about 400 km/s of data to either side of the Lya rest wavelength
# hence the -1000 - 1000 km/s grid below
# but this can all be adjusted
vgrid = np.arange(-1000, 1000, 1)
sig = 50
offset = -75
wave_depth_factor = np.exp(-(vgrid - offset)**2/2/sig**2)
wgrid = (vgrid/3e5 + 1)*1215.67

plt.figure()
plt.plot(vgrid, wave_depth_factor)
plt.title('transit multiplication factor v wavelength')
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
plt.title('transit depth v wavelength (x-axis) vs time (blue->red)')
## endregion


## region check model transit spectra
oot_model = table.Table.read('data/coadd_fit.ecsv')
oot_profile = np.interp(wgrid, oot_model['w'], oot_model['y'])
model_spectra = oot_profile[None,:] * (1 - depths)
sparse_spectra = model_spectra[::step, :]
plt.figure()
for spec, color in zip(sparse_spectra, colors):
    plt.plot(wgrid, spec, color=color, alpha=0.5)
plt.title('model spectra vs time (blue->red)')
## endregion
# endregion


# region compute chi2 for the fake transit model
## region generate the goods for fitting
the_goods = c2.make_transit_chi2_tools(wgrid, tgrid)
oot_profile, oot_data, transit_data, simulate_spectra, get_lightcurves, compute_chi2 = the_goods
## endregion


## region simulate observations of spectra and show compared to data
t, dt, w, we, scaled_data, scaled_err, sim_data, sim_err = simulate_spectra(depths, norm_rng=[50, 200])
v = (np.array(w) - wrest_Lya)/wrest_Lya * c
def plot_simspec(i, label):
    plt.figure()
    ln, = plt.step(v[i], sim_data[i], where='mid', label=label + ' sim')
    plt.step(v[i], sim_err[i], alpha=0.5, color=ln.get_color(), where='mid', label=label + ' sim err')
    ln, = plt.step(v[i], scaled_data[i], where='mid', label=label + ' data')
    plt.step(v[i], scaled_err[i], alpha=0.5, color=ln.get_color(), where='mid', label=label + ' data err')
    plt.legend()
plot_simspec(0, 't = -8.4 h')
plot_simspec(41, 't ~= 0 h')
plot_simspec(-1, 't = 9.2 h')
## endregion


## region compute chi2
chi2, dof = compute_chi2(depths, norm_rng=[50,200])
# Notes on the chi2 estimate:
# - the degrees of freedom are just the number of data points compared. The true dof will be fewer
#    because the spectra are normalized
# - poissonian errors are estimated based on the model predicted flux and the instrument-estimated background count rate
## endregion


## region lightcurves
# lightcurves are intended just for visualizing the data and model, not for fitting
# the chi2 values computed above use the full spectral resultion, comparing simulated spectra to observed spectra
# so that nothing potentially valuable is thrown away
bin_edges = [-200, -150, -100, -50, 0]
plt.figure()
vdata = (oot_data['w']/wrest_Lya - 1)*c
plt.step(vdata, oot_data['y'], color='k', where='mid')
plt.plot(vgrid, oot_profile)
for v in bin_edges:
    plt.axvline(v, color='0.5', lw=0.5)
plt.title('lightcurve bins')
# note how this makes it clear that the instrument LSF makes it tricky to say what part of the line the absorption
# is actually happening in. The velocities we integrate are actually kind of ambiguous. The code I wrote integrates
# the model absorption after convolving with the instrument LSF to make the comparison as apples to apples as possible.

t, dt, data_lc, data_lc_err, sim_lc, oot_fluxes = get_lightcurves(depths, bin_edges, norm_rng=[50,200])
fig, axs = plt.subplots(2,2)
for i, ax in enumerate(axs.ravel()):
    Fout = oot_fluxes[i]
    scaled_data = data_lc[:, i] / Fout
    scaled_err = data_lc_err[:, i] / Fout # assuming the error on the OoT flux is insignificant
    normsim = sim_lc[:,i]/Fout
    ax.errorbar(t, scaled_data, yerr=scaled_err, xerr=dt / 2, fmt='ko', ecolor='0.5', label='data')
    ax.plot(t, normsim, 'o', label='model')
    label = '({:.0f}, {:.0f}) km s-1'.format(bin_edges[i], bin_edges[i+1])
    ax.annotate(label, xy=(0.5, 0.95), xycoords='axes fraction', ha='center', va='top')
##

# endregion