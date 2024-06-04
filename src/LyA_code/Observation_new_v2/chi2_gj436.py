import numpy as np
from matplotlib import pyplot as plt
from astropy import table
from astropy import constants as const
from scipy.interpolate import interp1d


wrest_Lya = 1215.67
c = const.c.to_value('km s-1')


# region load in STIS LSFs
path_to_file = ''
path_lsf = path_to_file + 'Observation_new_v2/LSF_G140M_1200.txt'
path_aggregated_data = path_to_file + 'Observation_new_v2/data/g140m_aggregated_data.fits' #change to synthetic data
path_oot_line_profile = path_to_file + 'Observation_new_v2/data/coadd_fit.ecsv'   #out of transit line profile
path_oot_data = path_to_file + 'Observation_new_v2/data/gj436_g140m_oot_coadd.spec' #out of transit data
lsf_raw = np.loadtxt(path_lsf, skiprows=2)
lsf_raw = table.Table(lsf_raw, names='pixel 52X0.1 52X0.2 52X0.5 52X2.0'.split())

# lsf seems to be measured on a grid that alternates between 0.125 and 0.126 pixel spacing. So I will just interpolate onto an even 0.125 spacing.
dpix = 0.125
_x = np.arange(-34.5, 34.5+dpix, dpix)
empty = np.zeros((len(_x), len(lsf_raw.colnames)))
lsf = table.Table(empty, names=lsf_raw.colnames)
lsf['pixel'] = _x
for key in lsf.colnames[1:]:
    y = np.interp(_x, lsf_raw['pixel'], lsf_raw[key])
    ynorm = np.trapz(y)
    lsf[key] = y/ynorm
lsf['52X0.05'] = lsf['52X0.1']
n_lsf = len(lsf) // 2
# endregion


def v2w(velocities):
    """

    Parameters
    ----------
    velocities : km s-1

    Returns
    -------

    """
    return (1 + np.asarray(velocities)/c) * wrest_Lya


def interp_inst_waves(w, pixels):
    i = np.arange(len(w))
    p = np.polyfit(i, w, 3)
    return np.polyval(p, pixels)


def midpts(x):
    return (x[1:] + x[:-1])/2.


def cumtrapz(y, x):
    areas = midpts(y) * np.diff(x)
    result = np.cumsum(areas)
    result = np.insert(result, 0, 0)
    return result


def bin(x_edges, x, y):
    integral = cumtrapz(y, x)
    Iedges = np.interp(x_edges, x, integral)
    return np.diff(Iedges)


def rebin(new_edges, old_edges, y):
    dx = np.diff(old_edges)
    areas = y*dx
    integral = np.insert(np.cumsum(areas), 0, 0)
    Iedges = np.interp(new_edges, old_edges, integral)
    return np.diff(Iedges)/np.diff(new_edges)


def rebin_error(new_edges, old_edges, err):
    dx = np.diff(old_edges)
    E = err*dx
    V = E**2
    integral = np.insert(np.cumsum(V), 0, 0)
    Iedges = np.interp(new_edges, old_edges, integral)
    Vnew = np.diff(Iedges)
    return np.sqrt(Vnew) / np.diff(new_edges)


def cumtrapz_griddata(y, x):
    areas = midpts(y) * np.diff(x)[:, None]
    result = np.cumsum(areas, axis=0)
    result = np.vstack((np.zeros_like(result[0]),
                        result))
    return result


def intergolate(x_bin_edges, xin,yin):
    I = cumtrapz(yin, xin)
    Iedges = np.interp(x_bin_edges, xin, I)
    y_bin_avg = np.diff(Iedges)/np.diff(x_bin_edges)
    return y_bin_avg


def observe(inst_w, mod_w, mod_f, aperture, return_we=False):
    dw = np.diff(mod_w)
    if np.any(dw > 0.0051):
        raise ValueError('Spectrum not well resolved.')

    # get things on the right grids and convolve
    inst_lsf_pixel_grid = np.arange(lsf['pixel'][0], lsf['pixel'][-1] + len(inst_w)-1, dpix)
    inst_lsf_w_grid = interp_inst_waves(inst_w, inst_lsf_pixel_grid)
    mod_f_interp = np.interp(inst_lsf_w_grid, mod_w, mod_f)
    mod_f_conv = np.convolve(mod_f_interp, lsf[aperture], mode='valid')
    mod_w_conv = inst_lsf_w_grid[n_lsf:-n_lsf]

    # integrate over the pixels
    inst_pixel_edges = np.arange(-0.5, len(inst_w), 1)
    inst_w_edges = interp_inst_waves(inst_w, inst_pixel_edges)
    msrd_f = intergolate(inst_w_edges, mod_w_conv, mod_f_conv)

    if return_we:
        return msrd_f, inst_w_edges
    return msrd_f


def fast_observe_function(inst_w, mod_w, aperture):
    # get things on the right grids to convolve
    inst_lsf_pixel_grid = np.arange(2*lsf['pixel'][0], 2*lsf['pixel'][-1] + len(inst_w)-1, dpix)
    inst_lsf_w_grid = interp_inst_waves(inst_w, inst_lsf_pixel_grid)
    mod_w_conv = inst_lsf_w_grid[n_lsf:-n_lsf]

    # construct instrument pixel grid
    inst_pixel_edges = np.arange(-0.5, len(inst_w), 1)
    inst_w_edges = interp_inst_waves(inst_w, inst_pixel_edges)

    def fast_observe(mod_f):
        mod_f_interp = np.interp(inst_lsf_w_grid, mod_w, mod_f)
        mod_f_conv = np.convolve(mod_f_interp, lsf[aperture], mode='valid')
        msrd_f = intergolate(inst_w_edges, mod_w_conv, mod_f_conv)
        return msrd_f

    return fast_observe


def make_transit_chi2_tools(wgrid, tgrid, transit_rng):
    """

    Parameters
    ----------
    wgrid : wavelength grid of model in AA
    tgrid : time grid of model in *hours*

    Returns
    -------

    """
    norm_rng = (50, 250) # km/s #hmmmm
    #transit_rng = (1, 31.4) # h

    # load out of transit line profile
    oot_model = table.Table.read(path_oot_line_profile)
    oot_profile = np.interp(wgrid, oot_model['w'], oot_model['y'])

    # load out of transit data
    oot_data = table.Table.read(path_oot_data, format='ascii.ecsv')
    w_out, dw, f_out = oot_data['w'], oot_data['dw'], oot_data['y']
    we_out = np.append(w_out - dw/2, w_out[-1] + dw[-1]/2.)

    # load, filter, and sort data
    transit_data = table.Table.read(path_aggregated_data)
    keep = (transit_data['ph'] > transit_rng[0]) & (transit_data['ph'] < transit_rng[1])
    transit_data = transit_data[keep]
    transit_data.sort('ph')

    # create functions to simulate observation of a model spectrum
    wgrids = []
    observe_functions = []
    for row in transit_data:
        w, ap = row['w'], row['aperture']
        obsfun = fast_observe_function(w, wgrid, ap)
        wgrids.append(wgrid)
        observe_functions.append(obsfun)
    transit_data['wgrid'] = wgrids
    transit_data['obsfun'] = observe_functions

    def simulate_spectra(depths, norm_rng=norm_rng):
        """

        Parameters
        ----------
        depths
        norm_rng : km s-1

        Returns
        -------
        simdata
        """
        # convert norm_rng to wavelength
        norm_waves = v2w(norm_rng)

        # compute out of transit flux in normalization range
        Fout = rebin(norm_waves, we_out, f_out)

        # average depths over the observations times
        integral_depth = cumtrapz_griddata(depths, tgrid)
        interp = interp1d(tgrid, integral_depth, axis=0)
        dt = transit_data['phb'] - transit_data['pha']
        avg_depth = (interp(transit_data['phb']) - interp(transit_data['pha']))/dt[:,None]

        # multiply by OOT profile to get transit-absorbed profiles
        absorbed_profiles = (1 - avg_depth) * oot_profile[None,:]

        # simulate observation of the absorbed profiles
        scaled_data, scaled_err, sim_data, sim_err = [], [], [], []
        for row, profile in zip(transit_data, absorbed_profiles):
            sim = row['obsfun'](profile)
            Fsim = rebin(norm_waves, row['we'], sim)
            normfac = Fout/Fsim
            sim_data.append(sim*normfac)
            sim_source_cntrate = sim/row['fluxfac']
            T = (row['tb'] - row['ta'])*3600*24
            sim_total_cnts = (row['bkgnd'] + sim_source_cntrate)*T
            sim_err_cnts = np.sqrt(sim_total_cnts)
            sim_err.append(sim_err_cnts/T*row['fluxfac'])

            Fdata = rebin(norm_waves, row['we'], row['f'])
            normfac = Fout/Fdata
            scaled_data.append(row['f']*normfac)
            scaled_err.append(row['e']*normfac)

        t = transit_data['ph']
        dt = transit_data['phb'] - transit_data['pha']

        return t, dt, transit_data['w'], transit_data['we'], scaled_data, scaled_err, sim_data, sim_err

    def get_lightcurves(depths, bin_edges, norm_rng):
        """

        Parameters
        ----------
        depths
        bin_edges : km s-1
        norm_rng : km s-1

        Returns
        -------
        t, dt, data_lc, data_lc_err, sim_lc
        """
        bin_edges_w = v2w(bin_edges)

        oot_fluxes = rebin(bin_edges_w, we_out, f_out)

        t, dt, w_list, we_list, scaled_data, scaled_err, sim_data, sim_err = simulate_spectra(depths, norm_rng)
        data_lc = []
        data_lc_err = []
        sim_lc = []
        for we, data, err, sim in zip(we_list, scaled_data, scaled_err, sim_data):
            data_fluxes = rebin(bin_edges_w, we, data)
            data_err = rebin_error(bin_edges_w, we, err)
            sim_fluxes = rebin(bin_edges_w, we, sim)
            data_lc.append(data_fluxes)
            data_lc_err.append(data_err)
            sim_lc.append(sim_fluxes)

        data_lc, data_lc_err, sim_lc = map(np.asarray, (data_lc, data_lc_err, sim_lc))

        return t, dt, data_lc, data_lc_err, sim_lc, oot_fluxes


    # make the function to compute a chi2 value given model-generated depth data on the supplied wavelength and time grid
    def compute_chi2(depths, norm_rng=norm_rng):
        _, _, _, _, scaled_data, scaled_err, sim_data, sim_err = simulate_spectra(depths, norm_rng)
        chi2 = 0
        dof = 0
        for data, err, sim in zip(scaled_data, sim_err, sim_data):
            terms = (data - sim)**2/err**2
            dof += len(terms)
            chi2 += np.sum(terms)
        return chi2, dof


    def compute_logL(depths, only_blue = False, weight_fluxes = False, norm_rng=norm_rng):
        t, dt, w_list, we_list, scaled_data, scaled_err, sim_data, sim_err = simulate_spectra(depths, norm_rng)

        #make into arrays

        scaled_data = np.array(scaled_data)
        sim_data = np.array(sim_data)
        sim_err = np.array(sim_err)

        #only take into account blue wing

        if only_blue == True:

            scaled_data = scaled_data[w_list < 1215.67]
            sim_err = sim_err[w_list < 1215.67]
            sim_data = sim_data[w_list < 1215.67]

        logL = 0

        for data, err, sim in zip(scaled_data, sim_err, sim_data):
            terms = -(1/2) * ((data - sim)**2/err**2 + np.log(2 * np.pi * err**2))
            logL += np.sum(terms)

        return logL



    return oot_profile, oot_data, transit_data, simulate_spectra, get_lightcurves, compute_chi2, compute_logL
