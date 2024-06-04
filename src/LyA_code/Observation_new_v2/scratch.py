import glob

from astropy.io import fits
import numpy as np
from scipy.stats import gaussian_kde
from scipy.optimize import minimize
from tqdm import tqdm
from astropy import table
from astropy import units as u
from matplotlib import pyplot as plt

from spectralPhoton import Spectrum


def explore_geocoronal_breathing_correlation():
    files = glob.glob('data/*tag.fits')
    aperture = "52X0.05"
    tbin_target = 500

    class custom_kde(gaussian_kde):
        def __init__(self, dataset, covariance):
            self.covariance = covariance
            super().__init__(dataset, bw_method=1.0, weights=None)

        def _compute_covariance(self):
            # Copied from the base gaussian_kde class, except for the covariance part
            self.inv_cov = np.linalg.inv(self.covariance)

    root, mjda, mjdb, lya_red_ctrate, gc_width = [], [], [], [], []
    for file in tqdm(files):
        h = fits.open(file)
        if aperture not in h[0].header['aperture']:
            continue
        data = h[1].data
        t, x, y = data['time'], data['axis1'], data['axis2']
        yedges = np.linspace(0, 2048, 50)

        # rectify the image
        # find x coord of geocoronal across y steps
        xgrid = np.arange(760, 840, 0.5)
        gx, gy, zmxs, widths = [], [], [], []
        for ya, yb in zip(yedges[:-1], yedges[1:]):
            keep = (y >= ya) & (y < yb)
            xx = x[keep]
            kernel = custom_kde(xx, covariance=np.diag([2]))
            zkde = kernel(xgrid)
            # save this to remove gaps
            imx = np.argmax(zkde)
            zmx = zkde[imx]
            zmxs.append(zmx)
            # save this to remove lya
            left = slice(0, imx)
            xleft = np.interp(zmx/2, zkde[left], xgrid[left])
            right = slice(imx, None)
            xright = np.interp(zmx/2, zkde[right][::-1], xgrid[right][::-1])
            width = xright - xleft

            result = minimize(lambda x: -kernel(x), xgrid[imx])
            if not result.success:
                continue
            widths.append(width)
            gx.append(result.x[0])
            gy.append((ya + yb)/2)

        # get rid of pts with low maxes to remove gaps
        gx, gy, zmxs, widths = map(np.asarray, (gx ,gy, zmxs, widths))
        good = zmxs > np.median(zmxs)/2
        gx, gy, widths = gx[good], gy[good], widths[good]

        # get rid of pt with greatest width to remove Lya
        imx = np.argmax(widths)
        gx = np.delete(gx, imx)
        gy = np.delete(gy, imx)

        # fit and remove outliers
        while True:
            p = np.polyfit(gy, gx, 1)
            xln = np.polyval(p, gy)
            diffs = gx - xln
            std = np.std(gx - xln)
            good = diffs < 3*std
            if np.all(good):
                break
            gx, gy = gx[good], gy[good]

        # rotate pts
        run = 1
        rise = -p[0]
        hypot = np.sqrt(run**2 + rise**2)
        rotation_matrix = np.matrix([[run/hypot, rise/hypot], [-rise/hypot, run/hypot]])
        xr, yr = np.dot(rotation_matrix, [x, y])
        xr, yr = map(np.asarray, (xr, yr))
        xr, yr = xr[0], yr[0]

        # find center of gc trace
        nkeep = 10000
        skip = int(round(len(xr)/nkeep))
        kernel = custom_kde(xr[::skip], np.diag([2]))
        zkde = kernel(xgrid)
        imx = np.argmax(zkde)
        xmx = xgrid[imx]

        # find lya so I can mask it later
        xlyaa = xmx + 10
        xlyab = xmx + 30
        lya_search_mask = (xr > xlyaa) & (xr < xlyab)
        kernel = custom_kde(yr[lya_search_mask], np.diag([20]))
        ygrid = np.arange(0, 2048, 0.5)
        zkde = kernel(ygrid)
        zlya = ygrid[np.argmax(zkde)]

        # loop through time steps recording lya flux and gc width
        Texp = h[1].header['exptime']
        n = int(round(Texp/tbin_target))
        Ta, Tb = h[2].data['start'][0], h[2].data['stop'][0]
        tedges = np.linspace(Ta, Tb, n+1)
        dt = tedges[1] - tedges[0]
        xgrid = np.arange(xmx - 25, xmx + 25, 0.1)
        x_bkgnd_left_a = xmx - 200
        x_bkgnd_left_b = xmx - 100
        x_bkgnd_right_a = xmx + 100
        x_bkgnd_right_b = xmx + 200
        dx_bkgnd = (x_bkgnd_left_b - x_bkgnd_left_a) + (x_bkgnd_right_b - x_bkgnd_right_a)
        for ta, tb in zip(tedges[:-1], tedges[1:]):
            root.append(h[0].header['rootname'])
            mjda.append(h[1].header['expstart'] + ta/3600/24)
            mjdb.append(h[1].header['expend'] + tb/3600/24)

            tmask = (t >= ta) & (t < tb)
            xx, yy = xr[tmask], yr[tmask]

            # get lya count rate
            in_lya = ((yy > zlya - 15) & (yy < zlya + 15)
                      & (xx > xlyaa) & (xx < xlyab))
            lya_red_cts = np.sum(in_lya)
            lya_red_ctrate.append(lya_red_cts/dt)

            # get width of geocoronal line
            out_lya = (yy < zlya - 30) | (yy > zlya + 30)
            # get background density
            bkgnd_mask = ((((xx > x_bkgnd_left_a) & (xx < x_bkgnd_left_b))
                          | ((xx > x_bkgnd_right_a) & (xx < x_bkgnd_right_b)))
                          & out_lya)
            bkgnd_density = np.sum(bkgnd_mask)/dx_bkgnd/np.sum(out_lya)
            # get background subtracted KDE estimate
            kernel = custom_kde(xx[out_lya], np.diag([2]))
            zkde = kernel(xgrid)
            z = zkde - bkgnd_density
            imx = np.argmax(z)
            zmx = z[imx]
            left = slice(0, imx)
            xleft = np.interp(zmx / 2, z[left], xgrid[left])
            right = slice(imx, None)
            xright = np.interp(zmx / 2, z[right][::-1], xgrid[right][::-1])
            width = xright - xleft
            gc_width.append(width)

    tbl = table.Table((mjda, mjdb, root, lya_red_ctrate, gc_width),
                      names='mjda mjdb root lya_red_ctrate gc_width'.split())


def explore_correlation_with_temperatures():
    lya_rng = (1215.85, 1216.4)*u.AA
    path = "data/ocyh03040_spt.fits"
    h = fits.open(path)
    keys = list(filter(lambda key: 'dgC' in h[1].header.comments[key], list(h[1].header.keys())))

    x1ds = glob.glob('data/*corrected_x1d.fits')
    data = dict(lya_fluxes=[], lya_errs=[], mjda=[], mjdb=[])
    for key in keys:
        data[key] = []
    for file in x1ds:
        sp = Spectrum.read_x1d(file, keep_header=True)
        flux, err = sp.integrate(lya_rng)
        data['lya_fluxes'].append(flux)
        data['lya_errs'].append(err)
        data['mjda'].append(sp.meta['expstart'])
        data['mjdb'].append(sp.meta['expend'])

        sptfile = file.replace('corrected_x1d', 'spt')
        h = fits.open(sptfile)
        for key in keys:
            data[key].append(h[1].header[key])

    # normalize lya for each different epoch of data
    mjd_edges = [0, 55500, 56500, 57000, 57480, 57500, 58000]
    data = table.Table(data)
    data['lya_norm'] = 0.0
    data['lya_norm_err'] = 0.0
    data['epoch_index'] = -1
    ab = zip(mjd_edges[:-1], mjd_edges[1:])
    for i, (a, b) in enumerate(ab):
        mask = (data['mjda'] > a) & (data['mjdb'] < b)
        norm = np.mean(data['lya_fluxes'][mask])
        data['lya_norm'][mask] = data['lya_fluxes'][mask]/norm
        data['lya_norm_err'][mask] = data['lya_errs'][mask]/norm
        data['epoch_index'][mask] = i

    # now make a function to plot
    def plotstuff(i0, i1):
        for key in keys[i0:i1]:
            plt.figure()
            for i in range(6):
                mask = data['epoch_index'] == i
                plt.errorbar(data[key][mask], data['lya_norm'][mask], data['lya_norm_err'][mask], fmt='.')
                plt.xlabel(key)

    return data, plotstuff


def red_wing_variability(va, vb):
    norm_rng_v = np.array((va, vb))
    norm_rng_w = (norm_rng_v/3e5 + 1) * 1215.67 * u.AA
    cgs = u.Unit('erg s-1 cm-2 AA-1')
    tbl = table.Table.read('data/g140m_aggregated_data.fits')
    dw = np.mean(np.diff(tbl['we'][0]))
    newbins = np.arange(1215.7, 1216.5+dw, dw) * u.AA
    tbl['norm'] = np.zeros((len(tbl), len(newbins)-1))
    tbl['norm_err'] = np.zeros((len(tbl), len(newbins)-1))
    for i, row in enumerate(tbl):
        spec = Spectrum(None, row['f']*cgs, row['e']*cgs, wbins=row['we']*u.AA)
        F, E = spec.integrate(norm_rng_w)
        print("{:.2f}".format(E/F))
        newspec = spec.rebin(newbins)
        tbl['norm'][i] = newspec.y/F
        tbl['norm_err'][i] = newspec.e/F

    avg_spec = np.mean(tbl['norm'], 0)
    diffs = tbl['norm'] - avg_spec[None,:]
    chi_terms = diffs**2/tbl['norm_err']**2
    chi2 = np.sum(chi_terms)
    dof = diffs.size
    chi2_std = np.sqrt(2*dof)
    # this isn't really correct because normalizing by a noisy range could introduce
    # more variability that isn't accounted for, but it at least gives us a sense

    return tbl, chi2, dof, chi2_std