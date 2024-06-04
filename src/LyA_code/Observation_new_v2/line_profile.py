import time
import pickle
import os

from astropy import units as u
from astropy import constants as const
from astropy import table
import numpy as np
from scipy.special import wofz
from scipy.optimize import minimize
from matplotlib import pyplot as plt
import emcee
from tqdm import tqdm

import chi2_gj436 as c2

#region parameters you probably don't care about
class Lya(object):
    wlab_H = 1215.67*u.AA
    wlab_D = 1215.34*u.AA
    D_H = 1.5e-5  # from bourrier+ 2017 (kepler 444)
    f = 4.1641e-01
    A = 6.2649e8 / u.s
    mH, mD = 1 * u.u, 2 * u.u
#endregion


path_coadd = 'data/gj436_g140m_oot_coadd.spec'
path_chain = 'data/lya_profile_fit_chain.pickle'


def w2v(w):
    return (w/Lya.wlab_H.value - 1)*const.c.to('km/s').value


def v2w(v):
    return (v/const.c.to('km/s').value + 1)*Lya.wlab_H.value


def doppler_shift(w, velocity):
    return (1 + velocity/const.c)*w


def voigt(x, gauss_sigma, lorentz_FWHM):
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian component HWHM alpha.

    """
    sigma = gauss_sigma
    gamma = lorentz_FWHM/2.0
    return np.real(wofz((x + 1j*gamma)/sigma/np.sqrt(2))) / sigma /np.sqrt(2*np.pi)


def voigt_xsection(w, w0, f, gamma, T, mass, b=None):
    """
    Compute the absorption cross section using hte voigt profile for a line.

    Parameters
    ----------
    w : astropy quantity array or scalar
        Scalar or vector of wavelengths at which to compute cross section.
    w0: quanitity
        Line center wavelength.
    f: scalar
        Oscillator strength.
    gamma: quantity
        Sum of transition rates (A values) out of upper and lower states. Just Aul for a resonance line where only
        spontaneous decay is an issue.
    T: quantity
        Temperature of the gas. Can be None if you provide a b value instead.
    mass: quantity
        molecular mass of the gas
    b : quantity
        Doppler b value (in velocity units) of the line
    Returns
    -------
    x : quantity
        Cross section of the line at each w.
    """

    nu = const.c / w
    nu0 = const.c / w0
    if T is None:
        sigma_dopp = b/const.c*nu0/np.sqrt(2)
    else:
        sigma_dopp = np.sqrt(const.k_B*T/mass/const.c**2) * nu0
    dnu = nu - nu0
    gauss_sigma = sigma_dopp.to(u.Hz).value
    lorentz_FWHM = (gamma/2/np.pi).to(u.Hz).value
    phi = voigt(dnu.to(u.Hz).value, gauss_sigma, lorentz_FWHM) * u.s
    x = np.pi*const.e.esu**2/const.m_e/const.c * f * phi
    return x.to('cm2')


def voigt_emission(w, w0, gamma, b):
    """
    Compute a voigt emission profile, normalized so that it will integrate to unity.

    Parameters
    ----------
    w : astropy quantity array or scalar
        Scalar or vector of wavelengths at which to compute cross section.
    w0: quanitity
        Line center wavelength.
    gamma: quantity
        Sum of transition rates (A values) out of upper and lower states. Just Aul for a resonance line where only
        spontaneous decay is an issue.
    b : quantity
        Doppler b value (in velocity units) of the line
    Returns
    -------
    x : quantity
        Cross section of the line at each w.
    """
    nu = const.c / w
    nu0 = const.c / w0
    sigma_dopp = b/const.c*nu0/np.sqrt(2)
    dnu = nu - nu0
    gauss_sigma = sigma_dopp.to(u.Hz).value
    lorentz_FWHM = (gamma/2/np.pi).to(u.Hz).value
    phi_nu = voigt(dnu.to(u.Hz).value, gauss_sigma, lorentz_FWHM) * u.s
    phi_lam = phi_nu * (const.c/w**2)
    return phi_lam.to(w.unit**-1)


def transmission(w, rv, logNh, T):
    Nh = 10**logNh
    w *= u.AA
    w0s = doppler_shift(u.Quantity((Lya.wlab_H, Lya.wlab_D)), rv * u.km / u.s)
    xsections = [voigt_xsection(w, w0, Lya.f, Lya.A, T * u.K, m) for w0, m in zip(w0s, (Lya.mH, Lya.mD))]
    tau = xsections[0]*Nh/u.cm**2 + xsections[1]*Nh/u.cm**2*Lya.D_H
    return np.exp(-tau)


def emission(w, rv, bs, Is):
    w *= u.AA
    w0 = Lya.wlab_H * (1 + rv/const.c.to_value('km s-1'))
    ys = []
    for b, I in zip(bs, Is):
        y = voigt_emission(w, w0, Lya.A, b * u.km / u.s)
        y = y.value * I
        ys.append(y)
    y = np.sum(ys, 0)
    return y


def absorbed_profile(w, rv_lya, bs, Is, rv_ism, logNh, T_ism):
    y = emission(w, rv_lya, bs, Is)
    trans = transmission(w, rv_ism, logNh, T_ism)
    return y*trans


def fitting_tools(obs_w, obs_f, obs_e, mask, aperture, dw_mod=0.01):
    mod_w = np.arange(obs_w.min(), obs_w.max() + dw_mod, dw_mod)
    observe = c2.fast_observe_function(obs_w, mod_w, aperture)
    check_peak = (mod_w < 1216.5) & (mod_w > 1215.5)

    def parse_params(params):
        rv_lya = params[0]
        bs = params[1:3]
        Is = params[3:5]
        rv_ism, Nh, Tism = params[-3:]
        return rv_lya, bs, Is, rv_ism, Nh, Tism

    def loglike(params):
        args = parse_params(params)
        rv_lya, bs, Is, rv_ism, Nh, Tism = args

        # priors
        should_be_pos = bs + Is + [Tism]
        not_pos = [x <= 0 for x in should_be_pos]
        if any(not_pos):
            return -np.inf

        if bs[0] < bs[1]:
            return -np.inf

        # model
        mod_f = absorbed_profile(mod_w, *args)
        mod_f = mod_f.value
        if mod_f[check_peak].max() < 1e-14:
            return -np.inf
        if mod_f[check_peak].max() > 1e-12:
            return -np.inf
        mod_f_obs = observe(mod_f)
        terms = -(obs_f - mod_f_obs)**2/2/obs_e**2
        result = np.sum(terms[mask])
        if np.isnan(result):
            # print("NaN output for params. Providing -np.inf instead.")
            # print(params)
            return -np.inf
        return result

    def profile(w, params):
        args = parse_params(params)
        return absorbed_profile(w, *args)

    tools = dict(loglike=loglike,
                 mod_w=mod_w,
                 observe=observe,
                 profile=profile)

    return tools


def sample_to_convergence(sampler, state, min_steps=1000, timeout=3600, max_steps=1e5, sample_factor=100, dtau_rel=0.01):
    result = {}
    nsteps = []
    autocorrs = []
    old_tau = np.inf
    print('Sampling to convergence, min {} steps.'.format(min_steps))
    t0 = time.time()
    dt_check = 5.
    lastcheck = 0
    try:
        for sample in sampler.sample(state, iterations=int(max_steps)):

            dt = time.time() - t0
            interval = dt // dt_check

            if interval > lastcheck:
                nstep = sampler.iteration
                tau = sampler.get_autocorr_time(tol=0)
                nsteps.append(nstep)
                autocorrs.append(np.mean(tau))
                factor = nstep / tau
                medians = np.median(sampler.flatchain, 0)
                with np.printoptions(precision=2, linewidth=150):
                    print('\telapsed time {:.1f} min | step {} | tau {:.2f} | sample factor {:.1f}/{:.0f} | medians {}'
                          ''.format(dt/60., nstep, np.max(tau), np.min(factor), sample_factor, medians),
                          end='\r', flush=True)

                uncorrelated = np.all(factor > sample_factor)
                tau_converged = np.all(np.abs(tau - old_tau)/tau < dtau_rel)
                if uncorrelated and tau_converged and nstep > min_steps:
                    print('\nConverged.')
                    break
                if nstep > max_steps:
                    print('\nMax steps reached.')
                    break
                if dt > timeout:
                    print('\nTimed out.')
                    break
                old_tau = tau
                lastcheck = interval
        result['success'] = True
        result['error'] = None
    except Exception as err:
        result['success'] = False
        result['error'] = err
    finally:
        result['nsteps'] = nsteps
        result['autocorrs'] = autocorrs
        return result


def coadd_plus_fit_tools():
    coadd_tbl = table.Table.read(path_coadd, format='ascii.ecsv')

    keep = (coadd_tbl['w'] > 1205) & (coadd_tbl['w'] < 1225)
    coadd_tbl = coadd_tbl[keep]

    w = coadd_tbl['w'].quantity.to_value('AA')
    f = coadd_tbl['y'].quantity.to_value('erg s-1 cm-2 AA-1')
    e = coadd_tbl['err'].quantity.to_value('erg s-1 cm-2 AA-1')

    nofit_mask = ((w > 1218.11) & (w < 1219.02))
    fit_mask = ~nofit_mask

    aperture = coadd_tbl.meta['meta']['aperture']
    tools = fitting_tools(w, f, e, fit_mask, aperture)
    return coadd_tbl, tools


def fit_profile(test=False, timeout=3600*6):
    if test:
        nwalkers, sample_fac, dtau_rel, min_steps = 20, 20, 0.1, 100  # test
    else:
        nwalkers, sample_fac, dtau_rel, min_steps = 100, 100, 0.05, 1000  # production

    # region profile guesses
    rv_lya = -4.4
    b_broad = 219
    b_narrow = 72
    I_broad = 1.6e-14
    I_narrow = 2.5e-13
    rv_ism = -11.4
    logNh = 18.0
    Tism = 6700
    guess = [rv_lya, b_broad, b_narrow, I_broad, I_narrow, rv_ism, logNh, Tism]
    # endregion

    _, tools = coadd_plus_fit_tools()
    loglike = tools['loglike']
    neglike = lambda x: -loglike(x)

    if test:
        best = guess
    else:
        min_result = minimize(neglike, guess, method='Nelder-Mead', options=dict(maxfev=1e4))
        best = min_result.x # might not converge fully but it doesn't really matter

    mcmc_jitter = [3, 50, 10, best[3]/10, best[4]/10, 3, 0.2, 500]
    ndim = 8
    np.random.seed(20211220)
    state = np.tile(best, (nwalkers, 1)) + np.random.randn(nwalkers, ndim) * np.array(mcmc_jitter)[None, :]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, loglike)

    mc_result = sample_to_convergence(sampler, state, min_steps=min_steps, sample_factor=sample_fac,
                                   dtau_rel=dtau_rel, timeout=timeout)

    autocorr_trend = np.array((mc_result['nsteps'], mc_result['autocorrs']))
    nburn = int(autocorr_trend[1, -1] * 10)

    chain = sampler.get_chain(discard=nburn, flat=True)
    lnprob = sampler.get_log_prob(discard=nburn, flat=True)
    chain = chain[lnprob > -np.inf]
    max_chain_len = 50e6 / 8 / ndim
    if len(chain) > max_chain_len:
        thin = int(len(chain) / max_chain_len)
        chain = chain[::thin, :]

    chain_data = dict(chain=chain, autocorr=autocorr_trend, best=best)

    if os.path.exists(path_chain):
        answer = input('Overwrite existing chain (y/n)? ')
        if answer == 'y':
            with open(path_chain, 'wb') as file:
                pickle.dump(chain_data, file)

    return chain_data, mc_result


def save_and_plot_coadd_fit(chain):
    n = int(1e4)
    coadd, tools = coadd_plus_fit_tools()
    profile = tools['profile']
    observe = tools['observe']
    mod_w = tools['mod_w']

    step = int(len(chain)/n)
    slim = chain[::step, :]
    ymc = []
    omc = []
    for params in tqdm(slim):
        y = profile(mod_w, params)
        ymc.append(y)
        omc.append(observe(y))

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('Wavelength (Å)')
    ax.set_ylabel('Flux (erg s-1 cm-2 Å-1)')
    data = ax.step(coadd['w'], coadd['y'], where='mid')

    ylo, ymed, yhi = np.percentile(ymc, [16, 50, 84], axis=0)
    olo, omed, ohi = np.percentile(omc, [16, 50, 84], axis=0)

    ax.fill_between(mod_w, ylo, yhi, color='C1', alpha=0.5, lw=0)
    mod, = ax.plot(mod_w, ymed, color='C1')

    ax.fill_between(coadd['w'], olo, ohi, color='C2', alpha=0.5, lw=0, step='mid')
    omod = ax.step(coadd['w'], omed, color='C2', where='mid')

    ax.set_xlim(1213.5, 1218)
    ax.legend((data, mod, omod), labels='Data, Model, Model x Inst'.split(', '))

    fig.savefig('coadd_fit.pdf')
    fig.savefig('coadd_fit.png', dpi=300)

    fit = table.Table((mod_w, ymed), names='w y'.split())
    path = 'coadd_fit.ecsv'
    if os.path.exists(path):
        answer = input('Overwrite {} (y/n)? ')
        if answer != 'y':
            return
    fit.write(path)

