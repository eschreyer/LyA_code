try:
    import stistools as stis
except ImportError:
    "Couldn't import stistools. You probably need to quit ipython, enter `conda activate astroconda', then restart the session."

print("Make sure to CD into the directory with all the data or calstis won't find the files.")

import os
from pathlib import Path
import shutil

from matplotlib import pyplot as plt
from astropy.io import fits
from astropy import table
from astropy import units as u
from astropy import constants as const
import numpy as np
from matplotlib import pyplot as plt

import phase
import gj436b


subexp_target = 500
bkoffst = 20
bksize = 20
x1d_params = dict(bk1offst=-bkoffst, bk2offst=bkoffst, bk1size=bksize, bk2size=bksize)
path_data = Path('.')
path_g140m_tracelocs = path_data / 'g140m_tracelocs.ecsv'
path_aggregated_data = path_data / 'g140m_aggregated_data.fits'

def copy_data_from_mast_dnld(dnld_dir):
    dnld_dir = Path(dnld_dir)
    files = list(dnld_dir.glob('*/*.fits'))
    for file in files:
        newpath = path_data / file.name
        shutil.copyfile(file, newpath)


def get_reference_files():
    import crds
    files = []
    for suffix in ['tag', 'flt', 'raw']:
        f = path_data.glob('*_{}.fits'.format(suffix))
        f = map(str, f)
        files.extend(f)
    crds.bestrefs.assign_bestrefs(files, sync_references=True, verbosity=10)

setenvs = """
setenv CRDS_PATH /Users/parke/crds_cache/
setenv CRDS_SERVER_URL https://hst-crds.stsci.edu
setenv iref  ${CRDS_PATH}/references/hst/iref/
setenv jref  ${CRDS_PATH}/references/hst/jref/
setenv oref  ${CRDS_PATH}/references/hst/oref/
setenv lref  ${CRDS_PATH}/references/hst/lref/
setenv nref  ${CRDS_PATH}/references/hst/nref/
setenv uref  ${CRDS_PATH}/references/hst/uref/
"""
setenvs = setenvs.split('\n')
for line in setenvs:
    if line != '':
        _, key, path = line.split()
        path = path.replace('${CRDS_PATH}', '/Users/parke/crds_cache')
        os.environ[key] = path


tracetbl = table.Table.read(path_g140m_tracelocs)
tracetbl.add_index('id')


def midpts(x):
    return (x[:-1] + x[1:])/2


def locate_traces(tbl=None):
    from mypy import plotutils as pu
    tagfiles = list(path_data.glob('*_tag.fits'))

    if tbl is None:
        ids, locs = [], []
    else:
        ids = tbl['id'].tolist()
        locs = tbl['y_trace'].tolist()
    for tf in tagfiles:
        h = fits.open(tf)
        plt.figure()
        plt.title(tf.name)
        id = h[0].header['rootname']
        if id in ids:
            continue
        ids.append(id)
        data = h[1].data
        x = data['AXIS1']
        y = data['AXIS2']
        bins = np.arange(0, 2049)
        z, xe, ye = np.histogram2d(x, y, bins)

        xm, ym = map(midpts, (xe, ye))
        pu.pcolor_reg(xm, ym, z.T**0.25)
        plt.plot(x, y, ',')
        plt.draw()
        plt.show()

        _ = input('Zoom to trace and then press any key to continue (q to quit).')
        if _ == 'q':
            return
        print('Click trace, then click off axes.')
        xy = pu.click_coords()
        loc = xy[0,1]/2
        locs.append(loc)

        plt.close()
        h.close()

    tbl = table.Table([ids, locs], names='id y_trace'.split())
    tbl.write(path_g140m_tracelocs)
    return tbl


def create_subexposures():
    tagfiles = list(path_data.glob('*tag.fits'))

    overwrite_consent = False
    for file in tagfiles:
        expt = fits.getval(file, 'exptime', ext=1)
        n = int(round(expt // subexp_target))
        if n == 0:
            raise("WTF there was an exposure that was less than 250 s. Go delete it I guess.")

        root = file.name.split('_')[0]
        outroot = '{}_subs'.format(root)

        rawfile = outroot + '_raw.fits'
        for ext in 'raw flt x2d x1d'.split():
            path = '{}_{}.fits'.format(outroot, ext)
            if os.path.exists(path):
                if not overwrite_consent:
                    answer = input("Remove and replace existing subexposure files? (y/n)")
                    if answer == 'n':
                        return
                    else:
                        overwrite_consent = True
                os.remove(path)

        stis.inttag.inttag(file, rawfile, rcount=n)

        wavecal = root + '_wav.fits'
        status = stis.calstis.calstis(rawfile, wavecal=wavecal, outroot=outroot)
        assert status == 0
        x1dpath = outroot + '_x1d.fits'
        fltpath =outroot + '_flt.fits'
        os.remove(x1dpath)

        traceloc = tracetbl.loc[root]['y_trace']
        stis.x1d.x1d(fltpath, x1dpath, a2center=traceloc, maxsrch=0, **x1d_params)


def correct_full_exposure_x1ds():
    fltfiles = list(path_data.glob('*_flt.fits'))
    fltfiles = list(filter(lambda s: 'subs' not in s.name, fltfiles))

    overwrite_consent = False
    for file in fltfiles:
        root = fits.getval(file, 'rootname')
        traceloc = tracetbl.loc[root]['y_trace']

        x1dpath = root + '_corrected_x1d.fits'

        if os.path.exists(x1dpath):
            if not overwrite_consent:
                answer = input("Remove and replace existing x1d files? (y/n)")
                if answer == 'n':
                    return
                else:
                    overwrite_consent = True
            os.remove(x1dpath)

        stis.x1d.x1d(str(file), x1dpath, a2center=traceloc, maxsrch=10, **x1d_params)


def aggregate_data(keep_n_pixels=50):
    """pack together all the relevant data.
    
    times are barycentric
    wavelengths are in the stellar rest frame
    """
    nkeep = keep_n_pixels // 2
    sub_x1ds = path_data.glob('*subs_x1d.fits')

    keys = 'ta tb t pha phb ph we w f e bs bkgnd fluxfac root aperture'.split()
    lsts = [[] for _ in range(len(keys))]
    data = dict(zip(keys, lsts))

    for file in sub_x1ds:
        x1d = fits.open(file)
        root = x1d[0].header['rootname']
        aperture = x1d[0].header['aperture']
        for ext in x1d[1:]:
            tbary, ph = phase.get_x1d_ext_timeinfo(ext)

            w, = ext.data['wavelength']
            w = (1 - gj436b.rv/const.c) * w
            i = np.arange(len(w))
            p = np.polyfit(i, w, 3)
            wa, wb = np.polyval(p, [i - 0.5, i + 0.5])

            # keep the central 100 pixels (about 5 AA)
            imid = np.argmin(np.abs(w - 1215.67))
            keep = slice(imid-nkeep, imid+nkeep)

            f, = ext.data['flux']
            e, = ext.data['error']
            b, = ext.data['background']
            n, = ext.data['net']
            g, = ext.data['gross']
            bs = np.abs(b/n) # might use for deciding what data to mask in the future
            T = ext.header['exptime']

            # make errors poissonian only
            fluxfac = f/n
            bad = n <= 0
            p = np.polyfit(i[~bad], fluxfac[~bad], 3)
            fluxfac = np.polyval(p, i)
            cnts = np.abs(g*T)
            cnts[np.abs(cnts) <= 1] = 1
            emod = np.sqrt(cnts)*fluxfac/T

            slim = [a[keep] for a in (wa, w, wb, f, emod, b, bs, fluxfac)]
            wa, w, wb, f, emod, b, bs, fluxfac = slim
            we = np.append(wa, wb[-1])

            data['ta'].append(tbary[0] * u.d)
            data['t'].append(tbary[1] * u.d)
            data['tb'].append(tbary[2] * u.d)

            data['pha'].append(ph[0])
            data['ph'].append(ph[1])
            data['phb'].append(ph[2])

            data['we'].append(we * u.AA)
            data['w'].append(w * u.AA)

            data['f'].append(f * u.Unit('erg s-1 cm-2 AA-1'))
            data['e'].append(emod * u.Unit('erg s-1 cm-2 AA-1'))
            data['bs'].append(bs * u.Unit(''))
            data['bkgnd'].append(b * u.Unit('ct s-1 AA-1'))
            data['fluxfac'].append(fluxfac * u.Unit('erg s-1 cm-2 AA-1 ct-1'))
            data['root'].append(root)
            data['aperture'].append(aperture)

    for key in keys[:-2]:
        data[key] = u.Quantity(data[key])

    tbl = table.Table(data)
    tbl['bs'].description = 'ratio of background to signal counts'
    tbl.meta['notes'] = "times are barycentric, wavelengths are stellar rest frame"
    if os.path.exists(path_aggregated_data):
        answer = input('Overwrite {} (y/n)? '.format(path_aggregated_data))
        if answer != 'y':
            return tbl

    tbl.write(path_aggregated_data, overwrite=True)
    return tbl