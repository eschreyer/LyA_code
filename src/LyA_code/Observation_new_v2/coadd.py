import os.path
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy import constants as const

import muscles as ml
import spectralPhoton as sp

import phase
import gj436b

path_data = Path('data/')
path_coadd = path_data / 'gj436_g140m_oot_coadd.spec'


def get_oot_files():
    x1dfiles = list(path_data.glob('*corrected_x1d.fits'))

    oot_files = []
    for file in x1dfiles:
        x1d = fits.open(file)
        ext = x1d[1]
        t, ph = phase.get_x1d_ext_timeinfo(ext)
        ph = ph[1].to_value('h')
        if (ph < -10) | (ph > 24):
            oot_files.append(file)

    return oot_files


def inspect_x1ds():
    files = get_oot_files()
    plt.figure()
    for file in files:
        spec = sp.Spectrum.read_x1d(file, keep_header=True)
        spec.plot(label=spec.meta['rootname'])

    plt.legend()


def make_coadd():
    files = get_oot_files()
    spectbls = [ml.io.readfits(str(file), 'hst', 'sts', 'gj436')[0] for file in files]
    apertures = [fits.getval(file, 'aperture') for file in files]
    print("Apertures used are: ")
    print(apertures)
    aps, cnts = np.unique(apertures, return_counts=True)
    aperture = apertures[np.argmax(cnts)]
    coadd = ml.reduce.coadd(spectbls, savefits=False, maskbaddata=False)
    coadd['w0'] *= (1 - gj436b.rv/const.c).to_value('')
    coadd['w1'] *= (1 - gj436b.rv/const.c).to_value('')
    coadd = sp.Spectrum.read_muscles(coadd)
    coadd.meta['aperture'] = aperture
    if os.path.exists(path_coadd):
        answer = input("overwrite existing coadd? (y/n) ")
        if answer == 'n':
            return coadd
    coadd.write(str(path_coadd), overwrite=True)
    return coadd