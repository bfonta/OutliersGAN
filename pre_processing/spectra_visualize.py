import glob
import random
import numpy as np
import pandas as pd
from astropy.io import fits
from src.utilities import PlotGenSamples
from matplotlib import pyplot as plt

nrows = 5
ncols = 1

path = '/fred/oz012/Bruno/data/spectra/qso_zWarning/*/*.fits'
df = pd.read_csv('/fred/oz012/Bruno/data/qso_zWarning_ListAllValues.csv')
g = glob.glob(path)
random.seed()
r = [np.random.randint(0, len(g)-1) for _ in range(nrows)]

p = PlotGenSamples(nrows=nrows, ncols=ncols)
flux, lam = ([] for _ in range(2))
for i in r:
    with fits.open(df['local_path'].iloc[i]) as hdu:
        flux_ = hdu[1].data['flux'].astype(np.float32)
        lam_ = np.power( 10, hdu[1].data['loglam'] ).astype(np.float32)
        rshift_ = df['redshift'].iloc[i].astype(np.float32)
        lam_ = lam_ / (1 + rshift_)
        lam.append(lam_)
        flux.append(flux_)
p.plot_spectra(samples=flux, lambdas=lam, fix_size=False, name='2')
