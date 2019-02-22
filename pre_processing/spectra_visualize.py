import numpy as np
import pandas as pd
from astropy.io import fits
from src.utilities import PlotGenSamples
from matplotlib import pyplot as plt

table_path = '/fred/oz012/Bruno/data/BossGALOZListAllValues.csv'
df = pd.read_csv(table_path)

nrows = 5
ncols = 1
arr = np.random.randint(0,len(df.axes[0]), size=(nrows))

p = PlotGenSamples(nrows=nrows, ncols=ncols)
flux, lam = ([] for _ in range(2))
for j,i in enumerate(arr):
    with fits.open(df['local_path'].iloc[i]) as hdu:
        flux_ = hdu[1].data['flux'].astype(np.float32)
        lam_ = np.power( 10, hdu[1].data['loglam'] ).astype(np.float32)
        rshift_ = df['redshift'].iloc[i].astype(np.float32)
        lam_ = lam_ / (1 + rshift_)
        
        #selection = (lam_>3100) 
        lam.append(lam_)#[selection][:3500])
        flux.append(flux_)#[selection][:3500])
p.plot_spectra(flux, lam, '2')
