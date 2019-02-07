"""
Deredshifts all provided spectra and find a range which is compatible to all spectra.
This range should ideally be equal or larger to the width of the generated spectra.
Otherwise, I will ahve to add some zeros (or interpolated values).
"""
import gc
import time
import h5py
import numpy as np
import pandas as pd
from astropy.io import fits

df = pd.read_csv('/fred/oz012/Bruno/data/BossGALOZListAllValues.csv')

start = 0.
"""
with open('min_max_ranges.csv', 'w', buffering=1) as f:
    f.write('Id,min,max,local_path\n')
    for i_spectrum in range(len(df.axes[0])):
        if i_spectrum%1000==0:
            print(time.time()-start)
            start = time.time()
            print('{} iteration'.format(i_spectrum), flush=True)

        with fits.open(df['local_path'].iloc[i_spectrum]) as hdu:
            lam_ = np.power(10, hdu[1].data['loglam']).astype(np.float32)
            rshift_ = df['redshift'].iloc[i_spectrum].astype(np.float32)
            lam_ = lam_ / (1 + rshift_)
            lam_min = min(lam_)
            lam_max = max(lam_)

            #if lam_min > curr_min:
            #    curr_min = lam_min
            #if lam_max < curr_max:
            #    curr_max = lam_max

        gc.collect()
        f.write(str(i_spectrum) + ',' + str(lam_min) + ',' + str(lam_max) + ',' +
                df['local_path'].iloc[i_spectrum] + '\n')
"""

paths = df['local_path'].values
N = len(paths) # Number of files
d = 5000 # size of spectra file

save_path = '/fred/oz012/Bruno/data/sdss_spectra.hdf5'
with h5py.File(save_path, 'w') as f:
    dset = f.create_dataset('spectra', (N, d, 2), dtype=np.float32) 
    counter = 0
    for i_spectrum in range(N):
        if i_spectrum%500==0:
            print('{} iterations. Time: {:.5g} Counter: {}'
                  .format(i_spectrum, time.time()-start, counter), flush=True)
            counter = 0
            start = time.time()
            gc.collect()

        with fits.open(paths[i_spectrum]) as hdu:
            flux_ = hdu[1].data['flux'].astype(np.float32)
            lam_ = np.power(10, hdu[1].data['loglam']).astype(np.float32)
            rshift_ = df['redshift'].iloc[i_spectrum].astype(np.float32)
            lam_ = lam_ / (1 + rshift_)
            if len(lam_)<d:
                counter += 1
                fill = d - len(lam_)
                lam_ = np.append(lam_, [0.]*fill)
                flux_ = np.append(flux_, [0.]*fill)
            dset[i_spectrum, :, 0] = lam_[:d]
            dset[i_spectrum, :, 1] = flux_[:d]

"""
with h5py.File('sdss_spectra.hdf5', 'r') as f:
    dset = f['spectra']
    print(np.min(dset[1,:,0][np.nonzero(dset[1,:,0])]))
    print(np.max(dset[1,:,0][np.nonzero(dset[1,:,0])]))
"""
    
