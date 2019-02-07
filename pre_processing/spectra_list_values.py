"""
The ordering of the 'files' and 'files_bits' lists is the same.
This is required for the extracted parameters to match the correct file name.
"""
import os
import glob
import numpy as np
import pandas as pd

df = pd.read_csv('/fred/oz012/Bruno/data/BOSS_GALLOZ.csv', skiprows=0)
df = df[['plate', 'mjd', 'fiberid', 'redshift', 'ra', 'dec', 'u', 'g', 'r', 'i', 'z']]

spectra_path = '/fred/oz012/Bruno/data/spectra/boss/loz/*/spec*.fits'
files = glob.glob(spectra_path)
#0: plate; 1: mjd; 2: fiberid
files_bits = [v.split('/')[-1][5:-5].split('-') for v in files]
files_bits = list(map(np.int64,files_bits))

with open('/fred/oz012/Bruno/data/BossGALOZListAllValues.csv', 'w', buffering=1) as f:
    f.write('local_path,redshift,ra,dec,u,g,r,i,z,plate,mjd,fiberid\n')

    for i in range(len(df.axes[0])):
        if i%10000==0: 
            print(i, flush=True)

        df_tmp = df[(df.plate==files_bits[i][0]) & (df.mjd==files_bits[i][1]) 
                    & (df.fiberid==files_bits[i][2])]
        df_tmp = df_tmp.applymap(lambda x: str(x))

        redshift_ = df_tmp['redshift'].values[0]
        ra_ = df_tmp['ra'].values[0]
        dec_ = df_tmp['dec'].values[0]
        u_ = df_tmp['u'].values[0]
        g_ = df_tmp['g'].values[0]
        r_ = df_tmp['r'].values[0]
        i_ = df_tmp['i'].values[0]
        z_ = df_tmp['z'].values[0]
        plate_ = df_tmp['plate'].values[0]
        mjd_ = df_tmp['mjd'].values[0]
        fiberid_ = df_tmp['fiberid'].values[0]
        
        f.write(files[i] + ',' + redshift_ + ',' + ra_ + ',' + dec_ +',' 
                + u_ + ',' + g_ + ',' + r_ + ',' + i_ + ',' + z_ + ',' 
                + plate_ + ',' + mjd_ + ',' + fiberid_ + '\n')
