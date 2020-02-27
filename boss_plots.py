import glob
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
import time

save_path = '/fred/oz012/Bruno/figs/'
table_path = '/fred/oz012/Bruno/data/BOSS_GALLOZ.csv'
#spectra_path = '/fred/oz012/Bruno/data/spectra/boss/loz/*/spec*.fits'
#files = glob.glob(spectra_path)
df = pd.read_csv('/fred/oz012/Bruno/data/BossGALOZListAllValues.csv')

def get_spectra():
    r = np.random.randint(low=0, high=len(df.axes[0]))
    with fits.open(df['local_path'].iloc[r]) as hdu:
        f = hdu[1].data
        return np.array([list(np.power(10,f['loglam'])), f['flux']]), df['redshift'].iloc[r]

nrows, ncols = 3, 1
for i in range(5):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(25,15))
    spec, rshift = get_spectra()
    spec_dershift = spec[0]/(1+rshift)
    ax[0].plot(spec_dershift,spec[1])
    ax[0].set_title('Before selection')
    ax[0].set_xlabel(r'$\lambda$')
    ax[0].set_ylabel('Flux')

    spec = spec[:,250:3750]
    spec.astype(np.float32)
    spec_dershift = spec[0]/(1+rshift)
    ax[1].plot(spec_dershift,spec[1])
    ax[1].set_title('After selection')
    ax[1].set_xlabel(r'$\lambda$')
    ax[1].set_ylabel('Flux')
    
    mean = np.mean(spec_dershift, axis=0)
    diff = spec[1] - mean
    std = np.sqrt(np.mean(diff**2, axis=0))
    spec[1] = diff / std / 4

    ax[2].plot(spec_dershift,spec[1])
    ax[2].set_title('After normalization')
    ax[2].set_xlabel(r'$\lambda$')
    ax[2].set_ylabel('Flux')

    plt.savefig(save_path + 'boss_spectra' + str(i) + '.png')
    plt.close()

###############
df = pd.read_csv(table_path, skiprows=0)

left, right = 0., .8
diff = right-left
bins = np.linspace(left,right,100)

#s = pd.Series(df.redshift)
#counts, bins = np.histogram(s, bins=20)

df.redshift.plot.hist(bins=list(bins))
plt.xlim(left,right)
plt.xlabel('redshift')
plt.savefig(save_path + 'boss_redshift.png')
plt.close()

###############
magnitudes = ['u', 'g', 'r', 'i', 'z']
limits = ((17,28,1), 
          (16.5,22.5,1),
          (15,20.,1),
          (13.5,20,1),
          (13.5,19.5,1))
fig = plt.figure(figsize=(15,9))
nrows, ncols = 2, 3
for irow in range(nrows):
    for icol in range(ncols):
        i = icol + irow*ncols
        if i>=5: break
        ax = fig.add_subplot(nrows, ncols, i+1)
        df_tmp = df[(df[magnitudes[i]]>limits[i][0]) & (df[magnitudes[i]]<limits[i][1])]
        df_tmp = df_tmp[(df_tmp.redshift<limits[i][2])]
        df_tmp.plot.hexbin(x=magnitudes[i], y='redshift', gridsize=70, ax=ax)
        plt.xlabel(magnitudes[i] + ' [mag]')
        plt.ylabel('redshift')
plt.savefig(save_path + 'boss_mag_vs_z.png')
plt.close()
