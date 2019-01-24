import glob
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

def get_spectra(filename):
    spectra = glob.glob(filename+'*.fits')    
    spectra_number = len(spectra)

    with fits.open(spectra[0]) as hdu:
        f = hdu[1].data['flux']
        f = f[0:3750]
        f = f[250:]
        f = f.astype(np.float32)

    plt.plot(np.arange(0,3500), f)
    plt.savefig('test.png')
    plt.close()
    return f

spec = get_spectra('/fred/oz012/Bruno/data/spectra/spec-3699-55517-0981')
print("hhh")

def outlier_id_roll(arr, cut=2.5):
    l = len(arr)
    width = int(l / 10)
    id_flags = np.zeros(l)
    for m1 in range(2*width,l-3*width):
        m2 = m1 + width
        """
        dist = np.abs(arr[m1:m2] - np.median(arr[m1:m2]))
        medmed = np.median(dist)
        s = dist / (medmed if medmed else 1.)
        """
        dist = np.abs(arr[m1:m2] - np.mean(arr[m1:m2]))
        s = dist / np.std(arr[m1:m2])
        id_flags[m1:m2] = s > cut
    return id_flags

roll_ids = outlier_id_roll(spec)
spec[roll_ids == 1] = 0.

plt.plot(np.arange(0,3500), spec)
plt.ylim([-16,16])
plt.savefig('test2.png')
plt.close()
