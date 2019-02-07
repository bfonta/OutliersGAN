import h5py
import numpy as np
from matplotlib import pyplot as plt

with h5py.File('/fred/oz012/Bruno/data/sdss_spectra.hdf5', 'r') as f:
    dset = f['spectra']
    arr_min = np.empty([0])
    arr_max = np.empty([0])
    for i in range(len(dset)):
        lam = dset[i,:,0]
        lam = lam[np.nonzero(lam)]
        arr_min = np.append(arr_min, lam[0])
        arr_max = np.append(arr_max, lam[-1])

mean_min = np.round(np.mean(arr_min),2)
mean_max = np.round(np.mean(arr_max),2)

line_x = 6000
line_y = 800
plt.plot(np.arange(line_x, line_x+3500), np.full((3500), line_y), 
         linewidth=3., label='Training spectra length', color='blue')
plt.hist(arr_min, 5000, label='Wavelength minima', color='red')
plt.hist(arr_max, 5000, label='Wavelength maxima', color='green')
plt.text(3250, 1100, r'$\mu=%s$ A'%mean_min, color='red')
plt.text(8500, 400, r'$\mu=%s$ A'%mean_max, color='green')
plt.text(line_x-200, line_y+50, r'Mean difference: %s A'%(mean_max-mean_min), color='blue')
plt.xlabel('Wavelength [A]')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('plot_min_max.png')
