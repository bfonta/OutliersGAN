import numpy as np
import random
import glob

path = '/fred/oz012/Bruno/data/spectra/gal_form_burst/'
g = glob.glob(path + '/*/*.fits')

random.seed()
r = [np.random.randint(0, len(g)-1) for _ in range(100)]

with open('random_spectra_gal.txt', 'w') as f:
    for i in r:
        f.write(g[i]+'\n')
