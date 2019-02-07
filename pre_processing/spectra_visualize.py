import numpy as np
import pandas as pd
from astropy.io import fits
from matplotlib import pyplot as plt

table_path = '/fred/oz012/Bruno/data/BossGALOZListAllValues.csv'
df = pd.read_csv(table_path)

nrows = 6
ncols = 1
arr = np.random.randint(0,len(df.axes[0]), size=(nrows))

fig = plt.figure(figsize=(25,20))
for j,i in enumerate(arr):
    with fits.open(df['local_path'].iloc[i]) as hdu:
        flux_ = hdu[1].data['flux'].astype(np.float32)
        lam_ = np.power( 10, hdu[1].data['loglam'] ).astype(np.float32)
        rshift_ = df['redshift'].iloc[i].astype(np.float32)
        lam_ = lam_ / (1 + rshift_)
        
        selection = (lam_>3400) & (lam_<6900)
        lam_= lam_[selection]
        flux_= flux_[selection]

        fig.add_subplot(nrows, ncols, j+1) 
        plt.plot(lam_, flux_)
plt.savefig('1.png')
