import numpy as np
import random
import glob
import argparse
import pandas as pd
from astropy.io import fits
from src import argparser
from src.utilities import resampling_1d, is_invalid                                       
from src.data.fits import to_fits
 
def main():
    if FLAGS.data_type == 'gal':
        folder_path = '/fred/oz012/Bruno/data/spectra/gal_starforming_starburst_zWarning'
        table_path = '/fred/oz012/Bruno/data/gal_starforming_starburst_zWarning_ListAllValues.csv'
        bounds = (3750, 7000)
    elif FLAGS.data_type == 'qso':
        folder_path = '/fred/oz012/Bruno/data/spectra/qso_zWarning'
        table_path = '/fred/oz012/Bruno/data/qso_zWarning_ListAllValues.csv'
        bounds = (1800, 4150)
    else:
        raise ValueError('Please specify a valid data_type.')
    tot_length = 3500

    df = pd.read_csv(table_path)
    g = glob.glob(folder_path + '/*/*.fits')
    random.seed()
    N = 30
    r = [np.random.randint(0, len(g)-1) for _ in range(N)]

    yescount, nocount, i = 0, 0, 0
    while yescount < N:
        print(yescount, nocount, i)
        ri = r[i]
        i += 1
        rshift_ = float(df.loc[df['local_path'] == g[ri]]['redshift'].astype(np.float32))
        with fits.open(g[ri]) as hdul:
            flux_ = hdul[1].data['flux'].astype(np.float32)
            lam_ = np.power(10, hdul[1].data['loglam'])
            lam_ /= (1 + rshift_)

            if is_invalid(flux_) or lam_[0] >= bounds[0] or lam_[-1] <= bounds[1]:
                r.append(np.random.randint(0,len(g)-1))
                nocount += 1
                continue

            lam_, flux_ = resampling_1d(x=lam_, y=flux_, bounds=bounds, size=tot_length)
            lam_ = np.log10(lam_)
            init_l = lam_[0] #initial wavelength          
            delta_l = lam_[1]-lam_[0]
            assert np.isclose(delta_l, lam_[1000]-lam_[999], atol=1e-6) #check bin uniformity              
            print("ToFITS", ri, len(r))
            to_fits(y=[flux_], name='data_'+str(yescount), params=(1., delta_l, init_l))
            yescount += 1
            """
            from src.utilities import PlotGenSamples
            p = PlotGenSamples(nrows=1, ncols=1)
            p.plot_spectra(samples=[flux_], lambdas=[lam_], fix_size=True, name='2')
            """

if __name__ == "__main__":                  
    parser = argparse.ArgumentParser()
    FLAGS, _ = argparser.add_args(parser)
    main()
