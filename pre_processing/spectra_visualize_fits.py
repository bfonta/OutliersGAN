import numpy as np
from astropy.io import fits
from src.utilities import PlotGenSamples
from matplotlib import pyplot as plt
import argparse
from src import argparser

#real data
def convert_to_real_picture(name):
    print("Real:")
    flux_r, lam_r = ([] for _ in range(2))
    with fits.open( name ) as hdu:
        header = hdu[0].header.items

        print(header)
        print("NAXIS1: ", nelems, ' CRVAL1: ', start, ' CDELT1: ', delta)

        flux_ = np.array(hdu[0].data, dtype=np.float)
        #lam_ = np.power( 10, np.linspace(start, stop, nelems) ).astype(np.float32)
        lam_ = np.linspace(start, stop, nelems)
        lam_r.append(lam_)
        flux_r.append(flux_)
    return flux_r, lam_r

#fake data
def convert_to_fake_picture(name):
    print("Fake:")
    flux_f, lam_f = ([] for _ in range(2))
    with fits.open( name ) as hdu:
        header = hdu[0].header.items
        stop = start + delta*(nelems-1)

        print(header)
        print("NAXIS1: ", nelems, ' CRVAL1: ', start, ' CDELT1: ', delta)
        
        flux_ = np.array(hdu[0].data, dtype=np.float)
        #lam_ = np.power( 10, np.linspace(start, stop, nelems) ).astype(np.float32)
        lam_ = np.linspace(start, stop, nelems)
        lam_f.append(lam_)
        flux_f.append(flux_)
    return flux_f, lam_f

def main():
    nrows, ncols = 1, 1
    p = PlotGenSamples(nrows=nrows, ncols=ncols)

    global nelems
    global start
    global delta
    global stop
    nelems = 3500
    start = 3.2552723884583
    delta = 0.00010367979182879
    stop = start + delta*(nelems-1)

    flux_r, lam_r = convert_to_real_picture( FLAGS.real_fits_name )
    flux_f, lam_f = convert_to_fake_picture( FLAGS.fake_fits_name )
    p.plot_spectra(flux_r, lam_r, 'single_real')
    p.plot_spectra(flux_f, lam_f, 'single_fake')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    FLAGS, _ = argparser.add_args(parser)
    main()
