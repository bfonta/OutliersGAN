from astropy.io import fits
import argparse
from src import argparser

def main():
    with fits.open(FLAGS.fname) as hdul:
        print()
        print("###### HDUList Information ######")
        print(hdul.info())
        print()
        print("###### HDUList Items ######")
        for i in range(len(hdul)):
            print("Item ", i+1)
            print(hdul[i])
            print(hdul[i].data)      
            for k,v in hdul[i].header.items():
                print(k, v)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    FLAGS, _ = argparser.add_args(parser)
    main()
