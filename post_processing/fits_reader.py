from astropy.io import fits

with fits.open('generate0_0.fits') as hdu:
    for i in range(len(hdu)):
        print(hdu[i])
        for k,v in hdu[i].header.items():
            print(k, v)
        print(hdu[i].data)
