import numpy as np

#from src.data.spectra_generation import gen_spectrum_2lines as gen
from src.data.tfrecords import real_spectra_to_tfrecord
#from src.data.tfrecords import pictures_to_tfrecord
#from src.data.tfrecords import gen_spectra_to_tfrecord
#from src.data import fits as ff

"""
ff.to_fits(np.arange(3500),np.ones((3500)),"x", "y", "test.fits")
pictures_to_tfrecord(filename='/fred/oz012/Bruno/data/celebA/celebA', 
                     data_folder='/fred/oz012/Bruno/data/celebA/', nshards=100)
"""
real_spectra_to_tfrecord(filename='/fred/oz012/Bruno/data/spectra/qso_zWarning/spectra2_linearlysampled_', 
                         table_path='/fred/oz012/Bruno/data/qso_zWarning_ListAllValues.csv', 
                         nshards=100, write_fits=False)
"""
gen_spectra_to_tfrecord(100000, 3500, '/fred/oz012/Bruno/data/spectra/legacy/outliers/spectra',100,norm=True)
"""



