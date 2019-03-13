import matplotlib.pyplot as plt
import numpy as np
#from src.data.spectra_generation import gen_spectrum_2lines as gen
from src.data.tfrecords import real_spectra_to_tfrecord
#from src.data.tfrecords import pictures_to_tfrecord
#from src.data.tfrecords import gen_spectra_to_tfrecord

def plot(x, y, s):
    plt.plot(x, y)
    plt.show()
    #plt.savefig(s)
    plt.close()

#pictures_to_tfrecord(filename='/fred/oz012/Bruno/data/celebA/celebA', 
#                     data_folder='/fred/oz012/Bruno/data/celebA/', nshards=100)
real_spectra_to_tfrecord(filename='/fred/oz012/Bruno/data/spectra/legacy/legacy_bit6/spectra_grid', 
                         table_path='/fred/oz012/Bruno/data/Legacy6ListAllValues.csv', 
                         nshards=100, write_fits=True)
#gen_spectra_to_tfrecord(100000, 3500, '/fred/oz012/Bruno/data/spectra/legacy/outliers/spectra',100,norm=True)

"""
classes= [3]
x, y, _ = gen(len(classes), 3500, classes, norm=True)
for i in range(len(y)):
    plot(x[i],y[i], str(i)+'.png')
"""

