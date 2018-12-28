import matplotlib.pyplot as plt
import numpy as np
from src.data.spectra_generation import gen_spectrum_2lines as gen
from src.data.data import gen_spectra_to_tfrecord as totf

def plot(x, y, s):
    plt.plot(x, y)
    plt.show()
    #plt.savefig(s)
    plt.close()

totf(200000, 3500, '/fred/oz012/Bruno/data/gen_fixed_ratio_4lines/gen_', 20, norm=True)
"""
classes= [3]
x, y, _ = gen(len(classes), 3500, classes, norm=True)
for i in range(len(y)):
    plot(x[i],y[i], str(i)+'.png')
"""

