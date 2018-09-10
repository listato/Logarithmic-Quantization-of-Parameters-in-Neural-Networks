from __future__ import division
from matplotlib import pyplot as plt
import numpy as np
import cupy as cp

def dens_diag(data, intervalnum=100, save=None, dpi=1200):
    """"
    placeholder
    """
    if type(data) is np.ndarray:
        data = data
    else:
        if type(data) is cp.ndarray:
            data = cp.asnumpy(data)
        else:
            raise TypeError("type error")

    figure = plt.figure()

    plt.subplot(1,1,1)
    minima = np.min(data)
    maxima = np.max(data)


    #compoute the desity of the weights
    density = np.linspace(round(minima,2),round(maxima,2),intervalnum)
    y, x = np.histogram(data, bins=density)
    x = x + (maxima-minima)/(intervalnum*2)
    plt.plot(x[:-1], y,'k', linewidth=1)
    plt.scatter(x[:-1], y, s=5, c='k', label='Weights count within intervals')

    #compute each interval size
    interval = round((maxima-minima)/intervalnum, 3)
    plt.ylabel('density: count within {} interval'.format(interval))
    plt.xlabel('value of weights')
    ticks_to_be_showed = []

    for _ in range(intervalnum):
        ticks_to_be_showed.append('')

    for _ in range(0, intervalnum, int(intervalnum/5)):
        ticks_to_be_showed[_] = round(density[_],2)
    ticks_to_be_showed[intervalnum - 1] = round(density[intervalnum - 1],2)

    plt.xticks(density, ticks_to_be_showed)
    plt.legend(loc='best')

    plt.tight_layout()
    plt.show()
    if not save==None:
        figure.savefig(save , dpi)
