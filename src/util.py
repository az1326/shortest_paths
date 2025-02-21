# This file contains utility functions for pickling, plotting, and summarizing
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from constants import *

########### PICKLING ###########
# save pickle object
def save(obj, fname):
    Path(fname).parent.mkdir(parents=True, exist_ok=True)
    with open(fname, 'wb') as pickle_file:
        pickle.dump(obj, pickle_file)

# load pickle object
def load(fname):
    with open(fname, 'rb') as pickle_file:
        obj = pickle.load(pickle_file)
        return obj
    
########### PLOTTING ###########
def plot(density, fname=None):
    plt.rcParams['figure.figsize'] = [9.6, 7.2]
    plt.bar(np.arange(0, MAX_DIST), density)
    plt.ylim((0,1))
    plt.xlabel("Distance to Sample", fontsize=40)
    plt.ylabel("Fraction of Nodes", fontsize=40)
    plt.xticks(fontsize=40-10)
    plt.yticks(fontsize=40-10)
    plt.tight_layout()

    if fname is None:
        plt.show()
    else:
        with open(f'{fname}.txt', 'w') as txtfile:
            txtfile.write(f'{density.tolist()}\n')
        plt.savefig(f'{fname}.png')
    plt.clf()

########### TIMING ###########
def summarize_timings(timings, desc):
    print(f'{desc} total time: {np.sum(timings)}')
    print(f'{desc} mean time: {np.mean(timings)}')
    print(f'{desc} time std: {np.std(timings)}')