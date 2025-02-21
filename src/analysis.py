# This file contains code generating the graphs and tables in the paper
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import wasserstein_distance

from util import *
from constants import *

# calculates Wasserstein-1 distance between two distribution
def calc_distance(dist1, dist2):
    u_values = np.arange(len(dist1))
    v_values = np.arange(len(dist2))
    return wasserstein_distance(u_values, v_values, dist1, dist2)

# plots empirical vs estimated distribution
def plot_distribution(emp_all, est, fname=None):
    density = np.mean(emp_all, axis=0)
    errors = np.std(emp_all, axis=0)

    plt.rcParams['figure.figsize'] = [9.6, 7.2]
    plt.scatter(np.arange(0, MAX_DIST), est, s=200, facecolors='none', edgecolors='b', linewidths=10)
    plt.scatter(np.arange(0, MAX_DIST), density, s=200, c='r', marker='_', linewidths=10)
    plt.errorbar(np.arange(0, MAX_DIST), density, yerr=2*errors, fmt='none', color='r', elinewidth=10)

    plt.ylim((0,1))
    plt.xlabel("Distance to Sample", fontsize=40)
    plt.ylabel("Fraction of Nodes", fontsize=40)
    plt.xticks(fontsize=40-10)
    plt.yticks(fontsize=40-10)
    plt.tight_layout()

    if fname is None:
        plt.show()
    else:
        plt.savefig(f'{fname}.png')
    plt.clf()

# calculates mean and standard deviation of wasserstein distance
def distance_stats(emp_all, est):
    distances = []
    for d in emp_all:
        distances.append(calc_distance(d, est))
    return np.mean(distances), np.std(distances)

# utility function for writing latex for distance table
def build_distance_table(emp_all, est, desc, outfile):
    mean, std = distance_stats(emp_all, est)
    outfile.write(f'\\texttt{{{desc}}} & ${mean:.4f} \pm {std:.4f}$ \\\\\n')

# utility function for writing latex for timing table
def build_timing_table(emp_t, est_t, desc, outfile):
    emp_mean = emp_t.mean()
    emp_std = emp_t.std()
    est_mean = est_t.mean()
    est_std = est_t.std()

    if emp_mean < est_mean:
        outfile.write(
            f'\\texttt{{{desc}}} & ' + \
            f'$\mathbf{{{emp_mean:.4f} \pm {emp_std:.4f}}}$ & ' + \
            f'${est_mean:.4f} \pm {est_std:.4f}$ \\\\\n'
        )
    else:
        outfile.write(
            f'\\texttt{{{desc}}} & ' + \
            f'${emp_mean:.4f} \pm {emp_std:.4f}$ & ' + \
            f'$\mathbf{{{est_mean:.4f} \pm {est_std:.4f}}}$ \\\\\n'
        )
     


if __name__ == '__main__':
    graph_types = ['binomial', 'power_a', 'power_b', 'sbm']
    graph_sizes = [20000, 40000, 100000]
    sample_types = ['random', 'snowball']
    sample_sizes = [200, 400, 1000]

    # plots and tables for every experiment
    timings_file = open('output/timing/timings_table_pt.txt', 'w')
    distances_file = open('output/timing/distances_table_pt.txt', 'w')

    for g_t in graph_types:
        for g_s in graph_sizes:
            for s_t in sample_types:
                for s_s in sample_sizes:
                    print(f'{g_t} {g_s} {s_t} {s_s}')

                    emps = \
                        load(f'pickle/timing/{g_t}/' +
                             f'size_{g_s}_{s_t}_{s_s}_dists.pkl')
                    est = \
                        load(f'pickle/timing/{g_t}/' + 
                             f'size_{g_s}_{s_t}_{s_s}_dists_est.pkl')
                    emp_t = \
                        load(f'pickle/timing/{g_t}/' +
                             f'size_{g_s}_{s_t}_{s_s}_dists_timings.pkl')
                    est_t = \
                        load(f'pickle/timing/{g_t}/' + 
                             f'size_{g_s}_{s_t}_{s_s}_dists_est_timings.pkl')

                    emp_all = np.array(emps[1]).reshape(-1, MAX_DIST)
                    est = np.array(est)
                    emp_t = np.array(emp_t)
                    est_t = np.array(est_t)

                    build_distance_table(emp_all, est,
                                         f'{g_t} {g_s} {s_t} {s_s}',
                                         distances_file)
                    build_timing_table(emp_t, est_t,
                                       f'{g_t} {g_s} {s_t} {s_s}',
                                       timings_file)
                    
                    plot_distribution(emp_all, est,
                                      f'output/timing/{g_t}/' +
                                      f'size_{g_s}_{s_t}_{s_s}_both')
    
    timings_file.close()
    distances_file.close()

    # compare empirical and estimated distributions
    diffs = []
    for g_t in graph_types:
        for g_s in graph_sizes:
            for s_s in sample_sizes:
                emp_rand = load(f'pickle/timing/{g_t}/' +
                                f'size_{g_s}_random_{s_s}_dists.pkl')[0]
                emp_snow = load(f'pickle/timing/{g_t}/' +
                                f'size_{g_s}_snowball_{s_s}_dists.pkl')[0]
                est_rand = load(f'pickle/timing/{g_t}/' + 
                                f'size_{g_s}_random_{s_s}_dists_est.pkl')
                est_snow = load(f'pickle/timing/{g_t}/' + 
                                f'size_{g_s}_snowball_{s_s}_dists_est.pkl')
                weights = np.arange(0, MAX_DIST)

                emp_rand_mean = np.sum(weights * np.array(emp_rand))
                emp_snow_mean = np.sum(weights * np.array(emp_snow))
                est_rand_mean = np.sum(weights * np.array(est_rand))
                est_snow_mean = np.sum(weights * np.array(est_snow))

                print(f'{g_t} {g_s} {s_s}')
                diffs.append(np.abs(emp_rand_mean - emp_snow_mean))
                if emp_rand_mean > emp_snow_mean:
                    if est_rand_mean > est_snow_mean:
                        print('correct rand > snow')
                    else:
                        print('WRONG rand > snow')
                else:
                    if est_rand_mean > est_snow_mean:
                        print('WRONT snow > emp')
                    else:
                        print('correct snow > rand')

    # scatterplot difference in mean distances
    diffs = np.array(diffs)
    print(np.mean(diffs), np.std(diffs), np.min(diffs))
    fig = plt.gcf()
    fig.set_size_inches(4.8, 1.2)
    plt.scatter(diffs, np.zeros_like(diffs), marker='o', facecolors='none', edgecolors='b')
    plt.xlabel('abs(difference in mean distance)')
    frame1 = plt.gca()
    frame1.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.show()


