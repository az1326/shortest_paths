# This file contains code implementing the proposed algorithm
import argparse
import time
import numpy as np

from scipy.special import comb
from tqdm import tqdm

from util import *
from constants import *


########### NON SBM GRAPHS ###########
# single node degree distribution for binomial graphs
def binom_degree_dist(n):
    degs_single = np.arange(BINOM_LO, BINOM_HI, dtype=np.float64)
    ps = np.zeros_like(degs_single, dtype=np.float64)

    for k in range(BINOM_LO, BINOM_HI):
        i = k - BINOM_LO # get index

        # N-1 choose k ways, times probability for each way
        ps[i] = comb(n-1, k, exact=True) * \
                (BINOM_P_DICT[n] ** k) * \
                ((1 - BINOM_P_DICT[n]) ** (n - 1 - k))
        
    # normalize single node degree distribution
    ps = ps / np.sum(ps)

    return ps, degs_single

# single node degree distribution for power law graphs
def power_degree_dist(gamma, lo, hi):
    degs_single = np.arange(lo, hi, dtype=np.float64)
    ps = (degs_single ** gamma).astype(np.float64)

    # normalize single node degree distribution
    ps = ps / np.sum(ps)

    return ps, degs_single

# sample supernode and reached node degree distributions
def non_sbm_dists(graph_type, n, sample_size, sample_type):
    # get degree ranges and single node distribution
    if graph_type == 'binomial':
        ps, degs_single = binom_degree_dist(n)
        degs_sample = np.arange(BINOM_LO*sample_size, (BINOM_HI-1)*sample_size+1)
    elif graph_type == 'power_a':
        ps, degs_single = power_degree_dist(POWER_A_GAMMA, POWER_A_LO, POWER_A_HI)
        degs_sample = np.arange(POWER_A_LO*sample_size, (POWER_A_HI-1)*sample_size+1)
    else:
        assert(graph_type == 'power_b')
        ps, degs_single = power_degree_dist(POWER_B_GAMMA, POWER_B_LO, POWER_B_HI)
        degs_sample = np.arange(POWER_B_LO*sample_size, (POWER_B_HI-1)*sample_size+1)

    # calculate degree distribution given existence of edge
    c = np.sum(degs_single * ps)
    kps = (degs_single * ps) / c
    
    # calculate degree distribution for sample via polypow
    if sample_type == 'random':
        ps_sample = np.polynomial.polynomial.polypow(ps, sample_size)
    elif sample_type == 'snowball':
        ps_sample = np.polynomial.polynomial.polypow(kps, sample_size)
        degs_sample = degs_sample - 2 * sample_size

    keep = np.where(ps_sample > 1e-4)[0]
    ps_sample = ps_sample[keep]
    degs_sample = degs_sample[keep]

    # return everything
    return ps_sample, degs_sample, kps, degs_single

# m_{N,l} for 1 <= l < MAX_DIST
def non_sbm_m(graph_type, n, dists):
    # unpack dists
    ps_sample, degs_sample, kps, degs_single = dists

    # initialize m
    m = np.zeros(MAX_DIST, dtype=np.float64)

    for l in tqdm(range(MAX_DIST), position=1, leave=False,
                  desc='Distance'):
        if l == 0:
            # empty placeholder for ease-of-indexing
            continue
        elif l == 1:
            # get m_{N,1} from base case
            init = ((1 - 1/(n-1)) ** degs_sample).astype(np.float64)
            m[l] = np.sum(ps_sample * init)
        else:
            # initialize \tilde{m}_{N-l+1,1}
            init_tilde = ((1 - 1/(n-l)) ** (degs_single-1)).astype(np.float64)
            m_tilde = np.sum(kps * init_tilde)

            # recurse to get \tilde{m}_{N-1, l-1}
            for _ in range(n-l+1, n-1): # l-2 steps
                m_tilde = np.sum(kps * (m_tilde ** (degs_single - 1)))

            # get m_{N, l}
            m[l] = np.sum(ps_sample * (m_tilde ** degs_sample))

    return m


########### SBM GRAPHS ###########
def sbm_degree_dist(n):
    # within block degree, same pattern as binomial
    degs_within = np.arange(SBM_WITHIN_LO, SBM_WITHIN_HI)
    ps_within = np.zeros_like(degs_within, dtype=np.float64)
    for k in range(SBM_WITHIN_LO, SBM_WITHIN_HI):
        i = k - SBM_WITHIN_LO
        ps_within[i] = comb(SBM_BLOCK_SIZE-1, k, exact=True) * \
                       (SBM_P_WITHIN ** k) * \
                       ((1 - SBM_P_WITHIN) ** (SBM_BLOCK_SIZE - 1 - k))
    ps_within = ps_within / np.sum(ps_within)
        
    # across block degree, same pattern as binomial
    degs_across = np.arange(SBM_ACROSS_LO, SBM_ACROSS_HI)
    ps_across = np.zeros_like(degs_across, dtype=np.float64)
    for k in range(SBM_ACROSS_LO, SBM_ACROSS_HI):
        i = k-SBM_ACROSS_LO
        ps_across[i] = comb(SBM_N_DICT[n] - SBM_BLOCK_SIZE, k, exact=True) * \
                        (SBM_P_ACROSS ** k) * \
                        ((1 - SBM_P_ACROSS) **
                         (SBM_N_DICT[n] - SBM_BLOCK_SIZE - k))
    ps_across = ps_across / np.sum(ps_across)

    return (ps_within, degs_within, ps_across, degs_across)


def sbm_dists(n, sample_size, sample_type):
    # get degree ranges and single node distribution
    ps_within, degs_within, ps_across, degs_across = sbm_degree_dist(n)

    # calculate degree distribution given existence of edge
    c_within = np.sum(degs_within * ps_within)
    kps_within = (degs_within * ps_within) / c_within
    c_across = np.sum(degs_across * ps_across)
    kps_across = (degs_across * ps_across) / c_across


    # calculate degree distribution for sample via polypow
    if sample_type == 'random':
        # for random sampling, each node is the same and the RV representing
        # the number of within edges and across edges contributed is the same

        # sample's within degree distribution
        ps_within_sample = \
            np.polynomial.polynomial.polypow(ps_within, sample_size)

        # sample's across degree distribution
        ps_across_sample = \
            np.polynomial.polynomial.polypow(ps_across, sample_size)
        
        degs_within_sample = np.arange(SBM_WITHIN_LO * sample_size,
                                       (SBM_WITHIN_HI-1) * sample_size + 1)
        degs_across_sample = np.arange(SBM_ACROSS_LO * sample_size,
                                        (SBM_ACROSS_HI-1) * sample_size + 1)
    else:
        assert(sample_type == 'snowball')
        # for snowball we need to consider the edges that were taken to form
        # the sample, and subtract those from the final accordingly

        # get prob of taking a within edge given last edge was within/across
        # sum over all degrees by law of total probability
        p_w_given_w = 0
        p_w_given_a = 0
        for w in range(SBM_WITHIN_LO, SBM_WITHIN_HI):
            for a in range(SBM_ACROSS_LO, SBM_ACROSS_HI):
                p_w_given_w += kps_within[w-SBM_WITHIN_LO] * \
                            ps_across[a-SBM_ACROSS_LO] * \
                            ((w - 1) / (w + a - 1))
                p_w_given_a += ps_within[w-SBM_WITHIN_LO] * \
                            kps_across[a-SBM_ACROSS_LO] * \
                            (w / (w + a - 1))
        # derive probability of a snowball edge being via an within edge
        p_from_within = p_w_given_a / (1 + p_w_given_a - p_w_given_w)

        # obtain within and across distribution of average node
        ps_within_padded = np.zeros(ps_within.shape[0] + 2)
        ps_within_padded[2:] = ps_within

        ps_across_padded = np.zeros(ps_across.shape[0] + 2)
        ps_across_padded[2:] = ps_across

        kps_within_padded = np.zeros(kps_within.shape[0] + 2)
        kps_within_padded[:-2] = kps_within

        kps_across_padded = np.zeros(kps_across.shape[0] + 2)
        kps_across_padded[:-2] = kps_across

        ps_within_single = p_from_within * kps_within_padded + \
                           (1-p_from_within) * ps_within_padded
        ps_across_single = p_from_within * ps_across_padded + \
                            (1-p_from_within) * kps_across_padded

        ps_within_sample = \
                np.polynomial.polynomial.polypow(ps_within_single, sample_size)
        ps_across_sample = \
                np.polynomial.polynomial.polypow(ps_across_single, sample_size)
        
        degs_within_sample = np.arange((SBM_WITHIN_LO-2) * sample_size,
                                       (SBM_WITHIN_HI-1) * sample_size + 1)
        degs_across_sample = np.arange((SBM_ACROSS_LO-2) * sample_size,
                                        (SBM_ACROSS_HI-1) * sample_size + 1)

        # keep only parts with non-negative degree
        ps_within_sample = ps_within_sample[max(0, -degs_within_sample[0]):]
        degs_within_sample = degs_within_sample[max(0, -degs_within_sample[0]):]

        ps_across_sample = ps_across_sample[max(0, -degs_across_sample[0]):]
        degs_across_sample = degs_across_sample[max(0, -degs_across_sample[0]):]
        
    # cutoff low probabilities
    within_keep = np.where(ps_within_sample > 1e-4)[0]
    ps_within_sample = ps_within_sample[within_keep]
    degs_within_sample = degs_within_sample[within_keep]

    across_keep = np.where(ps_across_sample > 1e-4)[0]
    ps_across_sample = ps_across_sample[across_keep]
    degs_across_sample = degs_across_sample[across_keep]

    # return everything
    return (ps_within_sample, degs_within_sample,
            ps_across_sample, degs_across_sample,
            ps_within, ps_across,
            kps_within, kps_across)


def sbm_m(n, dists):    
    # sample dists
    ps_within_sample, degs_within_sample, \
    ps_across_sample, degs_across_sample, \
    ps_within, ps_across, \
    kps_within, kps_across \
        = dists
    
    if kps_within[0] == 0:
        kps_within = kps_within[1:]
        kps_within_lo = SBM_WITHIN_LO+1
    else:
        kps_within_lo = SBM_WITHIN_LO
    
    if kps_across[0] == 0:
        kps_across = kps_across[1:]
        kps_across_lo = SBM_ACROSS_LO+1
    else:
        kps_across_lo = SBM_ACROSS_LO
    
    # initialize m
    m = np.zeros(MAX_DIST, dtype=np.float64)

    for l in tqdm(range(MAX_DIST), position=1, leave=False,
                  desc='Distance'):
        if l == 0:
            # empty placeholder for ease-of-indexing
            continue
        elif l == 1:
            # get m_{N,1} from base case
            # k_i (within degree) component
            ps_within_vec = ps_within_sample * \
                            ((1 - 1/(SBM_N_DICT[n]-l)) ** degs_within_sample)
            ps_within_vec = ps_within_vec.reshape((-1, 1))

            # k_o (across degree) component
            ps_across_vec = ps_across_sample * \
                             ((1 - 1/(SBM_N_DICT[n]-l)) ** degs_across_sample)
            ps_across_vec = ps_across_vec.reshape((1, -1))

            # broadcast product gets all combos of k_i and k_o
            # for small samples and reasonable degree distributions,
            # no combo will be near N
            m[l] = np.sum(ps_within_vec * ps_across_vec)
        else:
            # get m_{N,l}
            # initialize \tilde{m}^i_{N-l+1,1} and \tilde{m}^o_{N-l+1,1}
            m_tilde_within = 0
            m_tilde_across = 0

            # sum over all values for within and across degree.
            ps_within_vec = ps_within * \
                            ((1 - 1/(SBM_N_DICT[n]-l)) **
                             np.arange(SBM_WITHIN_LO, SBM_WITHIN_HI))
            ps_within_vec = ps_within_vec.reshape((-1, 1))

            kps_within_vec = kps_within * \
                            ((1 - 1/(SBM_N_DICT[n]-l)) **
                             np.arange(kps_within_lo-1, SBM_WITHIN_HI-1))
            kps_within_vec = kps_within_vec.reshape((-1, 1))

            ps_across_vec = ps_across * \
                            ((1 - 1/(SBM_N_DICT[n]-l)) **
                             np.arange(SBM_ACROSS_LO, SBM_ACROSS_HI))
            ps_across_vec = ps_across_vec.reshape((1, -1))

            kps_across_vec = kps_across * \
                            ((1 - 1/(SBM_N_DICT[n]-l)) **
                             np.arange(kps_across_lo-1, SBM_ACROSS_HI-1))
            kps_across_vec = kps_across_vec.reshape((1, -1))

            m_tilde_within = np.sum(kps_within_vec * ps_across_vec)
            m_tilde_across = np.sum(ps_within_vec * kps_across_vec)

            # recurse to get \tilde{m}^i_{N-l,l-1} and \tilde{m}^o_{N-l,l-1}
            for _ in range(SBM_N_DICT[n]-l+1, SBM_N_DICT[n]-1): # l-2 steps
                # sum over all values for within and across degree.
                ps_within_vec = ps_within * \
                                (m_tilde_within **
                                np.arange(SBM_WITHIN_LO, SBM_WITHIN_HI))
                ps_within_vec = ps_within_vec.reshape((-1, 1))

                kps_within_vec = kps_within * \
                                (m_tilde_within **
                                np.arange(kps_within_lo-1, SBM_WITHIN_HI-1))
                kps_within_vec = kps_within_vec.reshape((-1, 1))

                ps_across_vec = ps_across * \
                                (m_tilde_across **
                                np.arange(SBM_ACROSS_LO, SBM_ACROSS_HI))
                ps_across_vec = ps_across_vec.reshape((1, -1))

                kps_across_vec = kps_across * \
                                (m_tilde_across **
                                np.arange(kps_across_lo-1, SBM_ACROSS_HI-1))
                kps_across_vec = kps_across_vec.reshape((1, -1))

                m_tilde_within = np.sum(kps_within_vec * ps_across_vec)
                m_tilde_across = np.sum(ps_within_vec * kps_across_vec)
            
            # get m_{N, l}
            # k_i (within degree) component
            ps_within_vec = ps_within_sample * \
                                (m_tilde_within ** degs_within_sample)
            ps_within_vec = ps_within_vec.reshape((-1, 1))

            # k_o (across degree) component
            ps_across_vec = ps_across_sample * \
                                (m_tilde_across ** degs_across_sample)
            ps_across_vec = ps_across_vec.reshape((1, -1))

            # broadcast product gets all combos of k_i and k_o.
            # for small samples and reasonable degree distributions,
            # no combo will be near N
            m[l] = np.sum(ps_within_vec * ps_across_vec)
    
    return m


########### ALL GRAPHS ###########
# gets pdf from m
def get_pdf(graph_type, n, sample_size, m):
    # get sample proportion
    if (graph_type == 'binomial' or
        graph_type == 'power_a' or graph_type == 'power_b'):
        sample_proportion = sample_size / n
    else:
        assert(graph_type == 'sbm')
        sample_proportion = sample_size / SBM_N_DICT[n]
    
    # calculate cdf
    cdf = np.zeros(MAX_DIST + 1, dtype=np.float64)
    cdf[0] = 1
    cdf[1] = 1 - sample_proportion
    cdf[2:MAX_DIST + 1] = m[1:MAX_DIST]
    cdf = np.cumprod(cdf)

    # return pdf
    return cdf[:MAX_DIST] - cdf[1:MAX_DIST + 1]


########### MAIN FUNCTION ###########
def estimate(args):
    timings = []
    for _ in tqdm(range(args.repeat), position=0, desc='repeats'):
        start_time = time.perf_counter()
        if args.graph_type == 'sbm':
            # get degree distributions
            dists = sbm_dists(args.graph_size, args.sample_size,
                              args.sample_type)
            # calculate m
            m = sbm_m(args.graph_size, dists)
        else:
            # get degree distributions
            dists = non_sbm_dists(args.graph_type, args.graph_size,
                                  args.sample_size,args.sample_type)
            # calculate m
            m = non_sbm_m(args.graph_type, args.graph_size, dists)

        # generate pdf
        pdf = get_pdf(args.graph_type, args.graph_size, args.sample_size, m)

        end_time = time.perf_counter()
        timings.append(end_time - start_time)

    # note time
    summarize_timings(timings, 'density est')

    # save pdf
    if args.save_dist:
        save(pdf, f'{args.save_dist}.pkl')
        save(timings, f'{args.save_dist}_timings.pkl')

    # plot pdf
    plot(pdf, args.save_plot)


########### ARGUMENT PARSING ###########
def print_args(args):
    print('=' * 80)
    for k, v in args.__dict__.items():
        print(f'    - {k} : {v}')
    print('=' * 80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='empirical_distribution.py')
    # graph settings
    parser.add_argument('--graph_type',
                        choices=['binomial', 'power_a', 'power_b', 'sbm'])
    parser.add_argument('--graph_size', type=int, default=20000)
    
    # sample settings
    parser.add_argument('--sample_size', type=int)
    parser.add_argument('--sample_type', choices=['random', 'snowball'])
    
    # save settings
    parser.add_argument('--save_dist', default=None)
    parser.add_argument('--save_plot', default=None)

    # timing settings
    parser.add_argument('--repeat', default=1, type=int)

    args = parser.parse_args()
    print_args(args)

    estimate(args)
