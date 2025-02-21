# This file contains code for obtaining empirical distributions, including:
#   - Generating graphs
#   - Sampling
#   - BFS to calculate DSPDs
import argparse
import time
import random
import networkx as nx
import numpy as np

from collections import deque
from tqdm import tqdm

from constants import *
from util import *


########### BINOMIAL GRAPHS ###########
# Generates a list of binomial graphs
def generate_binomial_graphs(num_graphs, n):
    graphs = []
    timings = []
    for _ in tqdm(range(num_graphs), desc='Generating graphs'):
        start_time = time.perf_counter()

        graph = nx.binomial_graph(n, BINOM_P_DICT[n])

        end_time = time.perf_counter()
        graphs.append(graph)
        timings.append(end_time - start_time)
    
    return graphs, timings

########### POWER LAW GRAPHS ###########
# obtains sampled degree list of power law graph
def get_degree_list(gamma, lo, hi, n):
    degs = np.arange(lo, hi, dtype=np.float64)
    ps = (degs ** gamma).astype(np.float64)
    ps = ps / np.sum(ps)

    def get_degree():
        return np.random.choice(degs, p=ps).astype(int)
    
    deg_list = [get_degree() for _ in range(n)]

    if np.sum(deg_list) % 2 == 1:
        deg_list[np.random.randint(0, n)] += np.random.choice([-1, 1])
    return deg_list

# Generates a list of power law graphs
def generate_power_law_graphs(num_graphs, gamma, lo, hi, n):
    graphs = []
    timings = []
    for _ in tqdm(range(num_graphs), desc='Generating graphs'):
        start_time = time.perf_counter()

        deg_list = get_degree_list(gamma, lo, hi, n)
        graph = nx.Graph(nx.configuration_model(deg_list))
        
        end_time = time.perf_counter()
        graphs.append(graph)
        timings.append(end_time - start_time)
    
    return graphs, timings


########### SBM GRAPHS ###########
# Generates a list of sbm graphs
def generate_sbm_graphs(num_graphs, n):
    graphs = []
    timings = []
    for _ in tqdm(range(num_graphs), desc='Generating graphs'):
        # create p matrix
        num_blocks = SBM_NUM_BLOCKS_DICT[n]
        p_matrix = np.ones((num_blocks, num_blocks)) * SBM_P_ACROSS + \
            np.eye(num_blocks) * (SBM_P_WITHIN - SBM_P_ACROSS)

        start_time = time.perf_counter()

        graph = nx.stochastic_block_model(
            [SBM_BLOCK_SIZE] * num_blocks, p_matrix
        )

        end_time = time.perf_counter()
        graphs.append(graph)
        timings.append(end_time - start_time)
    
    return graphs, timings


########### SAMPLING ###########
# Generates a list of random samples
def generate_random_samples(graphs, sample_size, samples_per_graph):
    all_random_samples = []
    timings = []
    for graph in tqdm(graphs, desc='Graphs to sample'):
        graph_random_samples = []
        for _ in range(samples_per_graph):
            start_time = time.perf_counter()
            sample = list(np.random.choice(list(graph), size=sample_size,
                                           replace=False))
            end_time = time.perf_counter()
            graph_random_samples.append(sample)
            timings.append(end_time - start_time)
        all_random_samples.append(graph_random_samples)
    
    return all_random_samples, timings

# Generates a single snowball sample on the given graph using given seeds
def generate_snowball_sample(graph, seeds, sample_size):
    sample = []
    remaining_seeds = seeds
    visited = set()
    selection_queue = deque()
    
    while len(sample) < sample_size:
        if not selection_queue:
            # add new seed
            if len(remaining_seeds) > 0:
                # find next unvisited seed
                new_seed = remaining_seeds[-1]
                remaining_seeds = remaining_seeds[:-1]
                while new_seed in visited:
                    if len(remaining_seeds) == 0:
                        # no new seeds left
                        return sample
                    new_seed = remaining_seeds[-1]
                    remaining_seeds = remaining_seeds[:-1]
                
                # found new seed
                sample.append(new_seed)
                visited.add(new_seed)
                for n in graph[new_seed]:
                    if n not in visited:
                        selection_queue.append(n)
                        visited.add(n)
            else:
                # no new seeds left
                return sample
        else:
            # do next in selection queue
            candidate = selection_queue.popleft()
            if np.random.sample() < 0.5:
                sample.append(candidate)
                for n in graph[candidate]:
                    if n not in visited:
                        selection_queue.append(n)
                        visited.add(n)

    return sample

# Generates a list of snowball samples
def generate_snowball_samples(graphs, sample_size, samples_per_graph):
    all_snowball_samples = []
    timings = []
    for graph in tqdm(graphs, desc='Graphs to sample'):
        graph_snowball_samples = []
        for _ in range(samples_per_graph):
            start_time = time.perf_counter()
            seeds = list(np.random.choice(list(graph), size=sample_size,
                                          replace=False))
            sample = generate_snowball_sample(graph, seeds, sample_size)
            end_time = time.perf_counter()
            graph_snowball_samples.append(sample)
            timings.append(end_time - start_time)
        all_snowball_samples.append(graph_snowball_samples)
    
    return all_snowball_samples, timings


########### CALCULATING DENSITIES ###########
# Returns the distance density for a single graph
def get_single_density(graph, sample):
    # bfs
    distances = [0] * MAX_DIST

    # initialize frontier and explored
    explored = set()
    frontier = deque()
    for node in sample:
        frontier.append((node, 0))
        explored.add(node)

    # bfs search
    while frontier:
        curr_node, distance = frontier.popleft()

        # updated distances
        if distance < len(distances):
            distances[distance] += 1
        
        # update frontier
        for neighbor in graph[curr_node]:
            if neighbor not in explored:
                frontier.append((neighbor, distance+1))
                explored.add(neighbor)
    
    # calculate density
    density = np.array(distances) / len(graph)

    return density

# Returns the distance density over all graphs
def get_density(graphs, samples):
    assert(len(graphs) == len(samples))
    cumsum_density = np.zeros(MAX_DIST)
    all_densities = []
    timings = []
    count = 0

    for i in tqdm(range(len(graphs)), position=0, desc='Graphs'):
        graph = graphs[i]
        graph_samples = samples[i]
        graph_densities = []
        for sample in tqdm(graph_samples, position=1, leave=False,
                           desc='Samples per graph'):
            start_time = time.perf_counter()
            this_density = get_single_density(graph, sample)
            end_time = time.perf_counter()
            graph_densities.append(this_density)
            timings.append(end_time - start_time)
            cumsum_density = cumsum_density + this_density
            count += 1
        all_densities.append(graph_densities)

    return cumsum_density / count, all_densities, timings


########### MAIN FUNCTION ###########
def obtain_empirical(args):
    # set seed
    if args.rand_seed is not None:
        np.random.seed(args.rand_seed)
        random.seed(args.rand_seed)

    graphs = None
    samples = None

    # obtain graphs
    print('Obtaining graphs')
    if args.load_graphs:
        graphs = load(f'{args.load_graphs}.pkl')
    else:
        if args.graph_type == 'binomial':
            graphs, graph_timings = \
                generate_binomial_graphs(args.num_graphs, args.graph_size)
        elif args.graph_type == 'power_a':
            graphs, graph_timings = \
                generate_power_law_graphs(args.num_graphs, POWER_A_GAMMA,
                                          POWER_A_LO, POWER_A_HI,
                                          args.graph_size)
        elif args.graph_type == 'power_b':
            graphs, graph_timings = \
                generate_power_law_graphs(args.num_graphs, POWER_B_GAMMA,
                                          POWER_B_LO, POWER_B_HI,
                                          args.graph_size)
        else:
            assert(args.graph_type == 'sbm')
            graphs, graph_timings = \
                generate_sbm_graphs(args.num_graphs, args.graph_size)
        
        summarize_timings(graph_timings, 'graph gen')

        if args.save_graphs:
            save(graphs, f'{args.save_graphs}.pkl')
            save(graph_timings, f'{args.save_graphs}_timings.pkl')
        

    # obtain samples
    print('Obtaining samples')
    if args.load_samples:
        samples = load(f'{args.load_samples}.pkl')
    else:
        if args.sample_type == 'random':
            samples, sample_timings = \
                generate_random_samples(graphs, args.sample_size,
                                        args.samples_per_graph)
        else:
            assert(args.sample_type == 'snowball')
            samples, sample_timings = \
                generate_snowball_samples(graphs, args.sample_size,
                                          args.samples_per_graph)
        
        summarize_timings(sample_timings, 'sample gen')

        if args.save_samples:
            save(samples, f'{args.save_samples}.pkl')
            save(sample_timings, f'{args.save_samples}_timings.pkl')
        

    # calculate distances
    print('Calculating density')
    density, all_densities, density_timings = get_density(graphs, samples)
    
    summarize_timings(density_timings, 'density calc')

    if args.save_densities:
        save((density, all_densities), f'{args.save_densities}.pkl')
        save(density_timings, f'{args.save_densities}_timings.pkl')
    plot(density, fname=args.save_plot)


########### ARGUMENT PARSING ###########
def print_args(args):
    print('=' * 80)
    for k, v in args.__dict__.items():
        print(f'    - {k} : {v}')
    print('=' * 80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='empirical_distribution.py')
    # graph settings
    parser.add_argument('--load_graphs', default=None)
    parser.add_argument('--graph_type',
                        choices=['binomial', 'power_a', 'power_b', 'sbm'])
    parser.add_argument('--graph_size', type=int, default=20000)
    parser.add_argument('--num_graphs', type=int, default=1)
    parser.add_argument('--save_graphs', default=None)
    
    # sample settings
    parser.add_argument('--load_samples', default=None)
    parser.add_argument('--sample_type', choices=['random', 'snowball'])
    parser.add_argument('--sample_size', type=int, default=200)
    parser.add_argument('--samples_per_graph', type=int, default=1)
    parser.add_argument('--save_samples', default=None)

    # dist settings
    parser.add_argument('--save_densities', default=None)
    parser.add_argument('--save_plot', default=None)

    # rand settings
    parser.add_argument('--rand_seed', type=int, default=1000)

    args = parser.parse_args()
    print_args(args)

    obtain_empirical(args)
