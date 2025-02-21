# Distribution of Shortest Path Distances to Sample

This repository contains code implementing and verifying a framework for effeciently and accurately estimating the distribution of shortest path distances (DSPD) to a sample selected from a graph. Currently, the framework supports estimating the distribution from a binomial, power law, or SBM graph using random or snowball sampling.

## Code

The code in this repository requires SciPy, NumPy, NetworkX, tqdm, and matplotlib, all of which can be installed with `pip`.
* `src` contains code implementing the proposed framework and experiments.
    * `constants.py` contains parameters specifying the configuration of graphs in the experiments.
    * `util.py` contains utility functions, such as for pickling or plotting.
    * `empirical_distribution.py` contains code for generating graphs, sampling from graphs, and computing empirical DSPDs on those graphs with those samples.
    * `estimated_distribution.py` contains code implementing the proposed framework for estimating DSPDs given graph configuration, sample type, and sample size.
    * `analysis.py` contains code generating the plots and tables in the paper.
* `ipynb/fb.ipynb` is a notebook detailing the analysis of the [Facebook Large Page-Page Network graph][1] used to obtain graph configuration parameters for experiments on Stochastic Block Model graphs.
* `scripts` contains shell scripts used to run the experiments and generate analysis plots and tables.
    * `timing.sh` runs a single experiment using the parameters specified at the top of the file.
    * `all_timings.sh` runs all experiments in sequence. Use this to replicate all the experiments reported in the paper.

[1]: https://snap.stanford.edu/data/facebook-large-page-page-network.html
