export SP_SAMPLE_SIZE=1000
export SP_GRAPH_SIZE=20000
export SP_SAMPLE_TYPE=random
export SP_GRAPH_TYPE=binomial

mkdir -p pickle/timing/$SP_GRAPH_TYPE/

python -u src/empirical_distribution.py \
--num_graphs 5 \
--graph_type $SP_GRAPH_TYPE \
--graph_size $SP_GRAPH_SIZE \
--sample_type $SP_SAMPLE_TYPE \
--sample_size $SP_SAMPLE_SIZE \
--samples_per_graph 20 \
--load_graphs pickle/timing/$SP_GRAPH_TYPE/graphs_${SP_GRAPH_SIZE} \
--save_samples pickle/timing/$SP_GRAPH_TYPE/size_${SP_GRAPH_SIZE}_${SP_SAMPLE_TYPE}_${SP_SAMPLE_SIZE}_samples \
--save_densities pickle/timing/$SP_GRAPH_TYPE/size_${SP_GRAPH_SIZE}_${SP_SAMPLE_TYPE}_${SP_SAMPLE_SIZE}_dists \
--save_plot output/timing/$SP_GRAPH_TYPE/size_${SP_GRAPH_SIZE}_${SP_SAMPLE_TYPE}_${SP_SAMPLE_SIZE} \
--rand_seed 1000 \
> output/timing/$SP_GRAPH_TYPE/size_${SP_GRAPH_SIZE}_${SP_SAMPLE_TYPE}_${SP_SAMPLE_SIZE}_timing.txt

python -u src/estimated_distribution.py \
--graph_type $SP_GRAPH_TYPE \
--graph_size $SP_GRAPH_SIZE \
--sample_size $SP_SAMPLE_SIZE \
--sample_type $SP_SAMPLE_TYPE \
--repeat 100 \
--save_dist pickle/timing/$SP_GRAPH_TYPE/size_${SP_GRAPH_SIZE}_${SP_SAMPLE_TYPE}_${SP_SAMPLE_SIZE}_dists_est \
--save_plot output/timing/$SP_GRAPH_TYPE/size_${SP_GRAPH_SIZE}_${SP_SAMPLE_TYPE}_${SP_SAMPLE_SIZE}_est \
>> output/timing/$SP_GRAPH_TYPE/size_${SP_GRAPH_SIZE}_${SP_SAMPLE_TYPE}_${SP_SAMPLE_SIZE}_timing.txt
