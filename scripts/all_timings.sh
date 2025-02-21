declare -a sample_sizes=(200 400 1000)
declare -a graph_sizes=(20000 40000 100000)
declare -a sample_types=("random" "snowball")
declare -a graph_types=("binomial" "power_a" "power_b" "sbm")

for graph_type in "${graph_types[@]}"
do
    mkdir -p pickle/timing/$graph_type/
    mkdir -p output/timing/$graph_type/

    for graph_size in "${graph_sizes[@]}"
    do
        for sample_type in "${sample_types[@]}"
        do
            for sample_size in "${sample_sizes[@]}"
            do
                echo $graph_type $graph_size $sample_type $sample_size

                if [ "$sample_type" = "random" ] && [ "$sample_size" = "200" ]
                then
                    # create graphs if first use of graph setting
                    python -u src/empirical_distribution.py \
                    --num_graphs 5 \
                    --graph_type $graph_type \
                    --graph_size $graph_size \
                    --sample_type $sample_type \
                    --sample_size $sample_size \
                    --samples_per_graph 20 \
                    --save_graphs pickle/timing/$graph_type/graphs_${graph_size} \
                    --save_samples pickle/timing/$graph_type/size_${graph_size}_${sample_type}_${sample_size}_samples \
                    --save_densities pickle/timing/$graph_type/size_${graph_size}_${sample_type}_${sample_size}_dists \
                    --save_plot output/timing/$graph_type/size_${graph_size}_${sample_type}_${sample_size} \
                    --rand_seed 1000 \
                    > output/timing/$graph_type/size_${graph_size}_${sample_type}_${sample_size}_timing.txt
                else
                    # otherwise load previously generated graph
                    python -u src/empirical_distribution.py \
                    --num_graphs 5 \
                    --graph_type $graph_type \
                    --graph_size $graph_size \
                    --sample_type $sample_type \
                    --sample_size $sample_size \
                    --samples_per_graph 20 \
                    --load_graphs pickle/timing/$graph_type/graphs_${graph_size} \
                    --save_samples pickle/timing/$graph_type/size_${graph_size}_${sample_type}_${sample_size}_samples \
                    --save_densities pickle/timing/$graph_type/size_${graph_size}_${sample_type}_${sample_size}_dists \
                    --save_plot output/timing/$graph_type/size_${graph_size}_${sample_type}_${sample_size} \
                    --rand_seed 1000 \
                    > output/timing/$graph_type/size_${graph_size}_${sample_type}_${sample_size}_timing.txt
                fi

                python -u src/estimated_distribution.py \
                --graph_type $graph_type \
                --graph_size $graph_size \
                --sample_size $sample_size \
                --sample_type $sample_type \
                --repeat 100 \
                --save_dist pickle/timing/$graph_type/size_${graph_size}_${sample_type}_${sample_size}_dists_est \
                --save_plot output/timing/$graph_type/size_${graph_size}_${sample_type}_${sample_size}_est \
                >> output/timing/$graph_type/size_${graph_size}_${sample_type}_${sample_size}_timing.txt
            done
        done
    done
done

