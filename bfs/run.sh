#!/bin/bash

DATASET="../datasets/roadNet-CA.txt"
DATASET2="../datasets/soc-LiveJournal1.txt"
EXEC="../build/bfs_parallel"

SOURCES=(0 423578 639908 1655248)
SOURCES2=(0 315472 1202500 4029991)
TASKS=(1 2 4 8 16 32 64)

for t in "${TASKS[@]}"; do
    echo "============== NODES$t"

    for src in "${SOURCES[@]}"; do
        srun -N 1 --ntasks-per-node=$t $EXEC $DATASET $src
        echo
    done

    echo
    echo
done

for t in "${TASKS[@]}"; do
    echo "============== NODES$t"

    for src in "${SOURCES2[@]}"; do
        srun -N 1 --ntasks-per-node=$t $EXEC $DATASET2 $src
        echo
    done

    echo
    echo
done
