#!/usr/bin/env bash

# tasks=( "H3" "H3K14ac" "H3K36me3" "H3K4me1" "H3K4me2" "H3K4me3" "H3K79me3" "H3K9ac" "H4" "H4ac" "prom_core_all" "prom_core_notata" "prom_core_tata" "prom_300_all" "prom_300_notata" "prom_300_tata" "tf0" "tf1" "tf2" "tf3" "tf4" "0" "1" "2" "3" "4")
tasks=("1")

cd /projects/p32013/DNABERT-meta/ADFAR/src

for task in "${tasks[@]}"
do
    echo "Running task: $task"
    python run.py --task $task
done

echo "All tasks completed."
