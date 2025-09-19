#!/usr/bin/env bash
tasks=("nt2")

cd ADFAR/src

for task in "${tasks[@]}"
do
    echo "Running task: $task"
    python run_nt2.py --task $task
done

echo "All tasks completed."
