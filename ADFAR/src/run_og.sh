#!/usr/bin/env bash
tasks=("og")

cd ADFAR/src

for task in "${tasks[@]}"
do
    echo "Running task: $task"
    python run_og.py --task $task
done

echo "All tasks completed."
