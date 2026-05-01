#!/usr/bin/env bash
tasks=("nt1")

cd ADFAR/src

for task in "${tasks[@]}"
do
    echo "Running task: $task"
    python run.py --task $task
done

echo "All tasks completed."
