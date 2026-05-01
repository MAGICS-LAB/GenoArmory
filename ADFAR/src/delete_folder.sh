#!/usr/bin/env bash

# 设置目标路径
base_dir="ADFAR/src/experiments/GUE"

# 遍历 base_dir 下所有子文件夹
for dir in "$base_dir"/*/; do
    # 删除 nt1 文件夹（如果存在）
    if [ -d "${dir}nt2" ]; then
        echo "Deleting ${dir}nt2"
        rm -rf "${dir}nt2"
    fi

    # # 删除 og 文件夹（如果存在）
    # if [ -d "${dir}dnabert" ]; then
    #     echo "Deleting ${dir}dnabert"
    #     rm -rf "${dir}dnabert"
    # fi
done
