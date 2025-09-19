#!/usr/bin/env python3
# update_og_average.py

import os
import re
import json
import sys

def normalize(name):
    """
    小写并去掉下划线和连字符，便于匹配
    """
    return re.sub(r'[_\-]', '', name).lower()

def compute_average_query(json_path):
    """
    读取 JSON 列表，提取 'query' 字段，计算平均值
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    total, count = 0.0, 0
    for entry in data:
        q = entry.get('query')
        try:
            total += float(q)
            count += 1
        except (TypeError, ValueError):
            pass
    return (total / count) if count > 0 else None

def main():
    # 根据实际情况修改这两行路径
    results_dir = 'BERT-Attack/results/hyena'
    meta_file   = 'metadata/BERTAttack/hyena.json'

    if not os.path.isdir(results_dir):
        print(f'ERROR: 目录不存在 {results_dir}', file=sys.stderr)
        sys.exit(1)

    # 1) 遍历子目录，计算每个文件夹同名 JSON 的平均 query
    averages = {}
    for folder in os.listdir(results_dir):
        subdir = os.path.join(results_dir, folder)
        if not os.path.isdir(subdir):
            continue
        js = os.path.join(subdir, f'{folder}.json')
        if not os.path.isfile(js):
            print(f'Warning: 未找到 {js}', file=sys.stderr)
            continue

        avg = compute_average_query(js)
        if avg is not None:
            averages[folder] = avg
            print(f'Found average_queries {avg:.4f} in folder {folder}')
        else:
            print(f'Warning: 在 {js} 中未找到有效的 query', file=sys.stderr)

    if not averages:
        print('WARNING: 未计算到任何平均值，请检查结果目录。', file=sys.stderr)

    # 2) 加载元数据并更新 average_queries
    with open(meta_file, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    updated = 0
    for item in meta.get('adversarial_attacks', []):
        ds      = item.get('datasets', '')
        ds_norm = normalize(ds)
        match_avg = None

        # —— 完全匹配优先 ——  
        for folder, avg in averages.items():
            if normalize(folder) == ds_norm:
                match_avg = avg
                print(f'Exact match {ds} ← {folder}: {avg:.4f}')
                break

        # —— 如果完全匹配失败，再做后缀匹配 ——  
        if match_avg is None:
            for folder, avg in averages.items():
                if ds_norm.endswith(normalize(folder)):
                    match_avg = avg
                    print(f'Suffix match {ds} ← {folder}: {avg:.4f}')
                    break

        if match_avg is not None:
            item['average_queries'] = round(match_avg, 4)
            updated += 1
        else:
            print(f'No average found for {ds}', file=sys.stderr)

    print(f'Total entries updated: {updated}')

    # 3) 写回元数据文件
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f'✔ 已将更新写回：{meta_file}')

if __name__ == '__main__':
    main()
