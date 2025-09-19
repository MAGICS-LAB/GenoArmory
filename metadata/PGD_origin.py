#!/usr/bin/env python3
# update_orign_acc.py

import json
import shutil

# 源文件和目标文件路径
src_path = 'metadata/BERTAttack/og.json'
dst_path = 'metadata/FREELB/og-PGD.json'

# 先备份目标文件
shutil.copyfile(dst_path, dst_path + '.bak')

# 读取两个 JSON
with open(src_path, 'r', encoding='utf-8') as f:
    src = json.load(f)

with open(dst_path, 'r', encoding='utf-8') as f:
    dst = json.load(f)

# 构建 index -> orign_acc 的映射
orign_acc_map = {
    entry['index']: entry['orign_acc']
    for entry in src.get('adversarial_attacks', [])
}

# 遍历目标文件中的攻击记录并更新 orign_acc
for entry in dst.get('adversarial_attacks', []):
    idx = entry.get('index')
    if idx in orign_acc_map:
        entry['orign_acc'] = orign_acc_map[idx]

# 将修改后的内容写回目标文件
with open(dst_path, 'w', encoding='utf-8') as f:
    json.dump(dst, f, ensure_ascii=False, indent=4)

print(f'更新完成，原文件已备份为 {dst_path}.bak')
