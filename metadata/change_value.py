#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from collections import OrderedDict

def update_file(path):
    # 以 OrderedDict 保留原有顺序
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f, object_pairs_hook=OrderedDict)

    # 更新 defense_success_rate
    params = data.get('parameters', {})
    if 'defense_success_rate' in params:
        old = params['defense_success_rate']
        params['defense_success_rate'] = 1 - old

    # 更新每条记录中的 dsr
    if 'adversarial_defense' in data:
        for entry in data['adversarial_defense']:
            if 'dsr' in entry:
                old = entry['dsr']
                entry['dsr'] = 1 - old

    # 写回，保证顺序与缩进
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f'已更新：{os.path.basename(path)}')

def main():
    folder = "metadata/AT"
    for name in os.listdir(folder):
        if name.lower().endswith('.json'):
            update_file(os.path.join(folder, name))

if __name__ == '__main__':
    main()
