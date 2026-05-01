import os
import json
import csv
import re

def normalize(name: str) -> str:
    """
    小写并去掉下划线，方便匹配：
    'mouse_0' -> 'mouse0'
    """
    return re.sub(r"_", "", name).lower()

def update_bertattack_from_csv(json_path: str, csv_path: str):
    """
    从 attack_results.csv 读取 origin_acc 和 aft_atk_acc，
    更新 metadata/BERTAttack/DNABERT2.json 中的 orign_acc 和 after_attack_acc。
    """
    # 1) 读 CSV，构建 {datasets: (orig, aft)} 映射
    mapping = {}
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = os.path.splitext(row['filename'])[0]  # 去掉 .json
            try:
                orig = float(row['origin_acc'])
                aft  = float(row['aft_atk_acc'])
            except ValueError:
                continue
            mapping[key] = (orig, aft)

    if not mapping:
        print("ERROR: 没有从 CSV 读取到任何数据")
        return

    # 2) 加载 JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 3) 更新每条攻击记录
    updated = 0
    for item in data.get("adversarial_defense", []):
        ds = item.get("datasets", "")
        # 先尝试完全匹配
        if ds in mapping:
            orig, aft = mapping[ds]
        else:
            # 再尝试 normalize 后缀匹配
            ds_norm = normalize(ds)
            match = None
            for k, (o, a) in mapping.items():
                if ds_norm == normalize(k):
                    match = (o, a)
                    break
                if ds_norm.endswith(normalize(k)):
                    match = (o, a)
                    break 
            if match is None:
                print(f"No match for '{ds}' in CSV")
                continue
            orig, aft = match

        item["origin_acc"]        = orig
        item["after_attack_acc"] = aft
        updated += 1
        print(f"Updated {ds}: orign_acc={orig}, after_attack_acc={aft}")

    print(f"Total entries updated: {updated}")

    # 4) 写回 JSON
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"✔ 已写入 {json_path}")

# 调用示例
json_file = "metadata/FREELB/hyena-BERTAttack.json"
csv_file  = "BERT-Attack/results/freelb/hyena/attack_results.csv"
update_bertattack_from_csv(json_file, csv_file)
