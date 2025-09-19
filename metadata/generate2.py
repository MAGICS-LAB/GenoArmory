import json
import re
import os

def normalize(name: str) -> str:
    """
    小写并去掉下划线和连字符，方便匹配：
    'mouse_0' -> 'mouse0', 'tf_0' -> 'tf0'
    """
    return re.sub(r"[_\-]", "", name).lower()

def update_orign_acc_from_baseline(json_file: str, baseline_str: str):
    """
    从 baseline_str 中解析出每个键对应的 baseline 值，
    并更新 json_file 中每个 adversarial_attacks 项的 orign_acc。
    """
    # 1) 解析 baseline_str
    lines = baseline_str.strip().splitlines()
    header = lines[0]
    entries = lines[1:]
    baseline_map = {}
    for line in entries:
        parts = re.split(r"\s+", line.strip())
        if len(parts) >= 2:
            key = parts[0]
            try:
                val = float(parts[1])
                baseline_map[key] = val
            except ValueError:
                # 跳过无法解析的行
                continue

    # 2) 读取 JSON
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 3) 更新 orign_acc
    updated = 0
    for item in data.get("adversarial_attacks", []):
        ds = item.get("datasets", "")
        ds_norm = normalize(ds)
        matched_val = None

        # 完全匹配
        for key, val in baseline_map.items():
            if normalize(key) == ds_norm:
                matched_val = val
                break

        # 后缀匹配
        if matched_val is None:
            for key, val in baseline_map.items():
                if ds_norm.endswith(normalize(key)):
                    matched_val = val
                    break

        if matched_val is not None:
            item["orign_acc"] = matched_val
            updated += 1
        else:
            print(f"No baseline match for dataset '{ds}'")

    print(f"Total orign_acc updated: {updated}")

    # 4) 写回 JSON
    backup = json_file + ".bak"
    os.replace(json_file, backup)  # 先备份原文件
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Updated JSON written to {json_file} (backup saved as {backup})")

baseline_data = """
Task    baseline
0       0.899286
1       0.717143
2       0.717143
3       0.732647
4       0.8632
H3      0.905333
H3K14ac 0.915
H3K36me3    0.896538
H3K4me1 0.939286
H3K4me2 0.844194
H3K4me3 0.93625
H3K79me3    0.9208
H3K9ac  0.965
H4      0.902857
H4ac    0.542
prom_300_all    0.941667
prom_300_notata 0.964583
prom_300_tata  0.8
prom_core_all   0.8689
prom_core_notata    0.8512
prom_core_tata  0.896667
tf0     0.542941
tf1     0.489375
tf2     0.4705
tf3     0.7455
tf4     0.485172
"""

json_path = "metadata/PGD/og.json"
update_orign_acc_from_baseline(json_path, baseline_data)
