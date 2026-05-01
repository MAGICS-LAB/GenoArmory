import os
import re
import json
import sys

def parse_results_log(path):
    """
    从 results_log 文件中提取指标：
      - original accuracy
      - adv accuracy
      - num of queries
    返回三个浮点数：orign_acc, after_attack_acc, average_queries
    """
    text = open(path, 'r').read()
    m = re.search(
        r"original accuracy: ([\d.]+)%, adv accuracy: ([\d.]+)%, .*num of queries: ([\d.]+)",
        text
    )
    if not m:
        raise ValueError(f"No match in {path}")
    orign_acc        = float(m.group(1)) / 100.0
    after_attack_acc = float(m.group(2)) / 100.0
    average_queries  = float(m.group(3))
    return orign_acc, after_attack_acc, average_queries

def normalize(name):
    """
    小写并去掉下划线和连字符，方便匹配
    例如 'mouse_0' -> 'mouse0'
    """
    return re.sub(r"[_\-]", "", name).lower()

def main():
    base_dir = "TextFooler/output/FreeLB/og"
    json_file = "metadata/FREELB/og-TextFooler.json"

    if not os.path.isdir(base_dir):
        print(f"ERROR: 找不到目录 {base_dir}", file=sys.stderr)
        sys.exit(1)

    # 1. 解析所有子文件夹里的 results_log
    parsed = {}
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        log_path = os.path.join(folder_path, "results_log")
        if os.path.isdir(folder_path) and os.path.isfile(log_path):
            try:
                orign_acc, after_attack_acc, average_queries = parse_results_log(log_path)
                parsed[folder] = {
                    "orign_acc":        orign_acc,
                    "after_attack_acc": after_attack_acc,
                    "average_queries":  average_queries
                }
                print(f"Found {folder}: {parsed[folder]}")
            except Exception as e:
                print(f"Failed to parse {log_path}: {e}", file=sys.stderr)

    if not parsed:
        print("WARNING: 没有找到任何 results_log 文件。请检查子文件夹和文件名。", file=sys.stderr)

    # 2. 读取并更新 JSON
    with open(json_file, 'r') as f:
        data = json.load(f)

    updated = 0
    for item in data.get("adversarial_attacks", []):
        ds = item.get("datasets", "")
        ds_norm = normalize(ds)
        metrics = None

        # 完全匹配优先
        for folder, met in parsed.items():
            if normalize(folder) == ds_norm:
                metrics = met
                print(f"Exact match {ds} ← {folder}")
                break

        # 后缀匹配
        if metrics is None:
            for folder, met in parsed.items():
                if ds_norm.endswith(normalize(folder)):
                    metrics = met
                    print(f"Suffix match {ds} ← {folder}")
                    break

        if metrics:
            item.update(metrics)
            updated += 1
        else:
            print(f"No metrics found for {ds}")

    print(f"Total entries updated: {updated}")

    # 3. 写回 JSON
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)
    print("✔ JSON 已更新并写回：", json_file)

if __name__ == "__main__":
    main()
