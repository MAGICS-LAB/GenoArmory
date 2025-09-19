import os
import re
import json
import sys

def parse_final_accuracy(path):
    """
    从任意文本文件中提取 'Final Accuracy: X.XXXX'，返回浮点数 X.XXXX
    """
    text = open(path, 'r').read()
    m = re.search(r"Final Accuracy:\s*([\d.]+)", text)
    if not m:
        raise ValueError(f"No 'Final Accuracy' in {path}")
    return float(m.group(1))

def normalize(name):
    """
    小写并去掉下划线和连字符，方便匹配
    e.g. 'mouse_0' -> 'mouse0'
    """
    return re.sub(r"[_\-]", "", name).lower()

def main():
    results_dir = "PGD/results/FreeLB/dnabert"
    json_file   = "metadata/FREELB/DNABERT2-PGD.json"

    # 1) 收集所有 final accuracy
    parsed = {}
    if not os.path.isdir(results_dir):
        print(f"ERROR: 找不到目录 {results_dir}", file=sys.stderr)
        sys.exit(1)

    for root, dirs, files in os.walk(results_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                acc = parse_final_accuracy(fpath)
                key = os.path.basename(root)
                parsed[key] = acc
                print(f"Found final accuracy {acc} in folder {key}")
                # 一旦在该目录中找到，就跳过同目录下其它文件
                break
            except ValueError:
                continue

    if not parsed:
        print("WARNING: 没有在任何子目录中找到 'Final Accuracy'，请检查文件内容和路径。", file=sys.stderr)

    # 2) 加载并更新 JSON
    with open(json_file, 'r') as f:
        data = json.load(f)

    updated = 0
    for item in data.get("adversarial_attacks", []):
        ds = item.get("datasets", "")
        ds_norm = normalize(ds)
        match_acc = None

        # 完全匹配优先
        for folder, acc in parsed.items():
            if normalize(folder) == ds_norm:
                match_acc = acc
                print(f"Exact match {ds} ← {folder}: {acc}")
                break

        # 后缀匹配
        if match_acc is None:
            for folder, acc in parsed.items():
                if ds_norm.endswith(normalize(folder)):
                    match_acc = acc
                    print(f"Suffix match {ds} ← {folder}: {acc}")
                    break

        if match_acc is not None:
            item["after_attack_acc"] = match_acc
            updated += 1
        else:
            print(f"No accuracy found for {ds}")

    print(f"Total entries updated: {updated}")

    # 3) 写回 JSON
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)
    print("✔ 已将更新写回：", json_file)

if __name__ == "__main__":
    main()
