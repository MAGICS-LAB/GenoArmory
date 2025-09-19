import os
import json
import re
import sys

def normalize(name: str) -> str:
    """
    小写并去掉下划线/连字符，方便匹配
    例如 'mouse_0' -> 'mouse0'
    """
    return re.sub(r"[_\-]", "", name).lower()

def parse_all_results(path: str):
    """
    读取 all_results.json，返回 (orign_acc, after_attack_acc)
    """
    with open(path, 'r') as f:
        data = json.load(f)
    m1 = data.get("metrics1", {})
    m2 = data.get("metrics", {})
    if "accuracy" not in m1 or "accuracy" not in m2:
        raise ValueError(f"accuracy 字段缺失 in {path}")
    return m1["accuracy"], m2["accuracy"]

def main():
    results_dir = "fimba-attack/results/og"
    meta_json   = "metadata/FIMBA/og.json"

    # 1) 收集所有子文件夹的指标
    parsed = {}
    if not os.path.isdir(results_dir):
        print(f"ERROR: 找不到目录 {results_dir}", file=sys.stderr)
        sys.exit(1)

    for name in os.listdir(results_dir):
        folder = os.path.join(results_dir, name)
        all_json = os.path.join(folder, "all_results.json")
        if os.path.isdir(folder) and os.path.isfile(all_json):
            try:
                orig_acc, adv_acc = parse_all_results(all_json)
                parsed[name] = {"orign_acc": orig_acc, "after_attack_acc": adv_acc}
                print(f"Loaded {name} → orign_acc={orig_acc}, after_attack_acc={adv_acc}")
            except Exception as e:
                print(f"Fail to parse {all_json}: {e}", file=sys.stderr)

    if not parsed:
        print("WARNING: 没有读取到任何 all_results.json", file=sys.stderr)

    # 2) 加载并更新元数据 JSON
    with open(meta_json, 'r') as f:
        meta = json.load(f)

    updated = 0
    for item in meta.get("adversarial_attacks", []):
        ds = item.get("datasets", "")
        ds_norm = normalize(ds)
        match = None

        # 完全匹配优先
        for key, vals in parsed.items():
            if normalize(key) == ds_norm:
                match = vals
                print(f"Exact match {ds} ← {key}")
                break

        # 后缀匹配
        if match is None:
            for key, vals in parsed.items():
                if ds_norm.endswith(normalize(key)):
                    match = vals
                    print(f"Suffix match {ds} ← {key}")
                    break

        if match:
            item["orign_acc"] = match["orign_acc"]
            item["after_attack_acc"] = match["after_attack_acc"]
            updated += 1
        else:
            print(f"No data for {ds}")

    print(f"Total entries updated: {updated}")

    # 3) 写回文件
    with open(meta_json, 'w') as f:
        json.dump(meta, f, indent=4)
    print(f"✔ 已写回更新至 {meta_json}")

if __name__ == "__main__":
    main()
