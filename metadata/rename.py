#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json

def update_json_file(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. 重命名 attack_success_rate -> defense_success_rate
    params = data.get("parameters", {})
    if "attack_success_rate" in params:
        params["defense_success_rate"] = params.pop("attack_success_rate")

    # 2. 重命名 adversarial_attacks -> adversarial_defense
    if "adversarial_attacks" in data:
        data["adversarial_defense"] = data.pop("adversarial_attacks")

        # 3. 在每个条目中重命名 asr -> dsr
        for entry in data["adversarial_defense"]:
            if "asr" in entry:
                entry["dsr"] = entry.pop("asr")

    # 将修改后的内容写回文件
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Updated {os.path.basename(path)}")

def main():
    folder = "metadata/AT"
    for filename in os.listdir(folder):
        if filename.lower().endswith(".json"):
            fullpath = os.path.join(folder, filename)
            update_json_file(fullpath)

if __name__ == "__main__":
    main()
