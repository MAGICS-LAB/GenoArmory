import os
import json
import pandas as pd

# 设置主路径
root_dir = "UAP/results"

# 存储结果
asr_results = []
print("strat calculating ASR...")
# 遍历每个子文件夹
for subfolder in os.listdir(root_dir):
    subfolder_path = os.path.join(root_dir, subfolder)
    if os.path.isdir(subfolder_path):
        json_path = os.path.join(subfolder_path, "eval_results.json")
        if os.path.isfile(json_path):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                
                clean_acc = data["clean_validation_metrics"]["accuracy"]
                perturbed_acc = data["perturbed_validation_metrics"]["accuracy"]
                asr = (clean_acc - perturbed_acc) / clean_acc if clean_acc != 0 else None

                asr_results.append({
                    "task_name": subfolder,
                    "clean_accuracy": clean_acc,
                    "perturbed_accuracy": perturbed_acc,
                    "ASR": round(asr, 4)
                })

            except Exception as e:
                print(f"Error processing {json_path}: {e}")

# 输出为 DataFrame
asr_df = pd.DataFrame(asr_results)

# 展示结果
print(asr_df)

# 保存为 CSV（可选）
asr_df.to_csv("asr_results.csv", index=False)
