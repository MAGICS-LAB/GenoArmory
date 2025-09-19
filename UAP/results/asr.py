import os
import json
import csv

# 设置你的results路径
results_dir = "UAP/results"
output_csv = "asr_results.csv"

# 准备输出数据
data = [("task_name", "ASR")]

# 遍历所有子文件夹
for folder in sorted(os.listdir(results_dir)):
    folder_path = os.path.join(results_dir, folder)
    if os.path.isdir(folder_path):
        # 查找文件夹内的json文件
        for file in os.listdir(folder_path):
            if file.endswith(".json"):
                json_path = os.path.join(folder_path, file)
                try:
                    with open(json_path, 'r') as f:
                        content = json.load(f)

                    clean_acc = content["clean_validation_metrics"]["accuracy"]
                    perturbed_acc = content["perturbed_validation_metrics"]["accuracy"]
                    task_name = content.get("task_name", folder)  # 如果json里没有，就用文件夹名

                    if clean_acc != 0:
                        asr = (clean_acc - perturbed_acc) / clean_acc
                    else:
                        asr = None  # 防止除以0

                    data.append((task_name, asr))

                except Exception as e:
                    print(f"Error reading {json_path}: {e}")

# 写入CSV文件
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(data)

print(f"ASR results written to {output_csv}")
