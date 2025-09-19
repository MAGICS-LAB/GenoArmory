import json
import os

def compute_relative_asr(json_path):
    """使用相对下降公式计算 ASR"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    batch0 = data['attack_results']['batch_results'][0]
    original_acc = batch0['original_accuracy']
    adversarial_acc = batch0['adversarial_accuracy']

    # 使用公式 ASR = (orig - adv) / orig * 100%
    asr = (original_acc - adversarial_acc) / original_acc * 100 if original_acc != 0 else 0

    return {
        'filename': os.path.basename(json_path),
        'original_accuracy': original_acc,
        'adversarial_accuracy': adversarial_acc,
        'asr_percent': asr
    }

def process_json_folder(folder_path):
    """遍历文件夹中所有 JSON 文件并计算 ASR"""
    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            path = os.path.join(folder_path, filename)
            try:
                result = compute_relative_asr(path)
                results.append(result)
            except Exception as e:
                print(f"[ERROR] {filename}: {e}")
    return results

# 示例运行
if __name__ == "__main__":
    folder = "/projects/p32013/DNABERT-meta/auto-attack/results"  # 替换为你的 JSON 文件夹路径
    results = process_json_folder(folder)
    
    for r in results:
        print(f"{r['filename']}: ASR = {r['asr_percent']:.2f}% (Original = {r['original_accuracy']:.4f}, Adversarial = {r['adversarial_accuracy']:.4f})")
        with open('asr_results.txt', 'a', encoding='utf-8') as f:
            f.write(f"{r['filename']}: ASR = {r['asr_percent']:.2f}% (Original = {r['original_accuracy']:.4f}, Adversarial = {r['adversarial_accuracy']:.4f})\n")
    print("ASR results saved to asr_results.txt")
            
        
