import os
import numpy as np
import subprocess
import json

def process_img(img_path):
    # 1. 预处理
    bin_path = img_path + '.bin'
    subprocess.run(['python3', 'preprocess.py', '--input', img_path, '--output', bin_path], check=True)

    # 2. 推理
    result_path = img_path + '.txt'
    subprocess.run(['python3', 'run_infer.py', '--model', 'model1.om', '--input', bin_path, '--output', result_path], check=True)

    # 3. 后处理
    try:
        outputs = np.loadtxt(result_path)
    except Exception:
        return []
    if outputs.size == 0:
        return []
    if outputs.ndim == 1:
        outputs = outputs.reshape(1, -1)
    result = []
    for box in outputs:
        x1, y1, x2, y2, conf, cls = box
        if conf < 0.3:
            continue
        x = int(x1)
        y = int(y1)
        w = int(x2 - x1)
        h = int(y2 - y1)
        result.append({"x": x, "y": y, "w": w, "h": h})
    return result


def batch_process_images(image_dir='test_images', output_json='result.json'):
    result_dict = {}
    for img_name in os.listdir(image_dir):
        if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        img_path = os.path.join(image_dir, img_name)
        boxes = process_img(img_path)
        result_dict[img_name] = boxes
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    batch_process_images() 
