import os
import numpy as np
import subprocess

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