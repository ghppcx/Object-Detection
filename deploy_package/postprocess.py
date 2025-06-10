import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help='推理输出txt')
parser.add_argument('--image', type=str, required=True, help='原始图片')
parser.add_argument('--output', type=str, required=True, help='输出图片')
args = parser.parse_args()

# 读取推理结果（需根据实际输出格式解析）
outputs = np.loadtxt(args.input)
# TODO: 解析outputs，获得检测框、类别、置信度
# 示例：假设每行[x1, y1, x2, y2, conf, cls]
boxes = outputs.reshape(-1, 6)
img = cv2.imread(args.image)
for box in boxes:
    x1, y1, x2, y2, conf, cls = box
    if conf < 0.3:
        continue
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
    cv2.putText(img, f'{int(cls)}:{conf:.2f}', (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
cv2.imwrite(args.output, img) 