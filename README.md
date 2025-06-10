# 网球目标检测部署包说明

## 1. 项目简介
本文件夹为"基于多模态感知的智能网球回收机器人系统"目标检测部分的端侧部署包，适用于昇腾310B（如香橙派AIpro 20T）等设备。支持高效、自动化的网球检测推理，满足大赛评测接口要求。

## 2. 评测接口
- **process.py** 实现了官方要求的 `process_img(img_path)` 函数，评测系统会自动调用。
- 返回格式为 `[{"x":..., "y":..., "w":..., "h":...}, ...]`，与官方要求一致。

## 3. 一键推理流程
- `process_img(img_path)` 会自动完成图片预处理、om模型推理、后处理，无需手动分步操作。
- 支持批量测试，可参考如下代码：

```python
import os
from process import process_img
imgs_folder = './imgs/'
img_paths = os.listdir(imgs_folder)
for img_path in img_paths:
    result = process_img(os.path.join(imgs_folder, img_path))
    print(img_path, result)
```

## 4. 环境准备
- 板端需已安装昇腾CANN runtime，配置好环境变量：
  `source /usr/local/Ascend/ascend-toolkit/set_env.sh`
- 板端需有python3、numpy（pyACL已随CANN自带）
- PC端本地测试依赖见 requirements.txt

## 5. 文件说明
- **process.py**      官方评测入口，集成全流程
- **preprocess.py**   图片转bin
- **run_infer.py**    om模型推理（pyACL）
- **postprocess.py**  可选后处理/画框
- **requirements.txt**  PC端依赖
- **model1.om**       om模型文件
- **test.jpg**        测试图片
- **test.bin**        测试图片bin（可选）
- **result.txt**      推理输出（可选）

## 6. 依赖安装（PC端）
```bash
pip install -r requirements.txt
```

## 7. 注意事项
- pyACL/昇腾相关依赖由CANN自带，无需pip安装
- 推理和评测需在板端（如香橙派）环境下进行
- 如需自定义后处理、可视化等，可修改 postprocess.py

---
如有问题请联系项目作者或在GitHub仓库提交issue。
