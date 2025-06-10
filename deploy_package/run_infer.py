import acl
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='om模型路径')
parser.add_argument('--input', type=str, required=True, help='输入bin文件路径')
parser.add_argument('--output', type=str, required=True, help='输出txt文件路径')
args = parser.parse_args()

acl.init()
dev_id = 0
acl.rt.set_device(dev_id)

# 加载模型
model_id, model_desc = acl.mdl.load_from_file(args.model)

# 读取输入
input_data = np.fromfile(args.input, dtype=np.float32).reshape(1, 3, 640, 640)
input_buffer = acl.util.bytes_to_ptr(input_data.tobytes())
input_size = input_data.nbytes
# 0: FLOAT32, 0: NCHW
input_desc = acl.create_tensor_desc(0, [1, 3, 640, 640], 0)
input_data_buffer = acl.create_data_buffer(input_buffer, input_size)
input_dataset = acl.mdl.create_dataset([input_data_buffer])

# 获取输出buffer大小
output_size = acl.mdl.get_output_size_by_index(model_desc, 0)
output_buffer = acl.rt.malloc(output_size, acl.rt.MemType.DEVICE)
output_data_buffer = acl.create_data_buffer(output_buffer, output_size)
output_dataset = acl.mdl.create_dataset([output_data_buffer])

# 推理
acl.mdl.execute(model_id, input_dataset, output_dataset)

# 结果拷贝到host
output_host = np.empty(output_size // 4, dtype=np.float32)
acl.rt.memcpy(output_host, output_size, output_buffer, output_size, acl.rt.MemcpyKind.DEVICE_TO_HOST)
np.savetxt(args.output, output_host)

# 释放资源
acl.destroy_data_buffer(input_data_buffer)
acl.destroy_tensor_desc(input_desc)
acl.mdl.destroy_dataset(input_dataset)
acl.mdl.destroy_dataset(output_dataset)
acl.rt.free(output_buffer)
acl.mdl.unload(model_id)
acl.rt.reset_device(dev_id)
acl.finalize()