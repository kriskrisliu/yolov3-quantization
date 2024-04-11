import torch, onnx
import json

weight_json = {
    "conv_bitwidth": {
        "kernel": 10,
        "bias": 16
    },
    "bn_bitwidth": 8,
    "working_bitwidth": 32,
    "datapath_bitwidth": 16,
    "lut_bitwidth": 14,
    "leaky_relu_alpha": 16,
    "average_pool_radix": 16,
    "example_customized_config_conv2d_1": {
        "kernel": 10,
        "bias": 16
    },
    # "Conv_139": {
    #     "kernel": 6,
    #     "bias": 16
    # }
}

layer_4bit = ['model.9.conv', 'model.10.0.cv1.conv', 'model.10.0.cv2.conv', 'model.10.1.cv1.conv', 'model.10.1.cv2.conv', 'model.10.2.cv1.conv', 'model.10.2.cv2.conv', 'model.10.3.cv1.conv', 'model.10.3.cv2.conv', 'model.11.cv1.conv', 'model.11.cv2.conv', 'model.13.conv',  'model.15.conv', ]
layer_6bit = ['model.12.conv', 'model.14.conv','model.16.conv']

model_onnx = onnx.load('yolov3_946c88ee1f03a6bf94a6d32285868448.onnx')

ct4 = 0
ct6 = 0
for node in model_onnx.graph.node:
    name_list = node.input
    # import ipdb; ipdb.set_trace()
    if any(((n+".weight") in name_list) for n in layer_4bit):
        print(node.name, name_list)
        ct4 += 1
        weight_json[node.name] = {
            "kernel": 4,
            "bias": 16
        }
    elif any(((n+".weight") in name_list) for n in layer_6bit):
        print(node.name, name_list)
        ct6 += 1
        weight_json[node.name] = {
            "kernel": 6,
            "bias": 16
        }
print(ct4, ct6)
with open("w8a8-w6a8-w4a8-weight.json","") as fp:
    json.dump(weight_json,fp)
import ipdb;ipdb.set_trace()