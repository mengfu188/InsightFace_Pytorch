# 使用tracing转换

from torch import onnx

from model import *

# model = torchvision.models.resnet18()
model = MobileFaceNet(128)
model.eval()

example = torch.randn(1, 3, 112, 112)

print(model(example))

# Providing input and output names sets the display names for values
# within the model's graph. Setting these does not change the semantics
# of the graph; it is only for readability.
#
# The inputs to the network consist of the flat list of inputs (i.e.
# the values you would pass to the forward() method) followed by the
# flat list of parameters. You can partially specify names, i.e. provide
# a list here shorter than the number of inputs to the model, and we will
# only set that subset of names, starting from the beginning.
# 为网络的每层重新定义一个名字
# input_names = ["actual_input_1"] + ["learned_%d" % i for i in range(16)]
# output_names = ["output1"]

# 调用onnx包将pyTorch模型转换成onnx格式的模型，命名为alexnet.onnx
torch.onnx.export(model, example, "model.onnx", verbose=True)

# run it'

import caffe2.python.onnx.backend as backend
import numpy as np
import onnx

# Load the ONNX model
model = onnx.load("model.onnx")

rep = backend.prepare(model, device="CPU") # or "CPU"
outputs = rep.run(example.numpy())
print(outputs)
