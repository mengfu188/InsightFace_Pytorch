# 使用tracing转换

import torch
from model import *
from torch import jit
import torchvision

# model = torchvision.models.resnet18()
model = MobileFaceNet(128)
model.eval()

example = torch.randn(1, 3, 112, 112)
# example = torch.randn(1, 3, 224, 224)

traces_script_module = jit.trace(model, (example, ))

traces_script_module.save('model.pt')

r1 = model(example)
model = jit.load('model.pt')
r2 = model(example)

print(r1)
print(r2)
