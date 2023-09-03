# from ptflops import get_model_complexity_info
from thop import profile, clever_format
import torch
from creat_model import Create_Model

model = "res18"
block = "MS"
dataset = "cifar10"
num_classes = 10

model = Create_Model(block="{}".format(block), model="{}".format(model), num_classes=num_classes, deploy=False)

input = torch.randn(1, 3, 224, 224)
macs, params = profile(model, inputs=(input, ))
macs, params = clever_format([macs, params], "%.2f")
print(macs, params)

print("end")
