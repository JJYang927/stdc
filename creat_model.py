from SE_block import SEBasicBlock, SEBottleneck
from SK_block import SKBasicBlock, SKBottleneck
from SW_block import SWBasicBlock, SWBottleneck
from SW3_block import SW3BasicBlock, SW3Bottleneck
from MR_block import MRBasicBlock, MRBottleneck
from MC_block import MCBasicBlock, MCBottleneck
from MS_block import MSBasicBlock, MSBottleneck
from resnet import ResNet, BasicBlock_old, Bottleneck_old
import torch
from thop import profile, clever_format



def Create_Model(block="SE", model="res18", num_classes=1000, deploy=False):

    if block == "SE":
        BasicBlock = SEBasicBlock
        Bottleneck = SEBottleneck
    elif block == "SK":
        BasicBlock = SKBasicBlock
        Bottleneck = SKBottleneck
    elif block == "SW":
        BasicBlock = SWBasicBlock
        Bottleneck = SWBottleneck
    elif block == "SW3":
        BasicBlock = SW3BasicBlock
        Bottleneck = SW3Bottleneck
    elif block == "MR":
        BasicBlock = MRBasicBlock
        Bottleneck = MRBottleneck
    elif block == "MC":
        BasicBlock = MCBasicBlock
        Bottleneck = MCBottleneck
    elif block == "MS":
        BasicBlock = MSBasicBlock
        Bottleneck = MSBottleneck
    else:
        print("----------Using the original network----------")
        BasicBlock = BasicBlock_old
        Bottleneck = Bottleneck_old


    if model == "res10":
        model = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes)
    elif model == "res18":
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    elif model == "res34":
        model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    elif model == "res50":
        model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    elif model == "res101":
        model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)
    elif model == "res152":
        model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)
    elif model == "res200":
        model = ResNet(Bottleneck, [3, 24, 36, 3], num_classes=num_classes)

    return model


model = "res18"
block = "MR"
num_classes = 10

net = Create_Model(block="{}".format(block), model="{}".format(model), num_classes=num_classes, deploy=False)

input = torch.randn(1, 3, 224, 224)
macs, params = profile(net, inputs=(input, ))
macs, params = clever_format([macs, params], "%.2f")
print(macs, params)
# .print(net)

model = "res18"
block = "MS"
num_classes = 10

net = Create_Model(block="{}".format(block), model="{}".format(model), num_classes=num_classes, deploy=False)

input = torch.randn(1, 3, 224, 224)
macs, params = profile(net, inputs=(input, ))
macs, params = clever_format([macs, params], "%.2f")
print(macs, params)
# print(net)

