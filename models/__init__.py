from models.ResNet import *
from models.ResNets import *
from models.VGG import *

model_dict = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "wide_resnet50_2": wide_resnet50_2,
    "wide_resnet101_2": wide_resnet101_2,
    "resnet101": resnet101,
    "resnet50": resnet50,
    "resnet20s": resnet20s,
    "resnet44s": resnet44s,
    "resnet56s": resnet56s,
    "vgg16_bn": vgg16_bn,
}
