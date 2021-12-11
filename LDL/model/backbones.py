import torchvision
from torch import nn
import timm

class ResNet(nn.Module):

    def __init__(self, pretrined=True):
        super().__init__()
        backbone = torchvision.models.resnet50(pretrained=pretrined)
        self.num_features = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractaor = nn.Sequential(*layers)

    def forward(self, x):
        return self.feature_extractaor(x)


def get_resnet(pretrained=True):
    return ResNet(pretrained)


def get_swin_tiny(pretrained=True):
    backbone = timm.create_model("swin_tiny_patch4_window7_224", pretrained=pretrained)
    backbone.reset_classifier(0)
    return backbone


def get_backbone(backbone, pretrined=True):
    back_d = {
        'resnet50': get_resnet,
        'swin_tiny': get_swin_tiny
    }

    return back_d[backbone](pretrined)