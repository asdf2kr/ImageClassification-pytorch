import torch.nn
import torch.nn as nn
class VGGNet(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGGNet, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
