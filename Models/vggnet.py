import torch.nn
import torch.nn as nn


__all__ = ['VGGNet', 'vgg16']
class VGGNet(nn.Module):
    def __init__(self, model, num_classes=1000):
        super(VGGNet, self).__init__()
        self.model = model
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
    def forward(self, x):
        x = self.model(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
def get_layers(layers):
    model_layers = []
    in_channels = 3
    for l in layers:
        if l == 'M':
            model_layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            model_layers += [nn.Conv2d(in_channels, l, kernel_size=3, padding=1)]
            model_layers += [nn.BatchNorm2d(l)]
            model_layers += [nn.ReLU(inplace=True)]
            in_channels = l
    return nn.Sequential(*model_layers)

def vggnet16(**kwargs):
    layers = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    return VGGNet(get_layers(layers), **kwargs)
