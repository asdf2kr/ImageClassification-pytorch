import torch
import torch.nn as nn

'''
        conv_i
        i:1 { x = self.conv2(x) # (56, 56, 64) -> (56, 56, 64) -> -(56, 56, 64) -> (56, 56, 256) }
        i:2 { x = self.conv2(x) # (56, 56, 256) -> (56, 56, 64) -> -(56, 56, 64) -> (56, 56, 256) }

        x = self.conv3(x) # (56, 56, 256) ->  (56, 56, 128) -> (28, 28, 128) ->(28, 28, 512)
'''
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

def conv1x1(in_channels, out_channels, stride=1):
    ''' 1x1 convolution '''
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_channels, out_channels, stride=1):
    ''' 3x3 convolution '''
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, hid_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, hid_channels, stride)
        self.bn1 = nn.BatchNorm2d(hid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(hid_channels, hid_channels)
        self.bn2 = nn.BatchNorm2d(hid_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BottleneckBlock(nn.Module): # bottelneck-block, over the 50 layers.
    expansion = 4
    def __init__(self, in_channels, hid_channels, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        self.downsample = downsample
        out_channels = hid_channels * self.expansion
        self.conv1 = conv1x1(in_channels, hid_channels)
        self.bn1 = nn.BatchNorm2d(hid_channels)

        self.conv2 = conv3x3(hid_channels, hid_channels, stride)
        self.bn2 = nn.BatchNorm2d(hid_channels)

        self.conv3 = conv1x1(hid_channels, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x # indentity
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    '''
    *Input
        (224, 224, 3)
        The stride applies only  is
        stride는 입력 영상의 크기가 큰 경우,
        연산량을 줄이기 위한 목적으로 입력단과 가까운 쪽에만 적용한다.
        (Input_size - conv_size) / stride + 1 = output_size
    '''
    '''
    *50-layer
        conv1 (output: 112x112)
            7x7, 64, stride 2
        conv2 (output: 56x56)
            3x3 max pool, stride 2
            [ 1x1, 64  ]
            [ 3x3, 64  ] x 3
            [ 1x1, 256 ]
        cov3 (output: 28x28)
            [ 1x1, 128 ]
            [ 3x3, 128 ] x 4
            [ 1x1, 512 ]
        cov4 (output: 14x14)
            [ 1x1, 256 ]
            [ 3x3, 256 ] x 6
            [ 1x1, 1024]
        cov5 (output: 28x28)
            [ 1x1, 512 ]
            [ 3x3, 512 ] x 3
            [ 1x1, 2048]
        _ (output: 1x1)
            average pool, 100-d fc, softmax
        FLOPs 3.8x10^9
    '''
    '''
    *101-layer
        conv1 (output: 112x112)
            7x7, 64, stride 2
        conv2 (output: 56x56)
            3x3 max pool, stride 2
            [ 1x1, 64  ]
            [ 3x3, 64  ] x 3
            [ 1x1, 256 ]
        cov3 (output: 28x28)
            [ 1x1, 128 ]
            [ 3x3, 128 ] x 4
            [ 1x1, 512 ]
        cov4 (output: 14x14)
            [ 1x1, 256 ]
            [ 3x3, 256 ] x 23
            [ 1x1, 1024]
        cov5 (output: 28x28)
            [ 1x1, 512 ]
            [ 3x3, 512 ] x 3
            [ 1x1, 2048]
        _ (output: 1x1)
            average pool, 100-d fc, softmax
        FLOPs 7.6x10^9
    '''
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.layers = layers
        self.in_channels = 64

        if num_classes == 1000:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
        self.conv2 = self.get_layers(block, 64, self.layers[0])
        self.conv3 = self.get_layers(block, 128, self.layers[1], stride=2)
        self.conv4 = self.get_layers(block, 256, self.layers[2], stride=2)
        self.conv5 = self.get_layers(block, 512, self.layers[3], stride=2)
        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        torch.nn.init.kaiming_normal(self.fc.weight)
        for m in self.state_dict():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_layers(self, block, hid_channels, n_layers, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != hid_channels * block.expansion:
            downsample = nn.Sequential(
                    conv1x1(self.in_channels, hid_channels * block.expansion, stride),
                    nn.BatchNorm2d(hid_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, hid_channels, stride, downsample))
        self.in_channels = hid_channels * block.expansion

        for _ in range(1, n_layers):
            layers.append(block(self.in_channels, hid_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        '''
            Example tensor shape based on resnet101
        '''
        # (224 + 6(padding*2) - 7 ) / 2 + 1 = 112
        x = self.conv1(x) # (224, 224, 3) -> (112, 112, 64), kernel(7,7)
        '''
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxPool(x) # (112, 112, 64) -> (56, 56, 64), kernel(3,3)
        '''
        x = self.conv2(x) # { (56, 56, 64) -> (56, 56, 64) -> -(56, 56, 64) -> (56, 56, 256) } -> ... -> {(56, 56, 256) -> ... -> (56, 56, 256)}
        x = self.conv3(x) # { (56, 56, 256) ->  (56, 56, 128) -> (28, 28, 128) ->(28, 28, 512) } -> ...
        x = self.conv4(x) # { (28, 28, 512) ->  (28, 28, 256) -> (14, 14, 256) ->(14, 14, 1024) } -> ...
        x = self.conv5(x) # { (14, 14, 1024) ->  (14, 14, 512) -> (7, 7, 512) ->(7, 7, 2048) } -> ...

        x = self.avgPool(x) # (7, 7, 2048) -> (1, 1, 2048), kernel(7, 7)
        x = x.view(x.size(0), -1) # (batch, )
        x = self.fc(x) # (1, 1, 2048) -> (1, 1, 1000)
        return x

def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def resnet50(**kwargs):
    return ResNet(BottleneckBlock, [3, 4, 6, 3], **kwargs)

def resnet101(**kwargs):
    ''' ResNet-101 Model'''
    return ResNet(BottleneckBlock, [3, 4, 23, 3], **kwargs)

def resnet152(**kwargs):
    return ResNet(BottleneckBlock, [3, 8, 36, 3], **kwargs)
