from typing import ForwardRef, OrderedDict
from PIL.Image import init
from torch._C import device
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Any, List, Tuple
from torch.nn.modules import conv
from torch.nn.modules.pooling import AvgPool2d


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = nn.Conv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)   # 保证输出大小等于输入大小
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)   # 保证输出大小等于输入大小
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


# class Sequential(nn.Module):
#     def __init__(self, in_channels, out_channels, bn_size) -> None:
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.bn_size = bn_size
#         self.conv = nn.Sequential(
#             #kernel = 1
#             nn.BatchNorm2d(self.in_channels),
#             nn.ReLU(),
#             nn.Conv2d(self.in_channels, self.bn_size*self.out_channels, kernel_size = 1, stride = 1),
#             #kernel = 3
#             nn.BatchNorm2d(self.bn_size*self.out_channels),
#             nn.ReLU(),
#             nn.Conv2d(self.bn_size*self.out_channels,self.out_channels,kernel_size=3,stride=1,padding=1)
#         )
#     def forward(self, x):
#         x = self.conv(x)
#         return x


# class Den(nn.Module):
#     def __init__(self, in_channels, out_channels, bn_size) -> None:
#         super().__init__()
#         self.in_channels = self.inc = in_channels
#         self.out_channels = self.outc = out_channels
#         self.bn_size = self.bn = bn_size
#         # self.Sequential = Sequential(self.in_channels, self.out_channels, self.bn_size)
    
#     def Conv(self,x:Tensor)->Tensor:
#         self.conv = nn.Sequential(
#             #kernel = 1
#             nn.BatchNorm2d(self.in_channels),
#             nn.ReLU(),
#             nn.Conv2d(self.in_channels, self.bn_size*self.out_channels, kernel_size = 1, stride = 1),
#             #kernel = 3
#             nn.BatchNorm2d(self.bn_size*self.out_channels),
#             nn.ReLU(),
#             nn.Conv2d(self.bn_size*self.out_channels,self.out_channels,kernel_size=3,stride=1,padding=1)
#         )
#         return self.conv(x)
    
#     def forward(self, init_x:Tensor)->Tensor:
#         x_cat = [init_x]
#         x = init_x
#         device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         for i in range(6):
#             if self.in_channels == self.inc + self.out_channels * 6:
#                 self.in_channels = self.inc
#             print(self.in_channels)
#             self.Conv.to(device)
#             new_x = self.Conv(x)
#             x_cat.append(new_x)
#             x = torch.cat(x_cat,1)
#             x = x.to(device)
#             print(device)
#             print(x.size())
#             self.in_channels = self.in_channels + self.out_channels
#         return x


# class Transition(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(),
#             nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1),
#             nn.AvgPool2d(kernel_size=2,stride=2)
#         )
#     def forward(self,x):
#         x = self.conv(x)
#         return x

class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_features))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))

class _DenseLayer(nn.Module):
    def __init__(
        self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float, memory_efficient: bool = False
    ) -> None:
        super().__init__()
        self.norm1: nn.BatchNorm2d
        self.add_module("norm1", nn.BatchNorm2d(num_input_features))
        self.relu1: nn.ReLU
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.conv1: nn.Conv2d
        self.add_module(
            "conv1", nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        )
        self.norm2: nn.BatchNorm2d
        self.add_module("norm2", nn.BatchNorm2d(bn_size * growth_rate))
        self.relu2: nn.ReLU
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.conv2: nn.Conv2d
        self.add_module(
            "conv2", nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output
    
    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        memory_efficient: bool = False,
    ) -> None:
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)

class Den(nn.Module):
    def __init__(
        self,
        growth_config: Tuple[int, int, int] = (10, 16, 32),
        num_layers: int = 6,
        num_init_features: int = 4,
        bn_size: int = 4,
        drop_rate: float = 0,
        num_classes: int = 9,
        memory_efficient: bool = False,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(OrderedDict([]))
        num_features = num_init_features
        for i, growth_rate in enumerate(growth_config):
            layer = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.features.add_module("denseblock%d" %(i+1), layer)
            num_features = num_features + num_layers * growth_rate
            if i != len(growth_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2
    

    def forward(self,x:Tensor) -> Tensor:
        x = self.features(x)
        return x


class G_D_net(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 1, kernel_size=1, stride=1)
        self.ReLU = nn.ReLU(inplace=True)
        self.Den = Den((10,16,32), 6, 4, 4)
        # self.Den1 = Den(4,10,4)
        # self.Tran1 = Transition(64,32)
        # self.Den2 = Den(32, 16, 4)
        # self.Tran2 = Transition(128, 64)
        # self.Den3 = Den(64, 32, 4)
  
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.Inception1 = Inception(256, 128, 128, 192, 32, 96, 96)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.Inception2 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception3 = Inception(512, 256, 160, 320, 32, 128, 128)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.Inception4 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.4)

        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x_1 = [x]
        # 224 224 3
        x = self.conv1(x)
        x = self.ReLU(x)
        # 224 224 1
        x_1.append(x)
        x = torch.cat(x_1,1)
        # 224 224 4
        # x = self.Den1(x)
        # # 224 224 64
        # x = self.Tran1(x)
        # # 112 112 32
        # x = self.Den2(x)
        # # 112 112 128
        # x = self.Tran2(x)
        # # 56 56 64
        # x = self.Den3(x)
        x = self.Den(x)
        # 56 56 256
        x = self.maxpool1(x)
        # 28 28 256
        x = self.Inception1(x)
        # 28 28 512
        x = self.maxpool2(x)
        # 14 14 512
        x = self.Inception2(x)
        # 14 14 512
        x = self.inception3(x)
        # 14 14 832
        x = self.maxpool3(x)
        # 7 7 832
        x = self.Inception4(x)
        # 7 7 1024
        x = self.avgpool(x)
        # 1 1 1024
        x = torch.flatten(x, 1)
        # 1024
        x = self.dropout(x)
        x = self.fc(x)
        # num_classes

        return x




