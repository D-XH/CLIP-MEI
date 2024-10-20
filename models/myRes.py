import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Callable, List, Optional, Type, Union
from torchvision.models.resnet import ResNet50_Weights, ResNet, _ovewrite_named_param, Bottleneck, BasicBlock, conv1x1
from torchvision.utils import _log_api_usage_once

def resnet50_1(weights=None, progress: bool = True, **kwargs: Any):
    weights = ResNet50_Weights.verify(weights)
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
    model = ResNet(newBottleneck, [3, 4, 6, 3], **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress), strict=False)
    return model

def resnet50_2(weights=None, progress: bool = True, **kwargs: Any):
    weights = ResNet50_Weights.verify(weights)
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
    model = newResNet_2(Bottleneck, [3, 4, 6, 3], **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress), strict=False)
    return model
###########################################################################################################

class newBottleneck(Bottleneck):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__(inplanes, planes, stride, downsample, groups, base_width, dilation, norm_layer)
        width = int(planes * (base_width / 64.0)) * groups
        self.blk = blk(width)
        self.bn4 = nn.BatchNorm2d(width)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)     # 1x1
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)   # 3x3
        out = self.bn2(out)
        out = self.relu(out)

        out = self.blk(out)
        out = self.bn4(out)
        out = self.relu(out)

        out = self.conv3(out)   # 1x1
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class newResnet(ResNet):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__(block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer)
        self.blk_1 = blk(512)
        self.blk_2 = blk(1024)
        del self.avgpool
        del self.fc
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        # print('ssss',x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        
        # print(x.shape)
        x = self.blk_1(x)
        x = self.layer3(x)

        # print(x.shape)
        x = self.blk_2(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        return x
    
class newResNet_2(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.blk_1 = blk(512)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.blk_2 = blk(1024)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.blk_1(x)
        x = self.layer3(x)
        x = self.blk_2(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
###########################################################################################################

class blk(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        in_ch = in_ch // 2
        self.lte = lte(in_ch)
        self.gte = gte(in_ch)
        pass
    def forward(self, x):
        # (160, 512, 224, 224)
        n, c, h, w = x.shape
        # print(x.shape)
        f1 = self.gte(x[:,:c//2])
        f2 = self.lte(x[:, c//2:])
        x = torch.concat([f1, f2], dim=1)
        return x

class ce(nn.Module):
    def __init__(self,):
        pass
    def forward(self, x):
        # (160, 64/g, 224, 224)
        n, c, h, w = x.shape

        pass

class gte(nn.Module):
    def __init__(self, in_ch, seq_len=8):
        super().__init__()
        self.seq_len = seq_len

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv3d(in_ch, in_ch*2, kernel_size=(3,1,1), stride=1, padding=(1,0,0), bias=False)
        self.conv2 = nn.Conv3d(in_ch*2, in_ch, kernel_size=(3,1,1), stride=1, padding=(1,0,0), bias=False)
        self.bn1 = nn.BatchNorm3d(in_ch*2)
        self.bn2 = nn.BatchNorm3d(in_ch)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)

    def forward(self, x):
        # (160, 64/g, 224, 224)
        n, c, h, w = x.shape
        # print(x.shape)
        x = x.reshape(-1, self.seq_len, c, h, w).transpose(1, 2)    # (20, 64/g, 8, 224, 224)
        res = x

        a, b = x[:, :-1], x[:, 1:]
        diff = b - a    # (20, 64/g, 7, 224, 224)
        diff = F.pad(diff.transpose(1, 2), (0,0,0,0,0,1), 'constant', 0)    # (20, 64/g, 8, 224, 224)
        diff = self.avgpool(diff.transpose(1, 2))   # (20, 64/g, 8, 1, 1)
        
        x = self.conv1(diff)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.sigmoid(x) * res

        return x.transpose(1, 2).reshape(n, c, h, w)

class lte(nn.Module):
    def __init__(self, in_ch, seq_len=8):
        super().__init__()
        self.seq_len = seq_len

        self.conv1 = nn.Conv3d(in_ch, in_ch*2, kernel_size=(3,1,1), stride=1, padding=(1,0,0), bias=False)
        self.conv2 = nn.Conv3d(in_ch*2, in_ch, kernel_size=(3,1,1), stride=1, padding=(2,0,0), dilation=(2,1,1), bias=False)
        self.bn1 = nn.BatchNorm3d(in_ch*2)
        self.bn2 = nn.BatchNorm3d(in_ch)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)

    def forward(self, x):
        # (160, 64/g, 224, 224)
        n, c, h, w = x.shape
        # print(x.shape)
        x = x.reshape(-1, self.seq_len, c, h, w).transpose(1, 2)    # (20, 64/g, 8, 224, 224)
        res = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.sigmoid(x) * res 

        return x.transpose(1, 2).reshape(n, c, h, w)

class se(nn.Module):
    def __init__(self,):
        pass
    def forward(self, x):
        # (160, 64/g, 224, 224)
        n, c, h, w = x.shape

        pass

if __name__ == '__main__':
    import torchvision.models as models
    mm = resnet50_2(weights=models.ResNet50_Weights.DEFAULT)
    mm = nn.Sequential(*list(mm.children())[:-1])
    # print(mm)
    i = torch.rand(8,3,224,224)
    o = mm(i)
    print(o.shape)