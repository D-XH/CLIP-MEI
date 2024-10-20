import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Callable, Optional
from torchvision.models.resnet import ResNet50_Weights, ResNet, _ovewrite_named_param, Bottleneck

def resnet50(weights=None, progress: bool = True, **kwargs: Any):
    weights = ResNet50_Weights.verify(weights)
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
    model = ResNet(newBottleneck, [3, 4, 6, 3], **kwargs)
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

###########################################################################################################

class blk(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        in_ch = in_ch // 2
        self.lte = lte(in_ch)
        self.gte = gte(in_ch)
        pass
    def forward(self, x):
        # (160, 3, 224, 224)
        n, c, h, w = x.shape
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
    mm = resnet50(weights=models.ResNet50_Weights.DEFAULT)
    mm = nn.Sequential(*list(mm.children())[:-1])
    i = torch.rand(8,3,224,224)
    o = mm(i)
    print(mm)
    print(o.shape)