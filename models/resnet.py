#modified from original source
#github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

import binarized_modules

#from .._internally_replaced_utils import load_state_dict_from_url
#from ..utils import _log_api_usage_once


__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


def conv3x3(in_planes: int, out_planes: int, layer_prec_config: dict, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    q_scheme=layer_prec_config["q_scheme"]
    bias=layer_prec_config["bias"]
    if q_scheme == "float":
        return nn.Conv2d(in_planes, 
            out_planes, kernel_size=3, stride=stride, 
            padding=dilation, groups=groups, bias=bias, 
            dilation=dilation,)
    elif q_scheme == "clamp_float":
        return binarized_modules.ClampFloatConv2d(in_planes, 
            out_planes, kernel_size=3, stride=stride, 
            padding=dilation, groups=groups, bias=bias, 
            dilation=dilation,)
    elif q_scheme == "bwn":
        weight_bw=layer_prec_config["weight_bw"]
        return binarized_modules.BWConv2d(weight_bw, in_planes, 
            out_planes, kernel_size=3, stride=stride, 
            padding=dilation, groups=groups, bias=bias, 
            dilation=dilation)
    elif q_scheme == "xnor":
        act_bw=layer_prec_config["act_bw"]
        weight_bw=layer_prec_config["weight_bw"]
        return binarized_modules.BinarizeConv2d(act_bw, act_bw, weight_bw, in_planes, 
            out_planes, kernel_size=3, stride=stride, 
            padding=dilation, groups=groups, bias=bias, 
            dilation=dilation)
    elif q_scheme == "fp":
        act_bw=layer_prec_config["act_bw"]
        weight_bw=layer_prec_config["weight_bw"]
        return binarized_modules.QuantizeConv2d(in_planes, 
            out_planes, kernel_size=3, stride=stride, 
            padding=dilation, groups=groups, bias=bias, 
            dilation=dilation, bitwidth=act_bw, weight_bitwidth=weight_bw)

def conv1x1(in_planes: int, out_planes: int, layer_prec_config: dict, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    q_scheme=layer_prec_config["q_scheme"]
    bias=layer_prec_config["bias"]
    if q_scheme == "float":
        return nn.Conv2d(in_planes, 
            out_planes, kernel_size=1, stride=stride,
            bias=bias)
    elif q_scheme == "clamp_float":
        return binarized_modules.ClampFloatConv2d(in_planes, 
            out_planes, kernel_size=1, stride=stride,
            bias=bias)
    elif q_scheme == "bwn":
        weight_bw=layer_prec_config["weight_bw"]
        return binarized_modules.BWConv2d(weight_bw, in_planes, 
            out_planes, kernel_size=1, stride=stride,
            bias=bias)
    elif q_scheme == "xnor":
        act_bw=layer_prec_config["act_bw"]
        weight_bw=layer_prec_config["weight_bw"]
        return binarized_modules.BinarizeConv2d(act_bw, act_bw, weight_bw, in_planes, 
            out_planes, kernel_size=1, stride=stride,
            bias=bias)
    elif q_scheme == "fp":
        act_bw=layer_prec_config["act_bw"]
        weight_bw=layer_prec_config["weight_bw"]
        return binarized_modules.QuantizeConv2d(in_planes, 
            out_planes, kernel_size=1, stride=stride,
            bias=bias, bitwidth=act_bw, weight_bitwidth=weight_bw)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        layer_prec_config: dict,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()

        self.q_scheme=layer_prec_config["q_scheme"] if "q_scheme" in layer_prec_config.keys() else None
        self.act_bw=layer_prec_config["act_bw"] if "act_bw" in layer_prec_config.keys() else None
        self.weight_bw=layer_prec_config["weight_bw"] if "weight_bw" in layer_prec_config.keys() else None
        self.activation_type=layer_prec_config["activation_type"] if "activation_type" in layer_prec_config.keys() else None
        self.leaky_relu_slope=layer_prec_config["leaky_relu_slope"] if "leaky_relu_slope" in layer_prec_config.keys() else None
        self.bias=layer_prec_config["bias"] if "bias" in layer_prec_config.keys() else None

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, layer_prec_config, stride=stride)
        self.bn1 = norm_layer(planes)
        #removed inplace due to error from torch 1.9
        self.act = binarized_modules.get_activation(self.activation_type, input_shape=(1, planes, 1, 1), leaky_relu_slope=self.leaky_relu_slope)
        self.conv2 = conv3x3(planes, planes, layer_prec_config)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.bn3 = norm_layer(planes)

    def forward(self, x: Tensor) -> Tensor:
        #identity = x
        identity = x.clone()
        #identity.retain_grad()
        
        out = self.conv1(x)
        out = self.bn1(out)

        out = self.act(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)
            identity = self.bn3(identity)

        out += identity
        out = self.act(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        layer_prec_config: dict,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        self.q_scheme=layer_prec_config["q_scheme"] if "q_scheme" in layer_prec_config.keys() else None
        self.act_bw=layer_prec_config["act_bw"] if "act_bw" in layer_prec_config.keys() else None
        self.weight_bw=layer_prec_config["weight_bw"] if "weight_bw" in layer_prec_config.keys() else None
        self.activation_type=layer_prec_config["activation_type"] if "activation_type" in layer_prec_config.keys() else None
        self.leaky_relu_slope=layer_prec_config["leaky_relu_slope"] if "leaky_relu_slope" in layer_prec_config.keys() else None
        self.bias=layer_prec_config["bias"] if "bias" in layer_prec_config.keys() else None
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, layer_prec_config)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, layer_prec_config, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, layer_prec_config)
        self.bn3 = norm_layer(planes * self.expansion)
        self.act = binarized_modules.get_activation(self.activation_type, input_shape=(1, planes, 1, 1), leaky_relu_slope=self.leaky_relu_slope)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act(out)

        return out


default_prec_config = {
        "conv1": {"dtype": "float"},
        "layer1": {"dtype": "xnor", "act_bw": 2, "weight_bw": 1},
        "layer2": {"dtype": "xnor", "act_bw": 2, "weight_bw": 1},
        "layer3": {"dtype": "xnor", "act_bw": 2, "weight_bw": 1},
        "layer4": {"dtype": "xnor", "act_bw": 2, "weight_bw": 1},
        "fc1": {"dtype": "float"}
    }
class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 128,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        input_channels=3, normalize_output=False,
        prec_config=default_prec_config
    ) -> None:
        super().__init__()
        #_log_api_usage_once(self)

        self.block = block
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.layers = layers
        self.num_classes = num_classes
        self.normalize_output=normalize_output
        self.input_channels = input_channels

        self.sqrt_eps = 1e-3

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.replace_stride_with_dilation = replace_stride_with_dilation

        self.groups = groups
        self.base_width = width_per_group

        self.update_prec_config(prec_config, save_weights=False)

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
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def update_prec_config(self, prec_config, save_weights=True):
        print(prec_config)
        if save_weights:
            binarized_modules.copy_org_to_data(self)
            state_dict = self.state_dict()
        
        self.prec_config = prec_config

        self.inplanes = 64
        self.dilation = 1
        
        l = list(self.prec_config.keys())
        l.sort()
        assert l == ["conv1", "fc", "layer1", "layer2", "layer3", "layer4"]

        conv1_config = self.prec_config["conv1"]
        if conv1_config["q_scheme"] == "float":
            self.conv1 = nn.Conv2d(self.input_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=conv1_config["bias"])
        elif conv1_config["q_scheme"] == "clamp_float":
            self.conv1 = binarized_modules.ClampFloatConv2d(self.input_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=conv1_config["bias"])
        elif conv1_config["q_scheme"] == "bwn":
            self.conv1 = binarized_modules.BWConv2d(conv1_config["weight_bw"], self.input_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=conv1_config["bias"])
        elif conv1_config["q_scheme"] == "xnor":
            self.conv1 = binarized_modules.BinarizeConv2d(conv1_config["act_bw"], conv1_config["act_bw"], conv1_config["weight_bw"], self.input_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=conv1_config["bias"])
        elif conv1_config["q_scheme"] == "fp":
            self.conv1 = binarized_modules.QuantizeConv2d(self.input_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=conv1_config["bias"], bitwidth=conv1_config["act_bw"], weight_bitwidth=conv1_config["weight_bw"])
        else:
            raise "Invalid conv1 quantization scheme {}".format(conv1_config["q_scheme"])
        
        self.bn1 = self._norm_layer(self.inplanes)
        self.act1 = binarized_modules.get_activation(conv1_config["activation_type"], input_shape=(1, self.inplanes, 1, 1), leaky_relu_slope=conv1_config["leaky_relu_slope"] if "leaky_relu_slope" in conv1_config.keys() else None)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(self.block, prec_config["layer1"], 64, self.layers[0])
        self.layer2 = self._make_layer(self.block, prec_config["layer2"], 128, self.layers[1], stride=2, dilate=self.replace_stride_with_dilation[0])
        self.bn_dim = 128
        if len(self.layers) >= 3:
            self.layer3 = self._make_layer(self.block, prec_config["layer3"], 256, self.layers[2], stride=2, dilate=self.replace_stride_with_dilation[1])
            self.bn_dim = 256
        else:
            self.layer3 = nn.Identity()
        if len(self.layers) >= 4:
            self.layer4 = self._make_layer(self.block, prec_config["layer4"], 512, self.layers[3], stride=2, dilate=self.replace_stride_with_dilation[2])
            self.bn_dim = 512 if self.block==BasicBlock else 2048
        else:
            self.layer4 = nn.Identity()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn2 = self._norm_layer(self.bn_dim)
        
        fc_config = self.prec_config["fc"]
        self.act2 = binarized_modules.get_activation(fc_config["activation_type"], input_shape=(1, self.bn_dim * self.block.expansion, 1, 1), leaky_relu_slope=fc_config["leaky_relu_slope"] if "leaky_relu_slope" in fc_config.keys() else None)
        if self.block == BasicBlock:
            if fc_config["q_scheme"] == "float":
                self.fc = nn.Linear(self.bn_dim * self.block.expansion, self.num_classes, bias=fc_config["bias"])
            elif fc_config["q_scheme"] == "clamp_float":
                self.fc = binarized_modules.ClampFloatLinear(self.bn_dim * self.block.expansion, self.num_classes, bias=fc_config["bias"])
            elif fc_config["q_scheme"] == "bwn":
                self.fc = binarized_modules.BWLinear(fc_config["weight_bw"], self.bn_dim * self.block.expansion, self.num_classes, bias=fc_config["bias"])
            elif fc_config["q_scheme"] == "xnor":
                self.fc = binarized_modules.BinarizeLinear(fc_config["act_bw"], fc_config["act_bw"], fc_config["weight_bw"], self.bn_dim * self.block.expansion, self.num_classes, bias=fc_config["bias"])
            elif fc_config["q_scheme"] == "fp":
                self.fc = binarized_modules.QuantizeLinear(self.bn_dim * self.block.expansion, self.num_classes, bias=fc_config["bias"], bitwidth=fc_config["act_bw"], weight_bitwidth=fc_config["weight_bw"])
            else:
                raise "Invalid fc quantization scheme {}".format(fc_config["q_scheme"])
        else:
            if fc_config["q_scheme"] == "float":
                self.fc = nn.Linear(self.bn_dim, self.num_classes, bias=fc_config["bias"])
            elif fc_config["q_scheme"] == "clamp_float":
                self.fc = binarized_modules.ClampFloatLinear(self.bn_dim, self.num_classes, bias=fc_config["bias"])
            elif fc_config["q_scheme"] == "bwn":
                self.fc = binarized_modules.BWLinear(fc_config["weight_bw"], self.bn_dim, self.num_classes, bias=fc_config["bias"])
            elif fc_config["q_scheme"] == "xnor":
                self.fc = binarized_modules.BinarizeLinear(fc_config["act_bw"], fc_config["act_bw"], fc_config["weight_bw"], self.bn_dim, self.num_classes, bias=fc_config["bias"])
            elif fc_config["q_scheme"] == "fp":
                self.fc = binarized_modules.QuantizeLinear(self.bn_dim, self.num_classes, bias=fc_config["bias"], bitwidth=fc_config["act_bw"], weight_bitwidth=fc_config["weight_bw"])
            else:
                raise "Invalid fc quantization scheme {}".format(fc_config["q_scheme"])

        self.bn3 = nn.BatchNorm1d(self.num_classes)

        if save_weights:
            self.load_state_dict(state_dict, strict=False)
            binarized_modules.copy_data_to_org(self)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layer_prec_config: dict,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, layer_prec_config, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, layer_prec_config, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    layer_prec_config,
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
        x = self.maxpool(x)
        x = self.bn1(x)
        
        x = self.act1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.bn3(x)

        #TODO: Add batchnorm
        if self.normalize_output:
          x_norm = torch.sqrt(torch.sum(torch.mul(x,x), dim=1) + self.sqrt_eps)  #torch.linalg.norm(x)
          x_norm = torch.unsqueeze(x_norm, 1)
          x = torch.div(x, x_norm)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    input_channels=3, normalize_output=False,
    **kwargs: Any,
) -> ResNet:
    model = ResNet(block, layers, input_channels=input_channels, normalize_output=normalize_output, **kwargs)
    #if pretrained:
        #state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        #model.load_state_dict(state_dict)
    """
    total = 0
    for p in model.parameters():
        count = 1
        for e in p.size():
            count *= e
        total += count
    print("Total parameters", total)
    exit(0)
    """
    return model


def resnetCustomLayers(pretrained: bool = False, progress: bool = True, input_channels=3, normalize_output=True, layers=[2, 2, 2, 2], blockType="BasicBlock", **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", BasicBlock if blockType=="BasicBlock" else Bottleneck, layers, pretrained, progress, input_channels=input_channels, normalize_output=normalize_output, **kwargs)

def resnet18(pretrained: bool = False, progress: bool = True, input_channels=3, normalize_output=True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, input_channels=input_channels, normalize_output=normalize_output, **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, input_channels=3, normalize_output=False, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, input_channels=input_channels, normalize_output=normalize_output, **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet152", Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet("resnext50_32x4d", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet("resnext101_32x8d", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet("wide_resnet50_2", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2

class Combined_Model(nn.Module):
    def __init__(self, face_model, speaker_model):
        super(Combined_Model, self).__init__()
        self.face_model = face_model
        self.speaker_model = speaker_model

        self.sqrt_eps = 1e-3

        self.fc1 = nn.Linear(1024, 512)

        """
        for param in self.face_model.parameters():
            param.requires_grad = False
        for param in self.speaker_model.parameters():
            param.requires_grad = False
        """
    
    def forward(self, face, spectrogram):
        """
        with torch.no_grad():
            face_embedding = self.face_model.forward(face)
            spectrogram_embedding = self.speaker_model.forward(spectrogram)
        """
        face_embedding = self.face_model.forward(face)
        spectrogram_embedding = self.speaker_model.forward(spectrogram)
        
        x = torch.cat((face_embedding, spectrogram_embedding), dim=1)
        x = self.fc1(x)
        
        x_norm = torch.sqrt(torch.sum(torch.mul(x,x), dim=1) + self.sqrt_eps)  #torch.linalg.norm(x)
        x_norm = torch.unsqueeze(x_norm, 1)
        x = torch.div(x, x_norm)

        return x