import math
from typing import Optional

import torch.nn as nn
import torchvision.models as models

import vision_transformer as vits
from bow import bow_utils as utils


class SequentialFeatureExtractorAbstractClass(nn.Module):
    def __init__(self, all_feat_names, feature_blocks):
        super(SequentialFeatureExtractorAbstractClass, self).__init__()

        assert (isinstance(feature_blocks, list))
        assert (isinstance(all_feat_names, list))
        assert (len(all_feat_names) == len(feature_blocks))

        self.all_feat_names = all_feat_names
        self._feature_blocks = nn.ModuleList(feature_blocks)

    def _parse_out_keys_arg(self, out_feat_keys):
        # By default return the features of the last layer / module.
        out_feat_keys = (
            [self.all_feat_names[-1], ] if out_feat_keys is None else
            out_feat_keys)

        if len(out_feat_keys) == 0:
            raise ValueError('Empty list of output feature keys.')

        for f, key in enumerate(out_feat_keys):
            if key not in self.all_feat_names:
                raise ValueError(
                    'Feature with name {0} does not exist. '
                    'Existing features: {1}.'.format(key, self.all_feat_names))
            elif key in out_feat_keys[:f]:
                raise ValueError(
                    'Duplicate output feature key: {0}.'.format(key))

        # Find the highest output feature in `out_feat_keys
        max_out_feat = max(
            [self.all_feat_names.index(key) for key in out_feat_keys])

        return out_feat_keys, max_out_feat

    def get_subnetwork(self, out_feat_key):
        if isinstance(out_feat_key, str):
            out_feat_key = [out_feat_key, ]
        _, max_out_feat = self._parse_out_keys_arg(out_feat_key)
        subnetwork = nn.Sequential()
        for f in range(max_out_feat + 1):
            subnetwork.add_module(
                self.all_feat_names[f],
                self._feature_blocks[f]
            )
        return subnetwork

    def forward(self, x, out_feat_keys=None):
        """Forward the image `x` through the network and output the asked features.
        Args:
          x: input image.
          out_feat_keys: a list/tuple with the feature names of the features
                that the function should return. If out_feat_keys is None (
                default value) then the last feature of the network is returned.

        Return:
            out_feats: If multiple output features were asked then `out_feats`
                is a list with the asked output features placed in the same
                order as in `out_feat_keys`. If a single output feature was
                asked then `out_feats` is that output feature (and not a list).
        """
        out_feat_keys, max_out_feat = self._parse_out_keys_arg(out_feat_keys)
        out_feats = [None] * len(out_feat_keys)

        feat = x
        for f in range(max_out_feat + 1):
            feat = self._feature_blocks[f](feat)
            key = self.all_feat_names[f]
            if key in out_feat_keys:
                out_feats[out_feat_keys.index(key)] = feat

        out_feats = (out_feats[0] if len(out_feats) == 1 else out_feats)

        return out_feats


class BasicBlock(nn.Module):
    def __init__(
            self,
            in_planes,
            out_planes,
            stride,
            drop_rate=0.0,
            kernel_size=3):
        super(BasicBlock, self).__init__()

        if not isinstance(kernel_size, (list, tuple)):
            kernel_size = [kernel_size, kernel_size]
        assert isinstance(kernel_size, (list, tuple))
        assert len(kernel_size) == 2

        kernel_size1, kernel_size2 = kernel_size

        assert kernel_size1 == 1 or kernel_size1 == 3
        padding1 = 1 if kernel_size1 == 3 else 0
        assert kernel_size2 == 1 or kernel_size2 == 3
        padding2 = 1 if kernel_size2 == 3 else 0

        self.equalInOut = (in_planes == out_planes and stride == 1)

        self.convResidual = nn.Sequential()

        if self.equalInOut:
            self.convResidual.add_module('bn1', nn.BatchNorm2d(in_planes))
            self.convResidual.add_module('relu1', nn.ReLU(inplace=True))

        self.convResidual.add_module(
            'conv1',
            nn.Conv2d(
                in_planes, out_planes, kernel_size=kernel_size1,
                stride=stride, padding=padding1, bias=False))

        self.convResidual.add_module('bn2', nn.BatchNorm2d(out_planes))
        self.convResidual.add_module('relu2', nn.ReLU(inplace=True))
        self.convResidual.add_module(
            'conv2',
            nn.Conv2d(
                out_planes, out_planes, kernel_size=kernel_size2,
                stride=1, padding=padding2, bias=False))

        if drop_rate > 0:
            self.convResidual.add_module('dropout', nn.Dropout(p=drop_rate))

        if self.equalInOut:
            self.convShortcut = nn.Sequential()
        else:
            self.convShortcut = nn.Conv2d(
                in_planes, out_planes, kernel_size=1, stride=stride,
                padding=0, bias=False)

    def forward(self, x):
        return self.convShortcut(x) + self.convResidual(x)


class NetworkBlock(nn.Module):
    def __init__(
            self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0):
        super(NetworkBlock, self).__init__()

        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, drop_rate)

    def _make_layer(
            self, block, in_planes, out_planes, nb_layers, stride, drop_rate):
        layers = []
        for i in range(nb_layers):
            in_planes_arg = i == 0 and in_planes or out_planes
            stride_arg = i == 0 and stride or 1
            layers.append(
                block(in_planes_arg, out_planes, stride_arg, drop_rate))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(SequentialFeatureExtractorAbstractClass):
    def __init__(
            self,
            depth,
            widen_factor=1,
            drop_rate=0.0,
            strides=[2, 2, 2],
            global_pooling=True):

        assert (depth - 4) % 6 == 0
        num_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        num_layers = [int((depth - 4) / 6) for _ in range(3)]

        block = BasicBlock

        all_feat_names = []
        feature_blocks = []

        # 1st conv before any network block
        conv1 = nn.Sequential()
        conv1.add_module(
            'Conv',
            nn.Conv2d(3, num_channels[0], kernel_size=3, padding=1, bias=False))
        conv1.add_module('BN', nn.BatchNorm2d(num_channels[0]))
        conv1.add_module('ReLU', nn.ReLU(inplace=True))
        feature_blocks.append(conv1)
        all_feat_names.append('conv1')

        # 1st block.
        block1 = nn.Sequential()
        block1.add_module(
            'Block',
            NetworkBlock(
                num_layers[0], num_channels[0], num_channels[1], BasicBlock,
                strides[0], drop_rate))
        block1.add_module('BN', nn.BatchNorm2d(num_channels[1]))
        block1.add_module('ReLU', nn.ReLU(inplace=True))
        feature_blocks.append(block1)
        all_feat_names.append('block1')

        # 2nd block.
        block2 = nn.Sequential()
        block2.add_module(
            'Block',
            NetworkBlock(
                num_layers[1], num_channels[1], num_channels[2], BasicBlock,
                strides[1], drop_rate))
        block2.add_module('BN', nn.BatchNorm2d(num_channels[2]))
        block2.add_module('ReLU', nn.ReLU(inplace=True))
        feature_blocks.append(block2)
        all_feat_names.append('block2')

        # 3rd block.
        block3 = nn.Sequential()
        block3.add_module(
            'Block',
            NetworkBlock(
                num_layers[2], num_channels[2], num_channels[3], BasicBlock,
                strides[2], drop_rate))
        block3.add_module('BN', nn.BatchNorm2d(num_channels[3]))
        block3.add_module('ReLU', nn.ReLU(inplace=True))
        feature_blocks.append(block3)
        all_feat_names.append('block3')

        # global average pooling.
        if global_pooling:
            feature_blocks.append(utils.GlobalPooling(type="avg"))
            all_feat_names.append('GlobalPooling')

        super(WideResNet, self).__init__(all_feat_names, feature_blocks)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ResNet(SequentialFeatureExtractorAbstractClass):
    def __init__(self, arch, pretrained=False, global_pooling=True, red: Optional[int] = None):
        net = models.__dict__[arch](num_classes=1000, pretrained=pretrained)
        print(f'==> Pretrained parameters: {pretrained}')
        all_feat_names = []
        feature_blocks = []

        # 1st conv before any resnet block
        conv1 = nn.Sequential()
        conv1.add_module('Conv', net.conv1)
        conv1.add_module('bn', net.bn1)
        conv1.add_module('relu', net.relu)
        conv1.add_module('maxpool', net.maxpool)
        feature_blocks.append(conv1)
        all_feat_names.append('conv1')

        # 1st block.
        feature_blocks.append(net.layer1)
        all_feat_names.append('block1')
        # 2nd block.
        feature_blocks.append(net.layer2)
        all_feat_names.append('block2')
        # 3rd block.
        feature_blocks.append(net.layer3)
        all_feat_names.append('block3')
        # 4th block.
        feature_blocks.append(net.layer4)
        all_feat_names.append('block4')

        # global average pooling.
        if global_pooling:
            feature_blocks.append(utils.GlobalPooling(type="avg"))
            all_feat_names.append('GlobalPooling')

        if red is not None:
            feature_blocks.append(nn.Flatten())
            all_feat_names.append("final_flatten")
            feature_blocks.append(nn.Linear(512 * net.layer4[-1].expansion, int(512 * net.layer4[-1].expansion / red)))
            all_feat_names.append("reduction")

        super(ResNet, self).__init__(all_feat_names, feature_blocks)
        self.num_channels = net.fc.in_features


class MultiPrototypes(nn.Module):
    def __init__(self, output_dim, nmb_prototypes):
        super().__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            self.add_module("prototypes" + str(i), nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, "prototypes" + str(i))(x))
        return out


def conv3x3(in_channels, out_channels, maxpool=True, ada_maxpool=False, **kwargs):
    tmp = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        # nn.MaxPool2d(2) if not ada_maxpool else nn.AdaptiveMaxPool2d(5)
    )
    if maxpool and not ada_maxpool:
        tmp.add_module("maxpool", nn.MaxPool2d(2))
    elif maxpool and ada_maxpool:
        tmp.add_module("ada_maxpool", nn.AdaptiveMaxPool2d(5))
    elif not maxpool and not ada_maxpool:
        pass
    return tmp


class CNN_4Layer(SequentialFeatureExtractorAbstractClass):
    def __init__(self, in_channels: int, out_channels=64, hidden_size=64, global_pooling=True,
                 graph_conv=False, final_maxpool=True, ada_maxpool=False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        all_feat_names = []
        feature_blocks = []

        feature_blocks.append(conv3x3(in_channels, hidden_size))
        all_feat_names.append('block1')

        feature_blocks.append(conv3x3(hidden_size, hidden_size))
        all_feat_names.append('block2')

        feature_blocks.append(conv3x3(hidden_size, hidden_size))
        all_feat_names.append('block3')

        feature_blocks.append(
            conv3x3(hidden_size, out_channels, maxpool=final_maxpool, ada_maxpool=ada_maxpool))
        all_feat_names.append('block4')

        if global_pooling:
            feature_blocks.append(utils.GlobalPooling(type='avg'))
            all_feat_names.append('GlobalPooling')
        if graph_conv:
            feature_blocks.append(nn.Flatten())
            all_feat_names.append('FinalFlatten')
            # feature_blocks.append(gnn.DynamicEdgeConv(nn.Linear(1600 * 2, 1600), k=5))
            # all_feat_names.append('EdgeConv')
        super(CNN_4Layer, self).__init__(all_feat_names, feature_blocks)
        self.num_channels = out_channels


class ViT(nn.Module):
    def __init__(self, arch: str, patch_size: int, num_channels: int):
        super(ViT, self).__init__()
        self.arch = arch
        self.patch_size = patch_size
        self.num_channels = num_channels
        if arch in vits.__dict__.keys():
            self.encoder = vits.__dict__[arch](patch_size=patch_size)

    def forward(self, x):
        return self.encoder(x)


def FeatureExtractor(arch, opts):
    all_architectures = (
        'wrn', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        'resnext101_32x8d', 'resnext50_32x4d', 'wide_resnet101_2',
        'wide_resnet50_2', 'conv4')

    assert arch in all_architectures
    if arch == 'conv4':
        conv4 = CNN_4Layer(in_channels=3)
        return conv4, conv4.num_channels
    elif arch in vits.__dict__.keys():
        vit = vits.__dict__[arch](**opts)
        return vit
    elif arch == 'wrn':
        num_channels = opts["widen_factor"] * 64
        return WideResNet(**opts), num_channels
    else:
        resnet_extractor = ResNet(arch=arch, **opts)
        return resnet_extractor, resnet_extractor.num_channels


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, relu=True, pool=True):
        super().__init__()
        self.layers = nn.Sequential()

        self.layers.add_module(
            "Conv",
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.layers.add_module("BatchNorm", nn.BatchNorm2d(out_planes))

        if relu:
            self.layers.add_module("ReLU", nn.ReLU(inplace=True))
        if pool:
            self.layers.add_module("MaxPool", nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

    def forward(self, x):
        out = self.layers(x)
        return out


class ConvNet(SequentialFeatureExtractorAbstractClass):
    def __init__(self, opt):
        self.in_planes = opt["in_planes"]
        self.out_planes = opt["out_planes"]
        self.num_stages = opt["num_stages"]
        self.average_end = opt["average_end"] if ("average_end" in opt) else False
        if type(self.out_planes) == int:
            self.out_planes = [self.out_planes for i in range(self.num_stages)]
        assert type(self.out_planes) == list and len(self.out_planes) == self.num_stages

        num_planes = [self.in_planes, ] + self.out_planes
        userelu = opt["userelu"] if ("userelu" in opt) else True

        self.use_pool = opt["use_pool"] if ("use_pool" in opt) else None
        if self.use_pool is None:
            self.use_pool = [True for i in range(self.num_stages)]
        assert len(self.use_pool) == self.num_stages

        feature_blocks = []
        for i in range(self.num_stages):
            feature_blocks.append(
                ConvBlock(
                    num_planes[i],
                    num_planes[i + 1],
                    relu=(userelu if i == (self.num_stages - 1) else True),
                    pool=self.use_pool[i],
                )
            )

        all_feat_names = ["conv" + str(s + 1) for s in range(self.num_stages)]

        if self.average_end:
            feature_blocks.append(utils.GlobalPooling(type="avg"))
            all_feat_names.append("GlobalAvgPooling")

        super().__init__(all_feat_names, feature_blocks)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 1, bias=False)


def norm_layer(planes):
    return nn.BatchNorm2d(planes)


def conv3x3_only(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 3, padding=1, bias=False)


class Block(nn.Module):

    def __init__(self, inplanes, planes, downsample):
        super().__init__()

        self.relu = nn.LeakyReLU(0.1)

        self.conv1 = conv3x3_only(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3_only(planes, planes)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv3x3_only(planes, planes)
        self.bn3 = norm_layer(planes)

        self.downsample = downsample

        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        out = self.maxpool(out)

        return out


class ResNet12(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.inplanes = 3

        self.layer1 = self._make_layer(channels[0])
        self.layer2 = self._make_layer(channels[1])
        self.layer3 = self._make_layer(channels[2])
        self.layer4 = self._make_layer(channels[3])

        self.out_dim = channels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes):
        downsample = nn.Sequential(
            conv1x1(self.inplanes, planes),
            norm_layer(planes),
        )
        block = Block(self.inplanes, planes, downsample)
        self.inplanes = planes
        return block

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.shape[0], x.shape[1], -1).mean(dim=2)
        return x


def resnet12():
    return ResNet12([64, 128, 256, 512])


def resnet12_wide():
    return ResNet12([64, 160, 320, 640])


def create_model(opt):
    return ConvNet(opt)
