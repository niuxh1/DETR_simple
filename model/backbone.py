import math
from collections import OrderedDict
from dataclasses import replace

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models._utils import IntermediateLayerGetter

from typing import Optional, List, Dict


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):

        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


class position_by_sin(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize

        if scale is True and normalize is False:
            raise ValueError('normalize should be True if scale is True')

        self.scale = scale if scale is not None else 2 * math.pi

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask

        assert mask is not None

        not_mask = ~mask

        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos


class back_bones_base(nn.Module):
    def __init__(self, back_bone, train: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in back_bone.named_parameters():
            if not train or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad(False)

        if return_interm_layers:
            return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
        else:
            return_layers = {'layer4': '0'}

        self.body = IntermediateLayerGetter(back_bone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}

        for name, x in xs.items():
            mask = tensor_list.mask
            assert mask is not None
            mask_ = F.interpolate(mask[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask_)

        return out


class back_bones(back_bones_base):
    def __init__(self,
                 name:str,
                 train:bool,
                 return_interm_layers: bool,
                 dilation: bool,):
        back_bone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=True,
            norm_layer=torch.nn.BatchNorm2d
        )
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(back_bone, train, num_channels=num_channels, return_interm_layers=return_interm_layers)


class joiner(nn.Sequential):
    def __init__(self, back_bone, position_embedding):
        super().__init__(back_bone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs= self[0](tensor_list)

        out: Dict[str, NestedTensor] = {}
        pos=[]

        for name, x in xs.items():
            out.append(x)
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_back_bone(name:str,train=False,return_interm_layers=False,dilation=False):
    position_embedding = position_by_sin()
    model = joiner(back_bones(name,train,return_interm_layers,dilation),
                   position_embedding)
    model.num_channels = model[0].num_channels
    return model


