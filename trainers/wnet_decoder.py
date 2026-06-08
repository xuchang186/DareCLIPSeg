\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
   

from __future__ import annotations

from typing import Iterable, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.models.layers import DropPath
except Exception:                    
    class DropPath(nn.Identity):
        def __init__(self, drop_prob: float = 0.0):
            super().__init__()


BNNorm2d = nn.BatchNorm2d
Activation = nn.GELU


class UpConv(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            BNNorm2d(ch_out),
            Activation(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class DownConv(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1, bias=False),
            BNNorm2d(ch_out),
            Activation(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class ResBlock(nn.Module):
                                                                    

    def __init__(self, inplanes: int, planes: int, groups: int = 1):
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes

        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=1, padding=1, bias=False)
        self.bn1 = BNNorm2d(planes)
        self.act = Activation()
        self.conv2 = nn.Conv2d(
            planes, planes, 3, stride=1, groups=groups, padding=1, bias=False
        )
        self.bn2 = BNNorm2d(planes)

        if self.inplanes != self.planes:
            self.down = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride=1, bias=False),
                BNNorm2d(planes),
            )
        else:
            self.down = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.down(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.act(out)
        return out


class OPE(nn.Module):
                                                                                 

    def __init__(self, inplanes: int, planes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, 3, stride=1, padding=1, bias=False)
        self.bn1 = BNNorm2d(inplanes)
        self.act = Activation()
        self.down = DownConv(inplanes, planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.down(out)
        return out


class LocalBlock(nn.Module):
                                                                                    

    def __init__(
        self,
        inplanes: int,
        hidden_planes: int,
        planes: int,
        groups: int = 1,
        down_or_up: str | None = None,
    ):
        super().__init__()
        if down_or_up is None:
            self.block = nn.Sequential(
                ResBlock(inplanes=inplanes, planes=hidden_planes, groups=groups),
            )
        elif down_or_up == "down":
            self.block = nn.Sequential(
                ResBlock(inplanes=inplanes, planes=hidden_planes, groups=groups),
                DownConv(hidden_planes, planes),
            )
        elif down_or_up == "up":
            self.block = nn.Sequential(
                ResBlock(inplanes=inplanes, planes=hidden_planes, groups=groups),
                UpConv(hidden_planes, planes),
            )
        else:
            raise ValueError(f"Unsupported down_or_up={down_or_up!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Pooling(nn.Module):
                                                                              

    def __init__(self, pool_size: int = 3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size // 2, count_include_pad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x) - x


class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels: int, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        drop: float = 0.0,
    ):
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = Activation()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalBlock(nn.Module):
\
\
\
\
\
\
       

    def __init__(
        self,
        in_dim: int,
        dim: int,
        pool_size: int = 3,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, dim, kernel_size=3, padding=1, bias=False)
        self.norm1 = GroupNorm(dim)
        self.attn = Pooling(pool_size=pool_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = GroupNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            drop=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
            BNNorm2d(out_ch),
            Activation(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _as_hw(size: Union[int, Sequence[int], torch.Size]) -> Tuple[int, int]:
    if isinstance(size, int):
        return size, size
    if len(size) == 2:
        return int(size[0]), int(size[1])
    raise ValueError(f"Expected output size as int or length-2 sequence, got {size}")


class TextGuidedWNetDecoder(nn.Module):
\
\
\
\
\
\
\
\
\
\
       

    def __init__(
        self,
        image_channels: int = 3,
        clip_dim: int = 512,
        text_dim: int = 512,
        layer_channels: Sequence[int] = (32, 64, 128, 256, 320),
        global_dims: Sequence[int] = (16, 32, 64, 128, 160),
        fusion_dim: int = 64,
        gate_init: float = -4.0,
        normalize_dot: bool = True,
    ):
        super().__init__()
        if len(layer_channels) != 5 or len(global_dims) != 5:
            raise ValueError("layer_channels and global_dims must both have 5 values")

        c0, c1, c2, c3, c4 = map(int, layer_channels)
        g0, g1, g2, g3, g4 = map(int, global_dims)

        self.clip_dim = clip_dim
        self.text_dim = text_dim
        self.fusion_dim = fusion_dim
        self.normalize_dot = normalize_dot

                                        
        self.input_l0 = nn.Sequential(
            nn.Conv2d(image_channels, c0, kernel_size=3, stride=1, padding=1, bias=False),
            BNNorm2d(c0),
            Activation(),
            nn.Conv2d(c0, c0, kernel_size=3, stride=1, padding=1, bias=False),
            BNNorm2d(c0),
            Activation(),
        )

        self.encoder1_l1_local = OPE(c0, c1)
        self.encoder1_l1_global = GlobalBlock(c0, g0)
        self.encoder1_l2_local = OPE(c1, c2)
        self.encoder1_l2_global = GlobalBlock(c1, g1)
        self.encoder1_l3_local = OPE(c2, c3)
        self.encoder1_l3_global = GlobalBlock(c2, g2)
        self.encoder1_l4_local = OPE(c3, c4)
        self.encoder1_l4_global = GlobalBlock(c3, g3)

                                                                          
        self.clip_proj = ConvNormAct(clip_dim, c4, kernel_size=1)
        self.clip_fuse_e1 = ConvNormAct(c4 + c4, c4, kernel_size=1)
        self.clip_fuse_e2 = ConvNormAct(c4 + c4, c4, kernel_size=1)

        self.decoder1_l4_local = LocalBlock(c4, c4, c3, down_or_up="up")
        self.decoder1_l4_global = GlobalBlock(c4, g4)
        self.decoder1_l3_local = LocalBlock(c3 + g3, c3, c2, down_or_up="up")
        self.decoder1_l3_global = GlobalBlock(c3 + g3, g3)
        self.decoder1_l2_local = LocalBlock(c2 + g2, c2, c1, down_or_up="up")
        self.decoder1_l2_global = GlobalBlock(c2 + g2, g2)
        self.decoder1_l1_local = LocalBlock(c1 + g1, c1, c0, down_or_up="up")
        self.decoder1_l1_global = GlobalBlock(c1 + g1, g1)

                                        
        self.encoder2_l1_local = LocalBlock(c0 + g0, c0, c1, down_or_up="down")
        self.encoder2_l1_global = GlobalBlock(c0 + g0, g0)
        self.encoder2_l2_local = LocalBlock(c1 + g1, c1, c2, down_or_up="down")
        self.encoder2_l2_global = GlobalBlock(c1 + g1, g1)
        self.encoder2_l3_local = LocalBlock(c2 + g2, c2, c3, down_or_up="down")
        self.encoder2_l3_global = GlobalBlock(c2 + g2, g2)
        self.encoder2_l4_local = LocalBlock(c3 + g3, c3, c4, down_or_up="down")
        self.encoder2_l4_global = GlobalBlock(c3 + g3, g3)

        self.decoder2_l4_local = LocalBlock(c4 + g4, c4, c3, down_or_up="up")
        self.decoder2_l3_local = LocalBlock(c3 + g3, c3, c2, down_or_up="up")
        self.decoder2_l2_local = LocalBlock(c2 + g2, c2, c1, down_or_up="up")
        self.decoder2_l1_local = LocalBlock(c1 + g1, c1, c0, down_or_up="up")
        self.output_l0_feature = LocalBlock(c0 + g0, c0, c0, down_or_up=None)

                                           
        self.feature_proj = nn.Sequential(
            ConvNormAct(c0, fusion_dim, kernel_size=3),
            nn.Conv2d(fusion_dim, fusion_dim, kernel_size=1, bias=False),
            GroupNorm(fusion_dim),
        )
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, fusion_dim),
        )
        self.spatial_gate = nn.Sequential(
            ConvNormAct(fusion_dim, fusion_dim, kernel_size=3),
            nn.Conv2d(fusion_dim, 1, kernel_size=1),
        )

                                                                              
        self.residual_gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

        self.apply(self._init_weights)
        self._init_residual_gate()

    def _init_residual_gate(self):
        final_gate = self.spatial_gate[-1]
        if isinstance(final_gate, nn.Conv2d):
            nn.init.zeros_(final_gate.weight)
            if final_gate.bias is not None:
                nn.init.zeros_(final_gate.bias)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def residual_weight(self) -> torch.Tensor:
        return torch.sigmoid(self.residual_gate)

    def _clip_tokens_to_map(
        self,
        clip_tokens: torch.Tensor,
        image_hw: Tuple[int, int],
        patch_size: int,
    ) -> torch.Tensor:
        b, n, c = clip_tokens.shape
        h, w = image_hw
        hp, wp = h // patch_size, w // patch_size
        if hp * wp != n:
                                                                                 
                                                                  
            side = int(n ** 0.5)
            if side * side != n:
                raise ValueError(
                    f"Cannot reshape clip tokens with n={n} into a 2D grid. "
                    f"Expected H/patch_size*W/patch_size={hp * wp}."
                )
            hp = wp = side
        return clip_tokens.reshape(b, hp, wp, c).permute(0, 3, 1, 2).contiguous()

    def forward(
        self,
        image: torch.Tensor,
        clip_tokens: torch.Tensor,
        text_features: torch.Tensor,
        patch_size: int,
        output_size: Union[int, Sequence[int], torch.Size],
    ) -> torch.Tensor:
        out_hw = _as_hw(output_size)
        h, w = int(image.shape[-2]), int(image.shape[-1])

                                                                            
        clip_map = self._clip_tokens_to_map(clip_tokens, (h, w), patch_size)
        clip_map = self.clip_proj(clip_map)

                            
        x_e1_l0 = self.input_l0(image)

        x_e1_l1_local = self.encoder1_l1_local(x_e1_l0)
        x_e1_l0_global = self.encoder1_l1_global(x_e1_l0)

        x_e1_l2_local = self.encoder1_l2_local(x_e1_l1_local)
        x_e1_l1_global = self.encoder1_l2_global(x_e1_l1_local)

        x_e1_l3_local = self.encoder1_l3_local(x_e1_l2_local)
        x_e1_l2_global = self.encoder1_l3_global(x_e1_l2_local)

        x_e1_l4_local = self.encoder1_l4_local(x_e1_l3_local)
        x_e1_l3_global = self.encoder1_l4_global(x_e1_l3_local)

        if clip_map.shape[-2:] != x_e1_l4_local.shape[-2:]:
            clip_map = F.interpolate(
                clip_map,
                size=x_e1_l4_local.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        x_e1_l4_sem = self.clip_fuse_e1(torch.cat([x_e1_l4_local, clip_map], dim=1))

        x_d1_l3_local = self.decoder1_l4_local(x_e1_l4_sem)
        x_d1_l4_global = self.decoder1_l4_global(x_e1_l4_sem)

        x_d1_l3 = torch.cat((x_d1_l3_local, x_e1_l3_global), dim=1)
        x_d1_l2_local = self.decoder1_l3_local(x_d1_l3)
        x_d1_l3_global = self.decoder1_l3_global(x_d1_l3)

        x_d1_l2 = torch.cat((x_d1_l2_local, x_e1_l2_global), dim=1)
        x_d1_l1_local = self.decoder1_l2_local(x_d1_l2)
        x_d1_l2_global = self.decoder1_l2_global(x_d1_l2)

        x_d1_l1 = torch.cat((x_d1_l1_local, x_e1_l1_global), dim=1)
        x_d1_l0_local = self.decoder1_l1_local(x_d1_l1)
        x_d1_l1_global = self.decoder1_l1_global(x_d1_l1)

                            
        x_e2_l0 = torch.cat((x_d1_l0_local, x_e1_l0_global), dim=1)
        x_e2_l1_local = self.encoder2_l1_local(x_e2_l0)
        x_e2_l0_global = self.encoder2_l1_global(x_e2_l0)

        x_e2_l1 = torch.cat((x_e2_l1_local, x_d1_l1_global), dim=1)
        x_e2_l2_local = self.encoder2_l2_local(x_e2_l1)
        x_e2_l1_global = self.encoder2_l2_global(x_e2_l1)

        x_e2_l2 = torch.cat((x_e2_l2_local, x_d1_l2_global), dim=1)
        x_e2_l3_local = self.encoder2_l3_local(x_e2_l2)
        x_e2_l2_global = self.encoder2_l3_global(x_e2_l2)

        x_e2_l3 = torch.cat((x_e2_l3_local, x_d1_l3_global), dim=1)
        x_e2_l4_local = self.encoder2_l4_local(x_e2_l3)
        x_e2_l3_global = self.encoder2_l4_global(x_e2_l3)

        x_e2_l4_sem = self.clip_fuse_e2(torch.cat([x_e2_l4_local, clip_map], dim=1))
        x_e2_l4 = torch.cat((x_e2_l4_sem, x_d1_l4_global), dim=1)
        x_d2_l3_local = self.decoder2_l4_local(x_e2_l4)

        x_d2_l3 = torch.cat((x_d2_l3_local, x_e2_l3_global), dim=1)
        x_d2_l2_local = self.decoder2_l3_local(x_d2_l3)

        x_d2_l2 = torch.cat((x_d2_l2_local, x_e2_l2_global), dim=1)
        x_d2_l1_local = self.decoder2_l2_local(x_d2_l2)

        x_d2_l1 = torch.cat((x_d2_l1_local, x_e2_l1_global), dim=1)
        x_d2_l0_local = self.decoder2_l1_local(x_d2_l1)

        x_d2_l0 = torch.cat((x_d2_l0_local, x_e2_l0_global), dim=1)
        feat = self.output_l0_feature(x_d2_l0)
        feat = self.feature_proj(feat)

        text_kernel = self.text_proj(text_features)

        if self.normalize_dot:
            feat = F.normalize(feat, dim=1)
            text_kernel = F.normalize(text_kernel, dim=-1)

        logits = torch.einsum("bc,bchw->bhw", text_kernel, feat)
        logits = self.logit_scale.exp().clamp(max=10.0) * logits
        logits = logits * torch.sigmoid(self.spatial_gate(feat)).squeeze(1)

        if logits.shape[-2:] != out_hw:
            logits = F.interpolate(
                logits.unsqueeze(1), size=out_hw, mode="bilinear", align_corners=False
            ).squeeze(1)

        return logits
