
import torch
import torch.nn as nn
import torch.nn.functional as F

VARIANTS = {

    'A': ('serial',   'conv',   True,  False, False),
    'B': ('parallel', 'conv',   True,  False, False),
    'C': ('serial',   'energy', True,  False, False),
    'D': ('parallel', 'energy', True,  False, False),
    'E': ('parallel', 'add',    True,  False, False),
    'F': ('parallel', 'energy', False, False, False),
    'G': ('parallel', 'energy', True,  True,  False),
    'H': ('parallel', 'energy', True,  True,  True),
}

class FreqModulator(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.noise_filter = nn.Sequential(
            nn.Conv2d(dim, dim, 5, padding=2, groups=dim),
            nn.GELU()
        )
        self.scale = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        hp = x - self.avg_pool(x)
        hp_clean = self.noise_filter(hp)
        energy = hp_clean ** 2
        return torch.sigmoid(self.scale * energy + self.bias)

class ConvFreqModulator(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.detector = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.detector(x - self.avg_pool(x))

def local_mean(x, kh, kw):

    H, W = x.shape[-2:]
    kh, kw = min(kh, H), min(kw, W)
    s = F.pad(x.float().cumsum(-2).cumsum(-1), (1, 0, 1, 0))
    out = (s[..., kh:, kw:] - s[..., :-kh, kw:] - s[..., kh:, :-kw] + s[..., :-kh, :-kw]) / (kh * kw)
    l, t = (kw - 1) // 2, (kh - 1) // 2
    out = F.pad(out, (l, kw - 1 - l, t, kh - 1 - t), mode='replicate')
    return out.to(x.dtype)

class GlobalHazeGate(nn.Module):

    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden = max(dim // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(dim, hidden, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, dim, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.tlc_kernel = None

    def forward(self, x):
        if self.tlc_kernel is not None and not self.training:
            ctx = local_mean(x, self.tlc_kernel, self.tlc_kernel)
        else:
            ctx = self.pool(x)
        return 2 * torch.sigmoid(self.fc2(self.act(self.fc1(ctx))))

class LowResContext(nn.Module):

    def __init__(self, dim, scale=8):
        super().__init__()
        self.scale = scale
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, padding_mode='reflect')
        self.pw = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.pw.weight)
        nn.init.zeros_(self.pw.bias)

    def forward(self, x):
        H, W = x.shape[2:]
        s = self.scale
        y = F.adaptive_avg_pool2d(x, (max(1, H // s), max(1, W // s)))
        y = self.pw(F.gelu(self.dw(y)))
        return F.interpolate(y, size=(H, W), mode='bilinear', align_corners=False)

class MixStructureBlock(nn.Module):
    def __init__(self, dim, large_mode='parallel', gate_mode='energy', use_medium=True,
                 global_gate=False, lowres_ctx=False):
        super().__init__()
        assert large_mode in ('serial', 'parallel')
        assert gate_mode in ('energy', 'conv', 'add')
        self.large_mode = large_mode
        self.gate_mode = gate_mode
        self.use_medium = use_medium
        self.use_global_gate = global_gate
        self.use_lowres_ctx = lowres_ctx
        if global_gate:
            self.global_gate = GlobalHazeGate(dim)
        if lowres_ctx:
            self.lowres_ctx = LowResContext(dim)

        self.norm1 = nn.BatchNorm2d(dim)
        k_size = 21
        pad = k_size // 2

        if large_mode == 'parallel':
            self.axial_branch = nn.Sequential(
                nn.Conv2d(dim, dim, (1, k_size), padding=(0, pad), groups=dim),
                nn.BatchNorm2d(dim),
                nn.Conv2d(dim, dim, (k_size, 1), padding=(pad, 0), groups=dim)
            )
            if use_medium:
                self.medium_branch = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        else:
            self.large_kernel = nn.Sequential(
                nn.Conv2d(dim, dim, (1, k_size), padding=(0, pad), groups=dim),
                nn.Conv2d(dim, dim, (k_size, 1), padding=(pad, 0), groups=dim),
                nn.Conv2d(dim, dim, 7, padding=9, groups=dim, dilation=3)
            )

        self.small_kernel = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

        if gate_mode == 'energy':
            self.modulator = FreqModulator(dim)
        elif gate_mode == 'conv':
            self.modulator = ConvFreqModulator(dim)

        self.proj = nn.Conv2d(dim, dim, 1)

        self.norm2 = nn.BatchNorm2d(dim)
        hidden_dim = int(dim * 4)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )

        self.store_mask = False
        self.last_mask = None

    def _large(self, x):
        if self.large_mode == 'parallel':
            out = self.axial_branch(x)
            if self.use_medium:
                out = out + self.medium_branch(x)
        else:
            out = self.large_kernel(x)
        if self.use_lowres_ctx:
            out = out + self.lowres_ctx(x)
        if self.use_global_gate:
            out = out * self.global_gate(x)
        return out

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)

        feat_large = self._large(x)
        feat_small = self.small_kernel(x)

        if self.gate_mode == 'add':
            x_fused = feat_large + feat_small
        else:
            mask = self.modulator(x)
            if self.store_mask:
                self.last_mask = mask.detach()
            x_fused = feat_large * (1 - mask) + feat_small * mask

        x = shortcut + self.proj(x_fused)

        shortcut = x
        x = shortcut + self.ffn(self.norm2(x))
        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, depth, **block_kw):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.blocks = nn.ModuleList([MixStructureBlock(dim=dim, **block_kw) for _ in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        if kernel_size is None:
            kernel_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        return self.proj(x)

class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        if kernel_size is None:
            kernel_size = 1
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        return self.proj(x)

class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()
        self.height = height
        d = max(int(dim / reduction), 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape
        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)
        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))
        return torch.sum(in_feats * attn, dim=1)

class MixDehazeNet(nn.Module):
    def __init__(self, in_chans=3, out_chans=4,
                 embed_dims=[24, 48, 96, 48, 24],
                 depths=[1, 1, 2, 1, 1],
                 variant='D'):
        super(MixDehazeNet, self).__init__()
        assert variant in VARIANTS, f'unknown variant {variant}, choose from {list(VARIANTS)}'
        large_mode, gate_mode, use_medium, global_gate, lowres_ctx = VARIANTS[variant]
        kw = dict(large_mode=large_mode, gate_mode=gate_mode, use_medium=use_medium,
                  global_gate=global_gate, lowres_ctx=lowres_ctx)
        self.variant = variant

        self.patch_size = 4
        self.patch_embed = PatchEmbed(patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)
        self.layer1 = BasicLayer(dim=embed_dims[0], depth=depths[0], **kw)
        self.patch_merge1 = PatchEmbed(patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1], kernel_size=3)
        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)
        self.layer2 = BasicLayer(dim=embed_dims[1], depth=depths[1], **kw)
        self.patch_merge2 = PatchEmbed(patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2], kernel_size=3)
        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)
        self.layer3 = BasicLayer(dim=embed_dims[2], depth=depths[2], **kw)
        self.patch_split1 = PatchUnEmbed(patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])
        assert embed_dims[1] == embed_dims[3]
        self.fusion1 = SKFusion(embed_dims[3])
        self.layer4 = BasicLayer(dim=embed_dims[3], depth=depths[3], **kw)
        self.patch_split2 = PatchUnEmbed(patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])
        assert embed_dims[0] == embed_dims[4]
        self.fusion2 = SKFusion(embed_dims[4])
        self.layer5 = BasicLayer(dim=embed_dims[4], depth=depths[4], **kw)
        self.patch_unembed = PatchUnEmbed(patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=3)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.layer1(x)
        skip1 = x
        x = self.patch_merge1(x)
        x = self.layer2(x)
        skip2 = x
        x = self.patch_merge2(x)
        x = self.layer3(x)
        x = self.patch_split1(x)
        x = self.fusion1([x, self.skip2(skip2)]) + x
        x = self.layer4(x)
        x = self.patch_split2(x)
        x = self.fusion2([x, self.skip1(skip1)]) + x
        x = self.layer5(x)
        x = self.patch_unembed(x)
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        feat = self.forward_features(x)
        K, B = torch.split(feat, (1, 3), dim=1)
        x = K * x - B + x
        return x[:, :, :H, :W]

def set_tlc(model, train_patch):

    scale = {'layer1': 1, 'layer2': 2, 'layer3': 4, 'layer4': 2, 'layer5': 1}
    n = 0
    for lname, s in scale.items():
        for m in getattr(model, lname).modules():
            if isinstance(m, GlobalHazeGate):
                m.tlc_kernel = (train_patch // s) if train_patch else None
                n += 1
    return n

def MixDehazeNet_t(variant='D'):
    return MixDehazeNet(embed_dims=[24, 48, 96, 48, 24], depths=[1, 1, 2, 1, 1], variant=variant)

def MixDehazeNet_s(variant='D'):
    return MixDehazeNet(embed_dims=[24, 48, 96, 48, 24], depths=[2, 2, 4, 2, 2], variant=variant)

def MixDehazeNet_b(variant='D'):
    return MixDehazeNet(embed_dims=[24, 48, 96, 48, 24], depths=[4, 4, 8, 4, 4], variant=variant)

def MixDehazeNet_l(variant='D'):
    return MixDehazeNet(embed_dims=[24, 48, 96, 48, 24], depths=[8, 8, 16, 8, 8], variant=variant)

def build_model(name, variant='D'):

    fn = {'MixDehazeNet-t': MixDehazeNet_t, 'MixDehazeNet-s': MixDehazeNet_s,
          'MixDehazeNet-b': MixDehazeNet_b, 'MixDehazeNet-l': MixDehazeNet_l}
    if name not in fn:
        raise ValueError(f'unknown model {name}, choose from {list(fn)}')
    return fn[name](variant=variant)
