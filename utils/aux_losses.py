
import torch
import torch.nn.functional as F

_KERNELS = {}

def _gauss(ch, size, sigma, device, dtype):
    key = (ch, size, sigma, device, dtype)
    if key not in _KERNELS:
        t = torch.arange(size, dtype=torch.float64) - (size - 1) / 2
        g = torch.exp(-t ** 2 / (2 * sigma ** 2)); g = g / g.sum()
        k = (g[:, None] * g[None, :]).to(device=device, dtype=dtype)
        _KERNELS[key] = k.expand(ch, 1, size, size).contiguous()
    return _KERNELS[key]

def dped_color_loss(pred, target, size=21, sigma=3.0):
    pred, target = pred.float(), target.float()
    C = pred.shape[1]
    k = _gauss(C, size, sigma, pred.device, pred.dtype)
    pad = size // 2
    bp = F.conv2d(F.pad(pred, (pad,) * 4, mode='reflect'), k, groups=C)
    bt = F.conv2d(F.pad(target, (pad,) * 4, mode='reflect'), k, groups=C)
    return F.mse_loss(bp, bt)

def multiscale_content_loss(pred, target, scales=(2, 4)):
    pred, target = pred.float(), target.float()
    terms = []
    for s in scales:
        p = F.interpolate(pred, scale_factor=1.0 / s, mode='bilinear', align_corners=False)
        t = F.interpolate(target, scale_factor=1.0 / s, mode='bilinear', align_corners=False)
        terms.append(F.l1_loss(p, t))
    return sum(terms) / len(terms)
