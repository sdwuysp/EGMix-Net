
import torch
import torch.nn.functional as F

def lf_loss(pred, target, scales=(8, 16, 32), use_global=True):
    terms = []
    H, W = pred.shape[-2:]
    for s in scales:
        if H >= s and W >= s:
            terms.append(F.l1_loss(F.avg_pool2d(pred, s, s), F.avg_pool2d(target, s, s)))
    if use_global:
        terms.append(F.l1_loss(pred.mean(dim=(2, 3)), target.mean(dim=(2, 3))))
    return sum(terms) / max(len(terms), 1)

def block_transmission(hazy, target, s, tau=0.02):

    with torch.no_grad():
        h, g = hazy.mean(1, keepdim=True), target.mean(1, keepdim=True)
        sd = lambda x: (F.avg_pool2d(x * x, s, s) - F.avg_pool2d(x, s, s) ** 2).clamp_min(0).sqrt()
        sh, sg = sd(h), sd(g)
        t = (sh / sg.clamp_min(1e-6)).clamp(0, 1)
        return torch.where(sg < tau, torch.ones_like(t), t)

def lf_loss_tw(pred, target, hazy, scales=(8, 16, 32), use_global=True, tau=0.02, global_w='t'):

    terms = []
    H, W = pred.shape[-2:]
    tg = None
    for s in scales:
        if H >= s and W >= s:
            t = block_transmission(hazy, target, s, tau)
            terms.append((t * (F.avg_pool2d(pred, s, s) - F.avg_pool2d(target, s, s)).abs()).mean())
            tg = t.mean(dim=(1, 2, 3)) if tg is None else tg
    if use_global:
        w = tg if (tg is not None and global_w == 't') else torch.ones(pred.shape[0], device=pred.device)
        terms.append((w[:, None] * (pred.mean(dim=(2, 3)) - target.mean(dim=(2, 3))).abs()).mean())
    return sum(terms) / max(len(terms), 1)
