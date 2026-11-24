
import os, sys, types, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F

def _fold_bn_after(conv, bn):
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    w = conv.weight * s.reshape(-1, 1, 1, 1)
    b = (conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)) * s + bn.bias - bn.running_mean * s
    c = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                  conv.dilation, conv.groups, bias=True).to(w.device)
    c.weight.data.copy_(w); c.bias.data.copy_(b)
    return c

def _fold_bn_before_1x1(bn, conv):
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps); t = bn.bias - bn.running_mean * s
    W = conv.weight[:, :, 0, 0]
    c = nn.Conv2d(conv.in_channels, conv.out_channels, 1, bias=True).to(W.device)
    c.weight.data.copy_((W * s[None, :])[:, :, None, None])
    c.bias.data.copy_((conv.bias if conv.bias is not None else 0) + W @ t)
    return c

@torch.no_grad()
def reparam_block(b, mode='tri'):
    assert b.large_mode == 'parallel' and b.gate_mode == 'energy' and b.use_medium
    assert not b.use_global_gate and not b.use_lowres_ctx
    C = b.norm1.num_features; dev = b.proj.weight.device
    W = torch.zeros(3 * C, 1, 5, 5, device=dev); B = torch.zeros(3 * C, device=dev)
    W[0::3] = b.medium_branch.weight; B[0::3] = b.medium_branch.bias
    W[1::3, :, 1:4, 1:4] = b.small_kernel.weight; B[1::3] = b.small_kernel.bias
    hp = -torch.full((3, 3), 1 / 9, device=dev); hp[1, 1] += 1
    W[2::3, 0, 1:4, 1:4] = hp
    b.rp_mode = mode
    if mode == 'tri':
        b.rp_tri = nn.Conv2d(C, 3 * C, 5, padding=2, groups=C, bias=True).to(dev)
        b.rp_tri.weight.copy_(W); b.rp_tri.bias.copy_(B)
    else:
        b.rp_med, b.rp_small = b.medium_branch, b.small_kernel
        b.rp_hp = nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False).to(dev)
        b.rp_hp.weight.copy_(hp.expand(C, 1, 3, 3))
    a1, bn, a2 = b.axial_branch
    b.rp_ax1 = _fold_bn_after(a1, bn); b.rp_ax2 = a2
    b.rp_gconv = b.modulator.noise_filter[0]
    b.rp_s = b.modulator.scale; b.rp_b = b.modulator.bias
    b.rp_ffn1 = _fold_bn_before_1x1(b.norm2, b.ffn[0]); b.rp_ffn2 = b.ffn[2]

    def fwd(self, x):
        xh = self.norm1(x)
        if self.rp_mode == 'tri':
            n, c, h, w = xh.shape
            t = self.rp_tri(xh).view(n, c, 3, h, w)
            med, fs, hpf = t[:, :, 0], t[:, :, 1], t[:, :, 2]
        else:
            med, fs, hpf = self.rp_med(xh), self.rp_small(xh), self.rp_hp(xh)
        fl = self.rp_ax2(self.rp_ax1(xh)) + med
        e = F.gelu(self.rp_gconv(hpf))
        m = torch.sigmoid(torch.addcmul(self.rp_b, self.rp_s, e * e))
        if self.store_mask:
            self.last_mask = m.detach()
        x = x + self.proj(torch.lerp(fl, fs, m))
        return x + self.rp_ffn2(F.gelu(self.rp_ffn1(x)))
    b.forward = types.MethodType(fwd, b)
    keep_g = b.rp_gconv
    for k in ('axial_branch', 'modulator', 'norm2', 'ffn') + (('medium_branch', 'small_kernel') if mode == 'tri' else ()):
        delattr(b, k)
    b.rp_gconv = keep_g
    return b

def reparam_model(net, mode='tri'):
    from models.MixDehazeNet import MixStructureBlock
    net.eval()
    for m in list(net.modules()):
        if isinstance(m, MixStructureBlock):
            reparam_block(m, mode)
    return net

if __name__ == '__main__':
    import copy, numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--model', default='MixDehazeNet-s'); ap.add_argument('--variant', default='D')
    ap.add_argument('--data_dir', default='~/datasets'); ap.add_argument('--dataset', default='RESIDE-OUT')
    ap.add_argument('--check', action='store_true'); ap.add_argument('--res', default=1024, type=int)
    ap.add_argument('--no_tf32', action='store_true', help='Disable TF32 to verify if implementation differences stem from convolution numerical precision')
    args = ap.parse_args()
    from models import build_model
    from datasets.loader import PairLoader
    torch.backends.cudnn.benchmark = True
    if args.no_tf32:
        torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    try:
        ck = torch.load(os.path.expanduser(args.ckpt), map_location='cpu', weights_only=False)
    except TypeError:
        ck = torch.load(os.path.expanduser(args.ckpt), map_location='cpu')
    sd = ck['state_dict'] if 'state_dict' in ck else ck
    ref = build_model(args.model, args.variant)
    ref.load_state_dict({(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()})
    ref = ref.cuda().eval(); fast = reparam_model(copy.deepcopy(ref), 'tri'); sep = reparam_model(copy.deepcopy(ref), 'sep')
    with torch.no_grad():
        x = torch.randn(1, 3, 256, 256, device='cuda')
        r0 = ref(x); print(f'Random input: Output amplitude {r0.abs().max().item():.1f}，tri max error {(r0 - fast(x)).abs().max().item():.2e}，sep max error {(r0 - sep(x)).abs().max().item():.2e}(unclamped, [-1,1] scale)')
        ds = PairLoader(os.path.join(os.path.expanduser(args.data_dir), args.dataset), 'test', 'test')
        pr, pf, dmax = [], [], 0.0
        for i in range(len(ds)):
            it = ds[i]; x = torch.as_tensor(it['source'])[None].cuda(); y = torch.as_tensor(it['target'])[None].cuda() * 0.5 + 0.5
            a = ref(x).clamp(-1, 1) * 0.5 + 0.5; b = fast(x).clamp(-1, 1) * 0.5 + 0.5
            dmax = max(dmax, (a - b).abs().max().item() * 255)
            pr.append(10 * np.log10(1 / F.mse_loss(a, y).item())); pf.append(10 * np.log10(1 / F.mse_loss(b, y).item()))
        print(f'{args.dataset} {len(ds)} images: Original impl PSNR {np.mean(pr):.4f} | Equivalent impl {np.mean(pf):.4f} | Max output diff {dmax:.3f} gray levels')

        def tm(m):
            z = torch.randn(1, 3, args.res, args.res, device='cuda')
            for _ in range(20):
                m(z)
            torch.cuda.synchronize(); s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record()
            for _ in range(100):
                m(z)
            e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / 100
        print(f'{args.res}x{args.res} Timing: Original impl {tm(ref):.1f} ms | tri {tm(fast):.1f} ms | sep {tm(sep):.1f} ms | Parameters Orig {sum(p.numel() for p in ref.parameters())/1e6:.4f} M，tri {sum(p.numel() for p in fast.parameters())/1e6:.4f} M')
