
import os, sys, json, csv, argparse, importlib
ap = argparse.ArgumentParser()
ap.add_argument('--method', action='append', required=True)
ap.add_argument('--res', default=1024, type=int, help='Input size for timing')
ap.add_argument('--mac_res', default=256, type=int)
ap.add_argument('--warmup', default=30, type=int)
ap.add_argument('--iters', default=100, type=int)
ap.add_argument('--model', default='MixDehazeNet-s')
ap.add_argument('--variant', default='D')
ap.add_argument('--gpu', default='0')
ap.add_argument('--out', default='', help='Output csv path')
ap.add_argument('--cudnn_benchmark', action='store_true', help='Enable cudnn.benchmark for all methods')
ap.add_argument('--fuse_bn', action='store_true',
                help='Fuse conv->BN structure before inference using torch.fx')
ap.add_argument('--compile', action='store_true', help='Use torch.compile for all methods')
ap.add_argument('--compile_mode', default='default', help='Mode for torch.compile')
ap.add_argument('--channels_last', action='store_true', help='Use channels_last memory layout for all methods')
ap.add_argument('--profile', default='', help='Run torch.profiler for operator-level timing analysis on this method')
args = ap.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch
if args.cudnn_benchmark:
    torch.backends.cudnn.benchmark = True

def build(spec):
    if spec == 'ours':
        from models import build_model
        return 'EGMix-Net', build_model(args.model, args.variant)
    if spec == 'ours_rp':
        from models import build_model
        from tools.reparam import reparam_model
        return 'EGMix-Net (reparam)', reparam_model(build_model(args.model, args.variant).cuda(), 'sep')
    name, root, mod, fn, kw = spec.split('|')
    root = os.path.abspath(os.path.expanduser(root))
    sys.path.insert(0, root)
    for k in [k for k in sys.modules if k.split('.')[0] == mod.split('.')[0]]:
        del sys.modules[k]
    m = getattr(importlib.import_module(mod), fn)(**json.loads(kw or '{}'))
    sys.path.remove(root)
    return name, m

def macs_of(m, res):
    from torch.utils.flop_counter import FlopCounterMode
    x = torch.randn(1, 3, res, res, device='cuda')
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        m(x)
    return fc.get_total_flops() / 2 / 1e9

rows = []
for spec in args.method:
    try:
        name, m = build(spec)
    except Exception as ex:
        print(f'⚠️ Skipped {spec.split("|")[0]}：Build failed {type(ex).__name__}: {ex}'); continue
    m = m.cuda().eval()
    fused = ''
    if args.fuse_bn and not name.endswith('(reparam)'):
        n_bn = sum(isinstance(x, torch.nn.modules.batchnorm._BatchNorm) for x in m.modules())
        if n_bn:
            try:
                from torch.fx.experimental.optimization import fuse
                m2 = fuse(m)
                with torch.no_grad():
                    xx = torch.randn(1, 3, 256, 256, device='cuda')
                    err = (m(xx) - m2(xx)).abs().max().item()
                left = sum(isinstance(x, torch.nn.modules.batchnorm._BatchNorm) for x in m2.modules())
                m, fused = m2, f'BN {n_bn}→{left}, Error {err:.1e}'
            except Exception as ex:
                fused = f'Not fused ({type(ex).__name__}）'
        print(f'{name}: {fused or "No BN"}')
    params = sum(p.numel() for p in m.parameters()) / 1e6
    try:
        macs = macs_of(m, args.mac_res)
    except Exception as ex:
        macs = float('nan'); print(f'⚠️ {name} MACs Stat failed: {ex}')
    x = torch.randn(1, 3, args.res, args.res, device='cuda')
    if args.channels_last:
        m = m.to(memory_format=torch.channels_last); x = x.contiguous(memory_format=torch.channels_last)
    if args.profile and name == args.profile:
        from torch.profiler import profile, ProfilerActivity
        with torch.no_grad():
            for _ in range(10):
                m(x)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(10):
                    m(x)
                torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=25))
    comp = ''
    if args.compile:
        try:
            mc = torch.compile(m, mode=args.compile_mode)
            with torch.no_grad():
                a, b = mc(x), m(x)
                a, b = (a[-1], b[-1]) if isinstance(a, (list, tuple)) else (a, b)
                err = (a - b).abs().max().item()
            m, comp = mc, f'compiled, Error {err:.1e}'
        except Exception as ex:
            comp = f'Not compiled ({type(ex).__name__}）'; print(f'⚠️ {name} Compile failed: {str(ex)[:300]}')
        print(f'{name}: {comp}')
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        with torch.no_grad():
            for _ in range(args.warmup):
                m(x)
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(args.iters):
                m(x)
            e.record(); torch.cuda.synchronize()
        ms, mem = s.elapsed_time(e) / args.iters, torch.cuda.max_memory_allocated() / 2 ** 20
    except torch.cuda.OutOfMemoryError:
        ms, mem = float('nan'), float('nan'); print(f'⚠️ {name} at {args.res}x{args.res} Out of memory')
    r = {'method': name, 'params_M': round(params, 4), f'MACs_G@{args.mac_res}': round(macs, 2),
         f'ms@{args.res}': round(ms, 2), 'peak_MB': round(mem, 0), 'fuse_bn': fused, 'compile': comp}
    rows.append(r); print(r, flush=True)
    del m; torch.cuda.empty_cache()
os.makedirs('results/speed', exist_ok=True)
out = args.out or (f'results/speed/speed_{args.res}' + ('_cudnn' if args.cudnn_benchmark else '') + ('_fuse' if args.fuse_bn else '') + ('_compile' if args.compile else '') + ('_cl' if args.channels_last else '') + '.csv')
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f'✔ Wrote to {out}(GPU: {torch.cuda.get_device_name(0)}，FP32, batch 1)')
