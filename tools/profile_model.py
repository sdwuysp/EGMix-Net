
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='MixDehazeNet-s')
parser.add_argument('--variants', default='ABCDEF')
parser.add_argument('--res', default=256, type=int)
parser.add_argument('--gpu', default='0')
parser.add_argument('--iters', default=200, type=int)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
from models import build_model

def count_macs(model, x):
    try:
        from torch.utils.flop_counter import FlopCounterMode   
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            model(x)
        return fc.get_total_flops() / 2
    except ImportError:
        from thop import profile
        macs, _ = profile(model, inputs=(x,), verbose=False)
        return macs

print(f'{"variant":8s}{"params(M)":>11s}{"MACs(G)":>10s}{"latency(ms)":>13s}  @ {args.res}x{args.res}')
for v in args.variants:
    m = build_model(args.model, v).eval()
    x = torch.randn(1, 3, args.res, args.res)
    params = sum(p.numel() for p in m.parameters()) / 1e6
    macs = count_macs(m, x) / 1e9
    lat = float('nan')
    if torch.cuda.is_available():
        m, x = m.cuda(), x.cuda()
        with torch.no_grad():
            for _ in range(50):
                m(x)
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(args.iters):
                m(x)
            e.record()
            torch.cuda.synchronize()
            lat = s.elapsed_time(e) / args.iters
    print(f'{v:8s}{params:11.4f}{macs:10.2f}{lat:13.2f}')
