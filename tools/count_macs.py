
import os, sys, json, argparse, importlib
import torch
from torch.utils.flop_counter import FlopCounterMode

ap = argparse.ArgumentParser()
ap.add_argument('--name', required=True)
ap.add_argument('--ext_root', required=True); ap.add_argument('--ext_module', required=True)
ap.add_argument('--ext_fn', required=True); ap.add_argument('--ext_kwargs', default='{}')
ap.add_argument('--res', default=256, type=int)
args = ap.parse_args()
proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path = [q for q in sys.path if os.path.abspath(q or '.') != proj]
sys.path.insert(0, os.path.abspath(os.path.expanduser(args.ext_root)))
net = getattr(importlib.import_module(args.ext_module), args.ext_fn)(**json.loads(args.ext_kwargs)).eval()
x = torch.randn(1, 3, args.res, args.res)
with FlopCounterMode(display=False) as fc, torch.no_grad():
    net(x)
line = f'{args.name},{sum(p.numel() for p in net.parameters()) / 1e6:.3f},{fc.get_total_flops() / 2e9:.2f}'
print(line)
os.makedirs(os.path.join(proj, 'results'), exist_ok=True)
with open(os.path.join(proj, 'results', 'macs_ext.csv'), 'a') as fp:
    fp.write(line + '\n')
