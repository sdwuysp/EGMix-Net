import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='MixDehazeNet-s', type=str)
parser.add_argument('--variant', default='D', type=str, help='Ablation variants A-F, must match the weights')
parser.add_argument('--ckpt', default='', type=str,
                    help='Checkpoint path; if empty, uses default path')
parser.add_argument('--exp', default='indoor', type=str)
parser.add_argument('--tag', default='', type=str)
parser.add_argument('--dataset', default='RESIDE-IN', type=str)
parser.add_argument('--data_dir', default='./data/', type=str)
parser.add_argument('--save_dir', default='./saved_models/', type=str)
parser.add_argument('--result_dir', default='./results/', type=str)
parser.add_argument('--tta', action='store_true', help='Enable 8x geometric self-ensemble (TTA)')
parser.add_argument('--save_imgs', action='store_true', help='Save dehazed output images')
parser.add_argument('--tlc_patch', default=0, type=int,
                    help='Enable TLC for global gate (>0 uses local average of training patch size during inference), only for variants G/H')
parser.add_argument('--ext_file', default='', type=str, help='External model definition file (same as training)')
parser.add_argument('--ext_fn', default='', type=str)
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--num_workers', default=4, type=int)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import csv
import json
import datetime
from collections import OrderedDict
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.utils.data import DataLoader

from utils import AverageMeter, write_img, chw_to_hwc
from datasets.loader import PairLoader
from models import build_model, set_tlc

def load_weights(path):

    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location='cpu')
    state = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    new_state = OrderedDict()
    for k, v in state.items():
        new_state[k[7:] if k.startswith('module.') else k] = v
    meta = {k: ckpt.get(k) for k in ('epoch', 'best_psnr', 'best_epoch', 'variant')} if isinstance(ckpt, dict) else {}
    return new_state, meta

def tta_forward(network, x):

    outs = []
    for k in range(4):
        xr = torch.rot90(x, k, dims=[2, 3])
        outs.append(torch.rot90(network(xr).clamp_(-1, 1), -k, dims=[2, 3]))
        xf = torch.flip(xr, dims=[3])
        of = torch.flip(network(xf).clamp_(-1, 1), dims=[3])
        outs.append(torch.rot90(of, -k, dims=[2, 3]))
    return torch.stack(outs).mean(0)

@torch.no_grad()
def evaluate(loader, network, out_dir):
    PSNR, SSIM_DS, SSIM_FULL = AverageMeter(), AverageMeter(), AverageMeter()
    rows = []
    network.eval()
    for idx, batch in enumerate(loader):
        x = batch['source'].cuda()
        y = batch['target'].cuda()
        name = batch['filename'][0]

        out = tta_forward(network, x) if args.tta else network(x).clamp_(-1, 1)
        out = out * 0.5 + 0.5
        y = y * 0.5 + 0.5

        mse = F.mse_loss(out, y).item()
        psnr = 100.0 if mse == 0 else 10 * torch.log10(torch.tensor(1.0 / mse)).item()

        _, _, H, W = out.shape
        r = max(1, round(min(H, W) / 256))
        ssim_ds = ssim(F.adaptive_avg_pool2d(out, (int(H / r), int(W / r))),
                       F.adaptive_avg_pool2d(y, (int(H / r), int(W / r))),
                       data_range=1, size_average=False).item()
        ssim_full = ssim(out, y, data_range=1, size_average=False).item()

        PSNR.update(psnr); SSIM_DS.update(ssim_ds); SSIM_FULL.update(ssim_full)
        rows.append((name, psnr, ssim_ds, ssim_full))

        if args.save_imgs:
            write_img(os.path.join(out_dir, 'imgs', name), chw_to_hwc(out.squeeze(0).cpu().numpy()))
        if idx % 50 == 0:
            print(f'[{idx}/{len(loader)}] PSNR {PSNR.avg:.2f} | SSIM(ds) {SSIM_DS.avg:.4f} | SSIM(full) {SSIM_FULL.avg:.4f}',
                  flush=True)
    return PSNR.avg, SSIM_DS.avg, SSIM_FULL.avg, rows

if __name__ == '__main__':
    run_name = f'{args.model}_{args.variant}' + (f'_{args.tag}' if args.tag else '')
    ckpt_path = args.ckpt or os.path.join(args.save_dir, args.exp, run_name, 'best.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Cannot find weights: {ckpt_path}')

    if args.ext_file:
        import importlib.util
        spec = importlib.util.spec_from_file_location('ext_model', os.path.expanduser(args.ext_file))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        network = getattr(mod, args.ext_fn)()
    else:
        network = build_model(args.model, args.variant)
    state, meta = load_weights(ckpt_path)
    if meta.get('variant') and meta['variant'] != args.variant:
        print(f'⚠️ Variant {meta["variant"]}, but --variant is {args.variant}')
    network.load_state_dict(state, strict=True)
    n_tlc = set_tlc(network, args.tlc_patch) if args.tlc_patch else 0
    if args.tlc_patch:
        print(f'TLC enabled: {n_tlc} global gates replaced by {args.tlc_patch} local average')
    network.cuda()
    epoch_str = '' if meta.get('epoch') is None else str(meta['epoch'] + 1)
    print(f'==>  {ckpt_path} | epoch {epoch_str or "?"} | Variant {args.variant} | TTA {"ON" if args.tta else "OFF"}')

    dataset_dir = os.path.join(args.data_dir, args.dataset)
    loader = DataLoader(PairLoader(dataset_dir, 'test', 'test'), batch_size=1,
                        num_workers=args.num_workers, pin_memory=True)

    ck_tag = os.path.splitext(os.path.basename(ckpt_path))[0]
    out_dir = os.path.join(args.result_dir, args.exp, run_name, ck_tag + ('_tta' if args.tta else '')
                           + (f'_tlc{args.tlc_patch}' if args.tlc_patch else ''))
    os.makedirs(os.path.join(out_dir, 'imgs'), exist_ok=True)

    psnr, ssim_ds, ssim_full, rows = evaluate(loader, network, out_dir)

    with open(os.path.join(out_dir, 'per_image.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['filename', 'psnr', 'ssim_ds', 'ssim_full'])
        for r in rows:
            w.writerow([r[0], f'{r[1]:.4f}', f'{r[2]:.5f}', f'{r[3]:.5f}'])

    summary = {'time': datetime.datetime.now().isoformat(timespec='seconds'), 'exp': args.exp,
               'dataset': args.dataset, 'run': run_name, 'ckpt': ckpt_path, 'epoch': epoch_str,
               'variant': args.variant, 'tta': int(args.tta), 'tlc': args.tlc_patch, 'n': len(rows),
               'psnr': f'{psnr:.4f}', 'ssim_ds': f'{ssim_ds:.5f}', 'ssim_full': f'{ssim_full:.5f}'}
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    sum_path = os.path.join(args.result_dir, 'summary.csv')
    if os.path.exists(sum_path):
        with open(sum_path) as f:
            header = f.readline().strip().split(',')
        if header != list(summary.keys()):
            sum_path = os.path.join(args.result_dir, 'summary_v2.csv')
    new = not os.path.exists(sum_path)
    with open(sum_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if new:
            w.writeheader()
        w.writerow(summary)

    print(f'✅ Finished | PSNR {psnr:.2f} dB | SSIM(ds) {ssim_ds:.4f} | SSIM(full) {ssim_full:.4f} | Result dir {out_dir}')
