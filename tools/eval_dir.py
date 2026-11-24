
import os, csv, argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim

ap = argparse.ArgumentParser()
ap.add_argument('--pred', required=True)
ap.add_argument('--gt', required=True)
ap.add_argument('--name', required=True)
args = ap.parse_args()
pred, gt = os.path.expanduser(args.pred), os.path.expanduser(args.gt)
gts = {os.path.splitext(f)[0]: f for f in os.listdir(gt)}

def load(p):
    return torch.from_numpy(cv2.imread(p)[:, :, ::-1].astype(np.float32) / 255).permute(2, 0, 1)[None]

rows = []
for f in sorted(os.listdir(pred)):
    if not f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        continue
    stem = os.path.splitext(f)[0].replace('_processed', '')
    key = stem if stem in gts else stem.split('_')[0]
    if key not in gts:
        print(f'⚠️ Cannot find {f} 's GT'); continue
    o, y = load(os.path.join(pred, f)), load(os.path.join(gt, gts[key]))
    if o.shape != y.shape:
        print(f'⚠️ {f} size {tuple(o.shape)} and GT {tuple(y.shape)} mismatch, skipped'); continue
    mse = F.mse_loss(o, y).item()
    psnr = 100.0 if mse == 0 else 10 * np.log10(1 / mse)
    _, _, H, W = o.shape
    r = max(1, round(min(H, W) / 256))
    sd = ssim(F.adaptive_avg_pool2d(o, (int(H / r), int(W / r))), F.adaptive_avg_pool2d(y, (int(H / r), int(W / r))),
              data_range=1, size_average=False).item()
    sf = ssim(o, y, data_range=1, size_average=False).item()
    e = (o - y).double() * 2
    sig = e.pow(2).mean().sqrt().item()
    cast = (e.mean(dim=(2, 3)).pow(2).mean() / e.pow(2).mean().clamp_min(1e-20)).item()
    rows.append((f, psnr, sd, sf, sig, cast))
os.makedirs('results/eval_dir', exist_ok=True)
with open(f'results/eval_dir/{args.name}.csv', 'w', newline='') as fp:
    w = csv.writer(fp); w.writerow(['filename', 'psnr', 'ssim_ds', 'ssim_full', 'sigma', 'cast_frac']); w.writerows(rows)
a = np.array([r[1:] for r in rows])
print(f'✔ {args.name}: {len(rows)}  | PSNR {a[:, 0].mean():.2f} | SSIM(ds) {a[:, 1].mean():.4f} | SSIM(full) {a[:, 2].mean():.4f}'
      f' | Shift {np.median(a[:, 4]):.3f} | Shift RMS  {np.mean(np.sqrt(a[:, 4]) * a[:, 3]) * 127.5:.2f}')
