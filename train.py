import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='MixDehazeNet-s', type=str, help='Model scale: MixDehazeNet-t/s/b/l')
parser.add_argument('--variant', default='D', type=str, help='Ablation variants A-F, D is the full model')
parser.add_argument('--exp', default='indoor', type=str, help='Experiment name, corresponds to configs/<exp>/ directory')
parser.add_argument('--config', default='', type=str, help='Specify config file path directly (optional, highest priority)')
parser.add_argument('--dataset', default='RESIDE-IN', type=str, help='Dataset subdirectory name')
parser.add_argument('--data_dir', default='./data/', type=str, help='Dataset root directory')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='Path to save weights')
parser.add_argument('--log_dir', default='./logs/', type=str, help='Path to csv logs')
parser.add_argument('--tag', default='', type=str, help='Additional tag appended to run name, e.g. p256 / seed1')
parser.add_argument('--seed', default=2026, type=int, help='Random seed')
parser.add_argument('--gpu', default='0', type=str, help='GPUs to use, e.g. 0 or 0,1')
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--no_amp', action='store_true', help='Disable mixed precision (AMP)')
parser.add_argument('--train_subset', default=0, type=int, help='Use only N training pairs (0 = all), for small-scale pilot experiments')
parser.add_argument('--val_holdout', default=0, type=int,
                    help='Hold out N scenes from training set for validation (0 = use test set for validation)')
parser.add_argument('--split_seed', default=0, type=int, help='Random seed for dataset splitting / hold-out validation')
parser.add_argument('--ext_file', default='', type=str,
                    help='External model definition file (for baseline comparison)')
parser.add_argument('--ext_fn', default='', type=str, help='External model constructor function name')
parser.add_argument('--init_ckpt', default='', type=str,
                    help='Initialize from existing weights before training (loads only network weights)')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import json
import time
import random
import shutil
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import AverageMeter
from datasets.loader import PairLoader
from models import build_model
from utils.CR_res import ContrastLoss_res
from utils.lf_loss import lf_loss, lf_loss_tw
from utils.ffl_loss import focal_frequency_loss
from utils.aux_losses import dped_color_loss, multiscale_content_loss
from utils.vgg_loss import VGGPerceptual

def amp_autocast(enabled):
    try:
        return torch.amp.autocast('cuda', enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)

def amp_scaler(enabled):
    try:
        return torch.amp.GradScaler('cuda', enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def worker_init_fn(worker_id):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s + worker_id)
    random.seed(s + worker_id)

def load_ckpt(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')

def log(msg):
    print(f'[{datetime.datetime.now().strftime("%m-%d %H:%M:%S")}] {msg}', flush=True)

def fft_loss(preds, target, norm='backward'):

    preds_fft = torch.fft.rfft2(preds.float(), norm=norm)
    target_fft = torch.fft.rfft2(target.float(), norm=norm)
    return torch.abs(preds_fft - target_fft).mean()

def train_one_epoch(loader, network, criterion, optimizer, scaler, setting, use_amp):
    losses = AverageMeter()
    network.train()
    accum = setting['accum_steps']
    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        source_img = batch['source'].cuda(non_blocking=True)
        target_img = batch['target'].cuda(non_blocking=True)

        with amp_autocast(use_amp):
            output = network(source_img)
            loss = criterion['l1'](output, target_img)
            if setting['cr_weight'] > 0:
                loss = loss + setting['cr_weight'] * criterion['cr'](output, target_img, source_img)
            if setting['fft_weight'] > 0:
                loss = loss + setting['fft_weight'] * fft_loss(output, target_img, setting['fft_norm'])
            if setting['ffl_weight'] > 0:
                loss = loss + setting['ffl_weight'] * focal_frequency_loss(output, target_img, setting['ffl_alpha'])
            if setting['lf_weight'] > 0:
                if setting['lf_tw']:
                    loss = loss + setting['lf_weight'] * lf_loss_tw(output.float(), target_img.float(), source_img.float(),
                                                                    tuple(setting['lf_scales']), setting['lf_global'], setting['lf_tw_tau'],
                                                                    setting['lf_tw_global'])
                else:
                    loss = loss + setting['lf_weight'] * lf_loss(output.float(), target_img.float(),
                                                                 tuple(setting['lf_scales']), setting['lf_global'])
            if setting['color_weight'] > 0:
                loss = loss + setting['color_weight'] * dped_color_loss(output, target_img)
            if setting['vgg_weight'] > 0:
                loss = loss + setting['vgg_weight'] * criterion['vgg'](output, target_img)
            if setting['ms_weight'] > 0:
                loss = loss + setting['ms_weight'] * multiscale_content_loss(output, target_img,
                                                                             tuple(setting['ms_scales']))

        losses.update(loss.item())
        scaler.scale(loss / accum).backward()

        if (i + 1) % accum == 0 or (i + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    return losses.avg

@torch.no_grad()
def valid(loader, network):
    PSNR = AverageMeter()
    network.eval()
    for batch in loader:
        source_img = batch['source'].cuda(non_blocking=True)
        target_img = batch['target'].cuda(non_blocking=True)
        output = network(source_img).clamp_(-1, 1)
        mse = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
        psnr = 10 * torch.log10(1 / mse.clamp_min(1e-10)).mean()
        PSNR.update(psnr.item(), source_img.size(0))
    return PSNR.avg

@torch.no_grad()
def recal_bn(loader, network, n_batches):

    bns = [m for m in network.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return
    mom = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats(); m.momentum = None
    network.train(); k = 0
    for batch in loader:
        network(batch['source'].cuda(non_blocking=True)); k += 1
        if k >= n_batches:
            break
    for m, v in zip(bns, mom):
        m.momentum = v
    network.eval()

@torch.no_grad()
def valid_cf(loader, network):

    P, Pc, C = AverageMeter(), AverageMeter(), AverageMeter()
    network.eval()
    for batch in loader:
        x = batch['source'].cuda(non_blocking=True); y = batch['target'].cuda(non_blocking=True) * 0.5 + 0.5
        o = network(x).clamp_(-1, 1) * 0.5 + 0.5
        e = o - y; m = e.mean((2, 3), keepdim=True)
        P.update((10 * torch.log10(1 / e.pow(2).mean((1, 2, 3)).clamp_min(1e-10))).mean().item(), x.size(0))
        Pc.update((10 * torch.log10(1 / (e - m).pow(2).mean((1, 2, 3)).clamp_min(1e-10))).mean().item(), x.size(0))
        C.update(m.pow(2).mean((1, 2, 3)).sqrt().mul(255).mean().item(), x.size(0))
    return P.avg, Pc.avg, C.avg

def load_setting():
    if args.config:
        path = args.config
    else:
        path = os.path.join('configs', args.exp, args.model + '.json')
        if not os.path.exists(path):
            path = os.path.join('configs', args.exp, 'default.json')
    with open(path, 'r') as f:
        setting = json.load(f)
    setting.setdefault('accum_steps', 1)
    setting.setdefault('cr_weight', 0.1)
    setting.setdefault('fft_weight', 0.1)
    setting.setdefault('fft_norm', 'backward')
    setting.setdefault('lr_min_ratio', 0.01)
    setting.setdefault('lf_weight', 0.0)
    setting.setdefault('lf_scales', [8, 16, 32])
    setting.setdefault('lf_tw', False)
    setting.setdefault('lf_tw_tau', 0.02)
    setting.setdefault('lf_tw_global', 't')
    setting.setdefault('lf_start_ratio', 0.0)
    setting.setdefault('ffl_weight', 0.0)
    setting.setdefault('ffl_alpha', 1.0)
    setting.setdefault('lf_global', True)
    setting.setdefault('color_weight', 0.0)
    setting.setdefault('ms_weight', 0.0)
    setting.setdefault('ms_scales', [2, 4])
    setting.setdefault('vgg_weight', 0.0)
    setting.setdefault('precise_bn', 0)
    return setting, path

if __name__ == '__main__':
    setting, setting_path = load_setting()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    use_amp = not args.no_amp

    run_name = f'{args.model}_{args.variant}' + (f'_{args.tag}' if args.tag else '')
    run_dir = os.path.join(args.save_dir, args.exp, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(args.log_dir, args.exp), exist_ok=True)
    csv_path = os.path.join(args.log_dir, args.exp, run_name + '.csv')
    best_path = os.path.join(run_dir, 'best.pth')
    latest_path = os.path.join(run_dir, 'latest.pth')
    final_path = os.path.join(run_dir, 'final.pth')

    log('=' * 60)
    log(f'Run name: {run_name} | Config: {setting_path}')
    log(f'Setting: {json.dumps(setting, ensure_ascii=False)}')
    log(f'Args: {json.dumps(vars(args), ensure_ascii=False)}')

    with open(os.path.join(run_dir, 'run_config.jsonl'), 'a') as f:
        f.write(json.dumps({'time': datetime.datetime.now().isoformat(timespec='seconds'),
                            'config_file': setting_path, 'setting': setting, 'args': vars(args)},
                           ensure_ascii=False) + '\n')

    if args.ext_file:
        import importlib.util
        spec = importlib.util.spec_from_file_location('ext_model', os.path.expanduser(args.ext_file))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        network = getattr(mod, args.ext_fn)()
        log(f'External model: {args.ext_file} :: {args.ext_fn}')
    else:
        network = build_model(args.model, args.variant)
    n_params = sum(p.numel() for p in network.parameters())
    log(f'Model: {args.model} Variant {args.variant} | Parameters {n_params / 1e6:.4f} M')
    network = nn.DataParallel(network).cuda()

    criterion = {'l1': nn.L1Loss()}
    if setting['cr_weight'] > 0:
        criterion['cr'] = ContrastLoss_res(ablation=False).cuda()
    if setting['vgg_weight'] > 0:
        criterion['vgg'] = VGGPerceptual().cuda()

    if setting['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'])
    elif setting['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'])
    else:
        raise ValueError('unsupported optimizer')

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=setting['epochs'], eta_min=setting['lr'] * setting['lr_min_ratio'])
    scaler = amp_scaler(use_amp)

    start_epoch, best_psnr, best_epoch = 0, 0.0, -1
    if os.path.exists(latest_path):
        shutil.copy(latest_path, latest_path + '.backup')
        ckpt = load_ckpt(latest_path)
        if ckpt.get('variant', args.variant) != args.variant:
            raise RuntimeError(f'Variant {ckpt.get("variant")}  --variant {args.variant} ')
        network.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['lr_scheduler'])
        scheduler.T_max = setting['epochs']
        done = scheduler.last_epoch
        for g, base in zip(optimizer.param_groups, scheduler.base_lrs):
            g['lr'] = float(scheduler.eta_min + (base - scheduler.eta_min) *
                            (1 + np.cos(np.pi * min(done, setting['epochs']) / setting['epochs'])) / 2)
        scaler.load_state_dict(ckpt['scaler'])
        best_psnr = ckpt.get('best_psnr', 0.0)
        best_epoch = ckpt.get('best_epoch', -1)
        start_epoch = ckpt['epoch'] + 1
        log(f'Resume: From epoch {start_epoch + 1} | Current best {best_psnr:.4f} dB (epoch {best_epoch})')
    else:
        if args.init_ckpt:
            init = load_ckpt(args.init_ckpt)
            sd = init['state_dict'] if 'state_dict' in init else init
            sd = {(k if k.startswith('module.') else 'module.' + k): v for k, v in sd.items()}
            network.load_state_dict(sd, strict=True)
            log(f'From {args.init_ckpt} Initialize network weights (optimizer and LR restarted)')
        else:
            log('，From')

    dataset_dir = os.path.join(args.data_dir, args.dataset)
    train_dataset = PairLoader(dataset_dir, 'train', 'train', setting['patch_size'],
                               setting['edge_decay'], setting['only_h_flip'])
    val_dataset = None
    if args.train_subset > 0 or args.val_holdout > 0:
        rng = np.random.RandomState(args.split_seed)
        gt_names = train_dataset.gt_names
        scenes = sorted(set(gt_names))
        rng.shuffle(scenes)
        hold = set(scenes[:args.val_holdout])
        train_idx = [i for i, gname in enumerate(gt_names) if gname not in hold]
        if args.train_subset > 0:
            train_idx = sorted(rng.permutation(train_idx)[:args.train_subset].tolist())
        if args.val_holdout > 0:
            full_res = PairLoader(dataset_dir, 'train', 'test')
            by_scene = {}
            for i, gname in enumerate(full_res.gt_names):
                if gname in hold:
                    by_scene.setdefault(gname, []).append(i)
            val_idx = sorted(int(rng.choice(v)) for v in by_scene.values())
            val_dataset = torch.utils.data.Subset(full_res, val_idx)
            val_files = [full_res.hazy_names[i] for i in val_idx]
        else:
            val_files = []
        split_path = os.path.join(run_dir, 'split.json')
        if not os.path.exists(split_path):
            with open(split_path, 'w') as f:
                json.dump({'split_seed': args.split_seed, 'train_subset': args.train_subset,
                           'val_holdout': args.val_holdout, 'n_train': len(train_idx),
                           'val_files': val_files,
                           'train_files': [train_dataset.hazy_names[i] for i in train_idx]}, f)
        train_dataset = torch.utils.data.Subset(train_dataset, train_idx)
        log(f'Subset mode: Train {len(train_idx)} pairs | Hold out {len(val_files)} scenes | Split saved to {split_path}')
    g = torch.Generator()
    g.manual_seed(args.seed + start_epoch)
    train_loader = DataLoader(train_dataset, batch_size=setting['batch_size'], shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True,
                              worker_init_fn=worker_init_fn, generator=g,
                              persistent_workers=args.num_workers > 0)
    if val_dataset is None:
        val_dataset = PairLoader(dataset_dir, 'test', setting['valid_mode'], setting['patch_size'])
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=args.num_workers, pin_memory=True)

    eff_bs = setting['batch_size'] * setting['accum_steps']
    log(f'Train set {len(train_dataset)} pairs | Val set {len(val_dataset)} pairs | '
        f'batch {setting["batch_size"]} x Accum {setting["accum_steps"]} = Effective {eff_bs} | patch {setting["patch_size"]}')

    cf_path = csv_path[:-4] + '.cf.csv'
    best_cf, best_cf_epoch = -1.0, -1
    if setting['precise_bn'] > 0:
        if os.path.exists(cf_path):
            for r in open(cf_path).read().strip().split('\n')[1:]:
                f_ = r.split(',')
                try:
                    if float(f_[2]) > best_cf:
                        best_cf, best_cf_epoch = float(f_[2]), int(f_[0])
                except (ValueError, IndexError):
                    pass
        else:
            open(cf_path, 'w').write('epoch,val_psnr,val_psnr_cf,cast,best_cf,best_cf_epoch\n')
        log(f'precise BN：Before each val, use {setting["precise_bn"]} training batches to re-estimate BN; de-color-shift PSNR saved to {cf_path}')
    new_csv = not os.path.exists(csv_path)
    csv_f = open(csv_path, 'a', buffering=1)
    if new_csv:
        csv_f.write('time,epoch,train_loss,lr,val_psnr,best_psnr,best_epoch,epoch_sec\n')

    total = setting['epochs']
    epoch_times = []
    lf_start = int(round(setting['lf_start_ratio'] * total)) if setting['lf_weight'] > 0 else 0
    if lf_start > 0:
        log(f'LF Delay addition: No LF for epochs 1-{lf_start}  epoch , add LF from epoch {lf_start + 1}  epoch (weight {setting["lf_weight"]}）；'
            f'best.pth at LF ')
    for epoch in range(start_epoch, total):
        t0 = time.time()
        if lf_start > 0 and epoch == lf_start:
            best_psnr, best_epoch = 0.0, -1
            log(f' {epoch + 1}  epoch  LF，best.pth From')
        ep_setting = setting if epoch >= lf_start else {**setting, 'lf_weight': 0.0}
        loss = train_one_epoch(train_loader, network, criterion, optimizer, scaler, ep_setting, use_amp)
        lr_used = optimizer.param_groups[0]['lr']
        scheduler.step()

        ep = epoch + 1
        do_eval = (ep % setting['eval_freq'] == 0) or (ep == total)
        if do_eval and setting['precise_bn'] > 0:
            recal_bn(train_loader, network, setting['precise_bn'])
            val_psnr, val_cf, val_cast = valid_cf(val_loader, network)
            if val_cf > best_cf:
                best_cf, best_cf_epoch = val_cf, ep
            with open(cf_path, 'a') as f_:
                f_.write(f'{ep},{val_psnr:.4f},{val_cf:.4f},{val_cast:.3f},{best_cf:.4f},{best_cf_epoch}\n')
        else:
            val_psnr = valid(val_loader, network) if do_eval else float('nan')

        is_best = do_eval and val_psnr > best_psnr
        if is_best:
            best_psnr, best_epoch = val_psnr, ep

        ckpt = {'state_dict': network.state_dict(), 'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                'epoch': epoch, 'best_psnr': best_psnr, 'best_epoch': best_epoch,
                'variant': args.variant, 'model': args.model, 'setting': setting}
        torch.save(ckpt, latest_path)
        if is_best:
            torch.save(ckpt, best_path)
        if do_eval and setting['precise_bn'] > 0 and best_cf_epoch == ep:
            torch.save(ckpt, os.path.join(run_dir, 'best_cf.pth'))
        if ep == total:
            torch.save(ckpt, final_path)

        dt = time.time() - t0
        epoch_times.append(dt)
        eta_h = np.mean(epoch_times[-5:]) * (total - ep) / 3600
        csv_f.write(f'{datetime.datetime.now().isoformat(timespec="seconds")},{ep},{loss:.6f},{lr_used:.3e},'
                    f'{"" if not do_eval else f"{val_psnr:.4f}"},{best_psnr:.4f},{best_epoch},{dt:.0f}\n')
        msg = f'Epoch {ep}/{total} | loss {loss:.4f} | lr {lr_used:.2e} | {dt / 60:.1f} min | ETA {eta_h:.1f} h'
        if do_eval:
            msg += f' | Val PSNR {val_psnr:.4f} (best {best_psnr:.4f} @ {best_epoch})'
            if setting['precise_bn'] > 0:
                msg += f' | De-shift {val_cf:.4f} Shift {val_cast:.2f}'
            if is_best:
                msg += ' *'
        log(msg)

    csv_f.close()
    log(f'Training finished | Best {best_psnr:.4f} dB @ epoch {best_epoch} | Final weights {final_path}')
