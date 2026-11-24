
import torch

def focal_frequency_loss(pred, target, alpha=1.0, log_matrix=False, batch_matrix=False):
    pf = torch.fft.fft2(pred.float(), norm='ortho')
    tf = torch.fft.fft2(target.float(), norm='ortho')
    diff = pf - tf
    dist = diff.real ** 2 + diff.imag ** 2
    with torch.no_grad():
        w = dist.sqrt() ** alpha
        if log_matrix:
            w = torch.log(w + 1.0)
        if batch_matrix:
            w = w / w.max()
        else:
            w = w / w.amax(dim=(-2, -1), keepdim=True)
        w = torch.nan_to_num(w, nan=0.0).clamp(0.0, 1.0)
    return (w * dist).mean()
