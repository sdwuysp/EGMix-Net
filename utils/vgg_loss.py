
import torch
import torch.nn as nn
import torchvision

class VGGPerceptual(nn.Module):
    CUTS = (4, 9, 18, 27, 36)
    W = (1 / 32, 1 / 16, 1 / 8, 1 / 4, 1.0)

    def __init__(self):
        super().__init__()
        try:
            f = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1).features
        except AttributeError:
            f = torchvision.models.vgg19(pretrained=True).features
        self.slices = nn.ModuleList([f[a:b] for a, b in zip((0,) + self.CUTS[:-1], self.CUTS)])
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.eval()

    def forward(self, x, y):
        x = ((x.float() * 0.5 + 0.5) - self.mean) / self.std
        y = ((y.float() * 0.5 + 0.5) - self.mean) / self.std
        loss = 0.0
        for w, s in zip(self.W, self.slices):
            x, y = s(x), s(y)
            loss = loss + w * (x - y).abs().mean()
        return loss
