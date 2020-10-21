import torch

def th_sp2cart(r, t, p):
    x = r*torch.cos(p)*torch.cos(t)
    y = r*torch.sin(p)*torch.cos(t)
    z = r*torch.sin(t)
    return x, y, z