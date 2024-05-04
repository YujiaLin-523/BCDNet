import torch.optim.lr_scheduler


def StepLR(optimizer, step_size, gamma=0.1):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)


def CosineAnnealingLR(optimizer, T_max, eta_min=0):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min)
