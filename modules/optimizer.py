import torch


def Adam(params, lr=0.001):
    return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)


def SGD(params, lr=0.001):
    return torch.optim.SGD(params, lr=lr, momentum=0, dampening=0, weight_decay=0, nesterov=False)
