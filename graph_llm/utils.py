import math
import random
import os
import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def adjust_lr(param_group, lr, epoch, args):
    min_lr = 5e-6  # TODO: Set to args
    if epoch < args.warmup_epochs:
        lr = lr / args.warmup_epochs
    else:
        lr = min_lr + (lr - min_lr) * 0.5 * (
            1 + math.cos(math.pi * (epoch - args.warmup_e))
        )
    param_group["lr"] = lr
    return lr
