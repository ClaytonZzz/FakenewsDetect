import numpy as np
import random
import torch
import torch.nn as nn
import os
import shutil

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

# EMA方法，防止模型过拟合：
def update_model_ema(model, model_ema, decay=0.99):
    net_g_params = dict(model.named_parameters())
    net_g_ema_params = dict(model_ema.named_parameters())

    for k in net_g_ema_params.keys():
        net_g_ema_params[k].data.mul_(decay).add_(net_g_params[k].data, alpha=1 - decay)

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=1 / np.sqrt(m.weight.size(1)))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)