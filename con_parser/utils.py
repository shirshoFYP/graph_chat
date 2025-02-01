import torch
import torch.nn as nn
import numpy as np
from omegaconf import DictConfig
from omegaconf.listconfig import ListConfig


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cpu")
        n_gpu = 0
    return device, n_gpu


def load_model(model_path):
    device, _ = get_device()
    model = torch.load(model_path, map_location=device)
    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_actions(pred, gt):
    if isinstance(pred[0], list):
        num_correct = np.sum(
            np.sum(x == y for x, y in zip(pred_seq, gt_seq))
            for pred_seq, gt_seq in zip(pred, gt)
        )
        num_total = sum(len(pred_seq) for pred_seq in pred)
    else:
        num_correct = np.sum([x == y for x, y in zip(pred, gt)])
        num_total = len(pred)
    return num_correct, num_total


def conf_to_list(config):
    config_list = []
    for v in config:
        if isinstance(v, DictConfig):
            config_list.append(conf_to_dict(v))
        elif isinstance(v, ListConfig):
            config_list.append(conf_to_list(v))
        else:
            config_list.append(v)
    return config_list


def conf_to_dict(config):
    config_dict = {}
    for k, v in config.items():
        if isinstance(v, DictConfig):
            config_dict[k] = conf_to_dict(v)
        elif isinstance(v, list):
            config_dict[k] = conf_to_list(v)
        else:
            config_dict[k] = v
    return config_dict
