from pathlib import Path
from copy import deepcopy
from typing import Callable


import torch


def is_name_compressible(name:str):
    is_backbone = name.startswith('backbone.encoder_layers') or name.startswith('backbone.decoder_layers')
    is_weight_matrix = not any([key in name for key in ['bias', 'ln_modulation', 'downsample', 'upsample']])
    return is_backbone and is_weight_matrix

def reset_param(current_param, new_param):
    # SUPER HACKY BUT IT WORKS
    current_param += new_param - current_param

def svd_param(param:torch.Tensor, ratio:float, name:str, grad_path:Path):
    # arguments not most pythonic but okay for now
    U, S, Vh = torch.linalg.svd(param.detach(), full_matrices=False)
    k = round(S.shape[0] * ratio)
    # Apply compression
    return U[:,:k] @ torch.diag(S[:k]) @ Vh[:k]


def _find_wt_path(name:str, grad_path: Path) -> Path:
    # Encoder or decoder
    if name.startswith('backbone.encoder_layers'):
        wt_path = grad_path / 'backbone_encoder'
    elif name.startswith('backbone.decoder_layers'):
        wt_path = grad_path / 'backbone_decoder'
    else:
        assert False, name

    name_parts = name.split('.')
    wt_name =  '.'.join(name_parts[1:3] + ['_checkpoint_wrapped_module'] + name_parts[3:] + ['pt'])
    return wt_path / wt_name
    

def fisher_param(param:torch.Tensor, ratio:float, name:str, grad_path: Path):
    # https://arxiv.org/pdf/2207.00112
    fisher_hat_path = _find_wt_path(name=name, grad_path=grad_path)
    assert fisher_hat_path.is_file(), str(fisher_hat_path) + ' | ' + name
    fisher = torch.load(fisher_hat_path)
    
    # Diagonalize as sum_sqrt of row
    fisher = torch.sqrt(torch.sum(fisher, dim=1))

    # SVD
    U, S, Vh = torch.linalg.svd(torch.diag(fisher) @ param)
    k = round(S.shape[0] * ratio)

    # Apply compression
    epsilon = 1e-8
    fisher = torch.sign(fisher) * torch.clamp(torch.abs(fisher), min=epsilon)
    return torch.diag(1./fisher) @ U[:,:k] @ torch.diag(S[:k]) @ Vh[:k]


def improved_fisher_param(param:torch.Tensor, ratio:float, name:str, grad_path: Path):
    # https://arxiv.org/pdf/2207.00112
    fisher_hat_path = _find_wt_path(name=name, grad_path=grad_path)
    assert fisher_hat_path.is_file(), str(fisher_hat_path) + ' | ' + name
    fisher = torch.load(fisher_hat_path)

    # SVD of element-wise product
    U, S, Vh = torch.linalg.svd(fisher * param)
    k = round(S.shape[0] * ratio)

    # Apply compression --> Very simple approach
    epsilon = 1e-8
    fisher = torch.sign(fisher) * torch.clamp(torch.abs(fisher), min=epsilon)
    return (1./fisher) * (U[:,:k] @ torch.diag(S[:k]) @ Vh[:k])


# ----------------------------------------------------------------
# Compression loops

def compression_loop(original_model, ratio:float, compress_func: Callable, grad_path:Path):
    model = deepcopy(original_model)

    for name,param in model.named_parameters():
        if not is_name_compressible(name):
            continue

        reset_param(
            current_param=param,
            new_param=compress_func(param=param, ratio=ratio, name=name, grad_path=grad_path)
        )

    return model

def svd_only_compression(original_model, ratio:float):
    return compression_loop(
        original_model=original_model, ratio=ratio, compress_func=svd_param, grad_path=Path('.')
    )

def fisher_base_compression(original_model, ratio:float, grad_path:Path):
    return compression_loop(
        original_model=original_model, ratio=ratio, compress_func=fisher_param, grad_path=grad_path
    )

def fisher_improved_compression(original_model, ratio:float, grad_path:Path):
    return compression_loop(
        original_model=original_model, ratio=ratio, compress_func=improved_fisher_param, grad_path=grad_path
    )
