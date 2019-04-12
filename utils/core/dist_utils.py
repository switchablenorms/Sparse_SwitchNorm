import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from utils import sparse_switchable_norm as ssn
from utils import switchable_norm as sn

__all__ = [
    'init_dist', 'broadcast_params', 'average_gradients', 'sync_bn_stat']


def init_dist(backend='nccl',
              master_ip='127.0.0.1',
              port=29500):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    os.environ['MASTER_ADDR'] = master_ip
    os.environ['MASTER_PORT'] = str(port)
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend)
    return rank, world_size

def average_gradients(model):
    for param in model.parameters():
        if param.requires_grad and not (param.grad is None):
            dist.all_reduce(param.grad.data)

def sync_bn_stat(model, world_size):
    if world_size == 1:
        return
    for mod in model.modules():
        if type(mod) == ssn.SSN2d or type(mod) == sn.SwitchNorm2d:
            dist.all_reduce(mod.running_mean.data)
            mod.running_mean.data.div_(world_size)
            dist.all_reduce(mod.running_var.data)
            mod.running_var.data.div_(world_size)

def broadcast_params(model):
    for p in model.state_dict().values():
        dist.broadcast(p, 0)
