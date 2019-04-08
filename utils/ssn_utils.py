from __future__ import division

from math import sin, sqrt, acos
import torch


def sparsemax(v, z=1):
    v_sorted, _ = torch.sort(v, dim=0, descending=True)
    cssv = torch.cumsum(v_sorted, dim=0) - z
    ind = torch.arange(1, 1 + len(v)).float().to(v.device)
    cond = v_sorted - cssv / ind > 0
    rho = ind.masked_select(cond)[-1]
    tau = cssv.masked_select(cond)[-1] / rho
    w = torch.clamp(v - tau, min=0)
    return w / z

def sparsestmax(v, rad_in=0, u_in=None):
    w = sparsemax(v)
    if max(w) - min(w) == 1:
        return w
    ind = torch.tensor(w>0).float()
    u = ind / torch.sum(ind)
    if u_in is None:
        u_in = 1 / len(w)
        rad = rad_in
    else:
        rad = sqrt(rad_in**2 - torch.sum((u-u_in)**2))
    distance = torch.norm(w-u)
    if distance >= rad:
        return w
    p = rad*(w-u)/distance+u
    if min(p) < 0:
        return sparsestmax(p, rad, u)
    return p.clamp_(min=0, max=1)
