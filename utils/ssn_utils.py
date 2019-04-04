from __future__ import division

from math import sin, sqrt, acos
import torch


def regulated_sparsemax(v, rad=0, z=1):
    v_sorted, ind_sorted = torch.sort(v, dim=0, descending=True)
    cssv = torch.cumsum(v_sorted, dim=0) - z
    ind = torch.arange(1, 1 + len(v)).float().to(v.device)
    cond = v_sorted - cssv / ind > 0
    rho = ind.masked_select(cond)[-1]
    tau = cssv.masked_select(cond)[-1] / rho
    w = torch.clamp(v - tau, min=0)

    u = torch.tensor(1/3).to(v.device)
    distance = torch.sum((w-u)**2)**(1/2)
    if distance >= rad:
        return w
    if distance < 1e-6:
        distance = distance.detach()
    p = rad*(w-u)/(distance+1e-6)+u

    if p[ind_sorted[2]] < 0:
        p[ind_sorted[0]] = (1+sqrt(2)*rad*sin(acos(1/(sqrt(6)*rad))))/2
        p[ind_sorted[1]] = (1-sqrt(2)*rad*sin(acos(1/(sqrt(6)*rad))))/2
        p[ind_sorted[2]] = 0
    p.clamp_(min=0, max=1)
    return p
