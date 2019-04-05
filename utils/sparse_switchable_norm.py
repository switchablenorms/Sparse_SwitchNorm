import torch
import torch.nn as nn
from .ssn_utils import sparsestmax


class SSN2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.997, using_moving_average=True, last_gamma=False):
        super(SSN2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        
        self.mean_weight = nn.Parameter(torch.ones(3))
        self.var_weight = nn.Parameter(torch.ones(3))
        self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
        self.register_buffer('running_var', torch.zeros(1, num_features, 1))

        self.rad = 0.
        self.register_buffer('mean_fixed', torch.LongTensor([0]))
        self.register_buffer('var_fixed', torch.LongTensor([0]))
        self.register_buffer('radius', torch.zeros(1))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.mean_fixed.data.fill_(0)
        self.var_fixed.data.fill_(0)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)

        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        if self.training:
            mean_bn = mean_in.mean(0, keepdim=True)
            var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
            if self.using_moving_average:
                self.running_mean.mul_(self.momentum)
                self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                self.running_var.mul_(self.momentum)
                self.running_var.add_((1 - self.momentum) * var_bn.data)
            else:
                self.running_mean.add_(mean_bn.data)
                self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
        else:
            mean_bn = torch.autograd.Variable(self.running_mean)
            var_bn = torch.autograd.Variable(self.running_var)

        if not self.mean_fixed:
            self.mean_weight_ = sparsestmax(self.mean_weight, self.rad)
            if max(self.mean_weight_) - min(self.mean_weight_) >= 1:
                self.mean_fixed.data.fill_(1)
                self.mean_weight.data = self.mean_weight_.data
                self.mean_weight_ = self.mean_weight.detach()
        else:
            self.mean_weight_ = self.mean_weight.detach()

        if not self.var_fixed:
            self.var_weight_ = sparsestmax(self.var_weight, self.rad)
            if max(self.var_weight_) - min(self.var_weight_) >= 1:
                self.var_fixed.data.fill_(1)
                self.var_weight.data = self.var_weight_.data
                self.var_weight_ = self.var_weight.detach()
        else:
            self.var_weight_ = self.var_weight.detach()

        mean = self.mean_weight_[0] * mean_in + self.mean_weight_[1] * mean_ln + self.mean_weight_[2] * mean_bn
        var = self.var_weight_[0] * var_in + self.var_weight_[1] * var_ln + self.var_weight_[2] * var_bn

        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias

    def get_mean(self):
        return self.mean_weight_

    def get_var(self):
        return self.var_weight_

    def set_rad(self, rad):
        self.radius[0].fill_(rad)
        self.rad = torch.squeeze(self.radius)

    def get_rad(self):
        return torch.squeeze(self.radius)
