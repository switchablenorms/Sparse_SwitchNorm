from __future__ import division
import argparse
import os
import time
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import yaml
import sys
from tensorboardX import SummaryWriter
import models
import os.path as osp

sys.path.append(osp.abspath(osp.join(__file__, '../../')))

from utils.core import *
from utils.imagenet_dataset import ColorAugmentation, ImagenetDataset
from utils import sparse_switchable_norm as ssn

parser = argparse.ArgumentParser(
    description='Pytorch Imagenet Training')
parser.add_argument('--config', default='configs/config_resnetv1ssn50.yaml')
parser.add_argument("--local_rank", type=int)
parser.add_argument('--port', default=29500, type=int, help='port of server')
parser.add_argument('--world-size', default=1, type=int)
parser.add_argument('--rank', default=0, type=int)
parser.add_argument('--model_dir', type=str)
parser.add_argument('--checkpoint_path', type=str, default=None)
parser.add_argument('--resume_from', default='', help='resume_from')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
args = parser.parse_args()


def main():
    global args, best_prec1
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)

    for key in config:
        for k, v in config[key].items():
            if (k in args.__dict__) is False:
                setattr(args, k, v)
            elif args.__dict__[k] is None:
                setattr(args, k, v)

    print('Enabled distributed training.')

    rank, world_size = init_dist(
        backend='nccl', port=args.port)
    args.rank = rank
    args.world_size = world_size

    if rank == 0:
        print('################################')
        print('Parameters')
        print(args)
        print('################################')

    # create model
    print("=> creating model '{}'".format(args.model))
    if 'resnetv1sn' in args.model:
        model = models.__dict__[args.model](using_moving_average=args.using_moving_average, using_bn=args.using_bn,
                                            last_gamma=args.last_gamma)
    else:
        model = models.__dict__[args.model](using_moving_average=args.using_moving_average, last_gamma=args.last_gamma)

    model.cuda()
    broadcast_params(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.model.lower().find('ssn') >= 0:
        ssn_param, param = model.get_ssn_param()
        optimizer = torch.optim.SGD(
            [
                {'params': param},
                {'params': ssn_param, 'lr': args.base_lr/10, 'weight_decay': 0},
            ],
            args.base_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.base_lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    if args.lr_mode == 'step':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.step, args.decay_factor)
    elif args.lr_mode == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # auto resume from a checkpoint
    model_dir = args.model_dir
    start_epoch = 0
    if args.rank == 0 and not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if args.evaluate:
        load_state_ckpt(args.checkpoint_path, model)
    else:
        best_prec1, start_epoch = load_state(model_dir, model, optimizer=optimizer, lr_scheduler=lr_scheduler)
    if args.rank == 0:
        writer = SummaryWriter(model_dir)
    else:
        writer = None

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = ImagenetDataset(
        args.train_root,
        args.train_source,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
            ),
            transforms.ToTensor(),
            ColorAugmentation(),
            normalize,
        ]))
    val_dataset = ImagenetDataset(
        args.val_root,
        args.val_source,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size // args.world_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, sampler=train_sampler)

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size // args.world_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, sampler=val_sampler)

    if args.evaluate:
        validate(val_loader, model, criterion, args.epochs, writer)
        return

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)

        if epoch == args.epochs-5:
            train_dataset = ImagenetDataset(
                args.train_root,
                args.train_source,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.4,
                    ),
                    transforms.ToTensor(),
                    ColorAugmentation(),
                    normalize,
                ]))
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size // args.world_size, shuffle=False,
                num_workers=args.workers, pin_memory=False, sampler=train_sampler)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, lr_scheduler, epoch, writer)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch, writer)

        if rank == 0:
            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint(model_dir, {
                'epoch': epoch + 1,
                'model': args.model,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
            }, is_best)


def train(train_loader, model, criterion, optimizer, lr_scheduler, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    world_size = args.world_size
    rank = args.rank
    lr_scheduler.step()

    end = time.time()
    trainset_length = len(train_loader)
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        lr_scheduler.update(i, epoch)
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input.cuda())
        target_var = torch.autograd.Variable(target)

        # update radius for SSN
        if args.model.lower().find('ssn') >= 0:
            rad = (epoch * trainset_length + i) / (args.epochs * trainset_length)
            assert (rad >= 0.0 and rad <= 1.0)
            for module in model.modules():
                if isinstance(module, ssn.SSN2d):
                    module.set_rad(rad)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var) / world_size

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))

        reduced_loss = loss.data.clone()
        reduced_prec1 = prec1.clone() / world_size
        reduced_prec5 = prec5.clone() / world_size

        dist.all_reduce(reduced_loss)
        dist.all_reduce(reduced_prec1)
        dist.all_reduce(reduced_prec5)

        losses.update(reduced_loss.item(), input.size(0))
        top1.update(reduced_prec1.item(), input.size(0))
        top5.update(reduced_prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # if weight in SSN is sparse, fix it by removing its gradient
        if args.model.lower().find('ssn') >= 0:
            for module in model.modules():
                if isinstance(module, ssn.SSN2d):
                    if module.mean_fixed:
                        module.mean_weight.grad = None
                    if module.var_fixed:
                        module.var_weight.grad = None

        average_gradients(model)
        if args.model.lower().find('ssn') >= 0:
            torch.nn.utils.clip_grad_norm_(model.get_ssn_param()[0], 1.)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and rank == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            niter = epoch * len(train_loader) + i
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], niter)
            writer.add_scalar('Train/Avg_Loss', losses.avg, niter)
            writer.add_scalar('Train/Avg_Top1', top1.avg / 100.0, niter)
            writer.add_scalar('Train/Avg_Top5', top5.avg / 100.0, niter)
            if args.model.lower().find('ssn') >= 0:
                rad_ = 0
                n = ['_in', '_ln', '_bn']
                for name, module in model.named_modules():
                    if isinstance(module, ssn.SSN2d):
                        mean = module.get_mean().detach()
                        var = module.get_var().detach()
                        rad_ = module.get_rad()
                        for j in range(3):
                            writer.add_scalar('Train/%s/mean%s' % (name, n[j]), mean[j].item(), niter)
                            writer.add_scalar('Train/%s/var%s' % (name, n[j]), var[j].item(), niter)
                writer.add_scalar('Train/radius', rad_, niter)


def validate(val_loader, model, criterion, epoch, writer):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    world_size = args.world_size
    rank = args.rank

    sync_bn_stat(model, world_size)

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input.cuda())
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var) / world_size

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))

            reduced_loss = loss.data.clone()
            reduced_prec1 = prec1.clone() / world_size
            reduced_prec5 = prec5.clone() / world_size

            dist.all_reduce(reduced_loss)
            dist.all_reduce(reduced_prec1)
            dist.all_reduce(reduced_prec5)

            losses.update(reduced_loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and rank == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
        if rank == 0:
            print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))

            niter = (epoch + 1)
            writer.add_scalar('Eval/Avg_Loss', losses.avg, niter)
            writer.add_scalar('Eval/Avg_Top1', top1.avg / 100.0, niter)
            writer.add_scalar('Eval/Avg_Top5', top5.avg / 100.0, niter)
            if args.model.lower().find('ssn') >= 0:
                rad = 0.
                n = ['_in', '_ln', '_bn']
                for name, module in model.named_modules():
                    if isinstance(module, ssn.SSN2d):
                        mean = module.get_mean().detach()
                        var = module.get_var().detach()
                        rad = module.get_rad()
                        for j in range(3):
                            writer.add_scalar('Eval/%s/mean%s' % (name, n[j]), mean[j].item(), niter)
                            writer.add_scalar('Eval/%s/var%s' % (name, n[j]), var[j].item(), niter)
                writer.add_scalar('Eval/radius', rad, niter)

    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
