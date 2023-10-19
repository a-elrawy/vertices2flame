#!/usr/bin/env python
import os
import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from tensorboardX import SummaryWriter
import cv2

from base.baseTrainer import save_checkpoint
from base.utilities import get_parser, get_logger, AverageMeter
from models import FlameInverter
from metrics.loss import calc_inv_loss
from torch.optim.lr_scheduler import StepLR
from dataset import convert_to_vertices

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def main():
    cfg = get_parser()
    cfg.gpu = cfg.train_gpu

    # ####################### Model ####################### #
    global logger, writer
    logger = get_logger()
    writer = SummaryWriter(cfg.save_path)
    model = FlameInverter(cfg)
        
    logger.info(cfg)
    logger.info("=> creating model ...")

    torch.cuda.set_device(gpu)
    model = model.cuda()

    # ####################### Optimizer ####################### #
    if cfg.use_sgd:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.base_lr, momentum=cfg.momentum,
                                    weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.base_lr)

    if cfg.StepLR:
        scheduler = StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    else:
        scheduler = None

    # ####################### Data Loader ####################### #
    from dataset.data_loader import get_dataloaders
    dataset = get_dataloaders(cfg)
    train_loader = dataset['train']
    test_loader = dataset['test']
    if cfg.evaluate:
        val_loader = dataset['valid']

    # ####################### Train ############################# #
    for epoch in range(cfg.start_epoch, cfg.epochs):
        try:
            rec_loss_train, pose_loss_train, exp_loss_train = train(train_loader, model, calc_inv_loss, optimizer, epoch, cfg)
            epoch_log = epoch + 1
            if cfg.StepLR:
                scheduler.step()
            if main_process(cfg):
                logger.info('TRAIN Epoch: {} '
                            'loss_train: {} '
                            'pose_train: {} '
                            'exp_train: {} '
                            .format(epoch_log, rec_loss_train, pose_loss_train, exp_loss_train)
                            )
                for m, s in zip([rec_loss_train,pose_loss_train,exp_loss_train],
                                ["train/rec_loss", "train/pose","train/exp"]):
                    writer.add_scalar(s, m, epoch_log)


            if cfg.evaluate and (epoch_log % cfg.eval_freq == 0):
                rec_loss_val, pose_loss_val, exp_loss_val = validate(val_loader, model, calc_inv_loss, epoch, cfg)
                if main_process(cfg):
                    logger.info('VAL Epoch: {} '
                                'loss_val: {} '
                                'pose_val: {} '
                                'exp_val: {} '
                                .format(epoch_log, rec_loss_val, pose_loss_val,exp_loss_val)
                                )
                    for m, s in zip([rec_loss_val, pose_loss_val, exp_loss_val],
                                    ["val/rec_loss","val/pose", "val/exp"]):
                        writer.add_scalar(s, m, epoch_log)


            if (epoch_log % cfg.save_freq == 0) and main_process(cfg):
                save_checkpoint(model,
                                sav_path=os.path.join(cfg.save_path, 'model')
                                )
        except:
            continue

def train(train_loader, model, loss_fn, optimizer, epoch, cfg):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    rec_loss_meter = AverageMeter()
    pose_loss = AverageMeter()
    exp_loss = AverageMeter()

    model.train()
    end = time.time()
    max_iter = cfg.epochs * len(train_loader)
    for i, (vertices, pose, exp) in enumerate(train_loader):
        current_iter = epoch * len(train_loader) + i + 1
        data_time.update(time.time() - end)
        vertices = vertices.cuda(cfg.gpu, non_blocking=True)
        pose = pose.cuda(cfg.gpu, non_blocking=True)
        exp = exp.cuda(cfg.gpu, non_blocking=True)

        pose_out, exp_out = model(vertices)
        loss, loss_details = loss_fn(pose_out, exp_out, pose.squeeze(0), exp.squeeze(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        for m, x in zip([rec_loss_meter, pose_loss, exp_loss],
                        [loss_details[0], loss_details[1], loss_details[2]]): #info[0] is perplexity
            m.update(x.item(), 1)
        
        # Adjust lr
        current_lr = optimizer.param_groups[0]['lr']

        # calculate remain time
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % cfg.print_freq == 0 and main_process(cfg):
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain: {remain_time} '
                        'Loss: {loss_meter.val:.4f} '
                        .format(epoch + 1, cfg.epochs, i + 1, len(train_loader),
                                batch_time=batch_time, data_time=data_time,
                                remain_time=remain_time,
                                loss_meter=rec_loss_meter
                                ))
            for m, s in zip([rec_loss_meter],
                            ["train_batch/loss"]):
                writer.add_scalar(s, m.val, current_iter)
            writer.add_scalar('learning_rate', current_lr, current_iter)

    return rec_loss_meter.avg, pose_loss.avg, exp_loss.avg


def validate(val_loader, model, loss_fn, epoch, cfg):
    rec_loss_meter = AverageMeter()
    pose_loss = AverageMeter()
    exp_loss = AverageMeter()
    model.eval()

    with torch.no_grad():
        for i, (vertices, pose, exp) in enumerate(val_loader):
            vertices = vertices.cuda(cfg.gpu, non_blocking=True)
            pose = pose.cuda(cfg.gpu, non_blocking=True)
            exp = exp.cuda(cfg.gpu, non_blocking=True)

            pose_out, exp_out = model(vertices)

            # LOSS
            loss, loss_details = loss_fn(pose_out, exp_out, pose, exp)

            for m, x in zip([rec_loss_meter, pose_loss, exp_loss],
                            [loss_details[0], loss_details[1], loss_details[2]]):
                m.update(x.item(), 1) #batch_size = 1 for validation


    return rec_loss_meter.avg, pose_loss.avg, exp_loss.avg


if __name__ == '__main__':
    main()
