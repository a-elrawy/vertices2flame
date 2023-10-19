#!/usr/bin/env python
import torch.nn.functional as F
import torch.nn as nn

def calc_inv_loss( pose_out, exp_out, pose, exp):
    pose_loss = nn.L1Loss()(pose_out, pose)
    exp_loss = nn.L1Loss()(exp_out, exp)
    total_loss = pose_loss + exp_loss
    return total_loss, [total_loss, pose_loss, exp_loss]
