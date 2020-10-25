import torch

from util.geom import orient2pose


def orientation_loss(pred_bo, gt_3d, bl, root):
    res = orient2pose(pred_bo, bl, root)
    return torch.nn.functional.l1_loss(res, gt_3d)
