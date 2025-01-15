# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
import torch

from mmdet3d.models.losses import RotatedIoU3DLoss


def test_rotated_iou_3d_loss():

    if not torch.musa.is_available():
        return

    boxes1 = torch.tensor([[.5, .5, .5, 1., 1., 1., .0],
                           [.5, .5, .5, 1., 1., 1., .0],
                           [.5, .5, .5, 1., 1., 1., .0],
                           [.5, .5, .5, 1., 1., 1., .0],
                           [.5, .5, .5, 1., 1., 1., .0]]).musa()
    boxes2 = torch.tensor([[.5, .5, .5, 1., 1., 1., .0],
                           [.5, .5, .5, 1., 1., 2., np.pi / 2],
                           [.5, .5, .5, 1., 1., 1., np.pi / 4],
                           [1., 1., 1., 1., 1., 1., .0],
                           [-1.5, -1.5, -1.5, 2.5, 2.5, 2.5, .0]]).musa()

    expect_ious = 1 - torch.tensor([[1., .5, .7071, 1 / 15, .0]]).musa()
    ious = RotatedIoU3DLoss(reduction='none')(boxes1, boxes2)
    assert torch.allclose(ious, expect_ious, atol=1e-4)
