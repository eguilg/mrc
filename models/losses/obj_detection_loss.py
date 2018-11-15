import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjDetectionLoss(nn.Module):
  def __init__(self, B, S, dynamic_score=False, coord=5.0, noobj=0.5):
    super().__init__()
    self.B = B
    self.S = S
    self.dynamic_score = dynamic_score
    self.coord = coord
    self.noobj = noobj

    if self.B > 1:
      ## FIXME: 如果B>1 静态score可能会导致模型不知道输出哪个结果??
      assert dynamic_score

  def forward(self, out, width, center, score):
    """

    :param out:  [batch_size,S,B*3]
    :param width:
    :param center:
    :param score:
    :return:
    """
    if self.dynamic_score:
      pass

    # fixme: 临时的做法, 统一成[bs,S] ,# dataset里的B好像不需要？
    width = width.view(-1, self.S)
    center = center.view(-1, self.S)
    score = score.view(-1, self.S)

    pos_mask = score.gt(0).float()
    neg_mask = (1 - score).gt(0).float()
    loss_center = torch.pow(pos_mask * (out[:, :, 0] - center), 2)
    loss_width = torch.pow(pos_mask * (out[:, :, 1] - width), 2)

    loss_score = self.coord * torch.pow(pos_mask * (out[:, :, 2] - score), 2)
    loss_noobj_score = self.noobj * torch.pow(neg_mask * (out[:, :, 2] - score), 2)

    loss = loss_center + loss_width + loss_score + loss_noobj_score
    loss = loss.mean()

    return loss
