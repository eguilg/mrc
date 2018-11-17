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

    # for b in range(2):
    #   for pr_width, pr_center, pr_score, gt_width, gt_center, gt_score in zip(
    #       out[b, :, 1], out[b, :, 0], out[b, :, 2], width[b, :], center[b, :], score[b, :]):
    #     print('Pr:', float(pr_width), float(pr_center), float(pr_score))
    #     print('Gt:', float(gt_width), float(gt_center), float(gt_score))
    #     print()

    pos_mask = score.gt(0).float()
    neg_mask = (1 - score).gt(0).float()  # FIXME: 如果评分不是1或0 这里会出问题

    ## pos loss:
    loss_center = pos_mask * torch.abs(torch.sqrt(out[:, :, 0]) - torch.sqrt(center))
    loss_width = pos_mask * torch.abs(torch.sqrt(out[:, :, 1]) - torch.sqrt(width))
    loss_prob = pos_mask * torch.pow((out[:, :, 2] - score), 2)

    ## neg loss:
    # neg_loss_center = torch.pow(neg_mask * (out[:, :, 0] - center), 2)
    # neg_loss_width = torch.pow(neg_mask * (out[:, :, 1] - width), 2)
    neg_loss_prob = neg_mask * torch.pow((out[:, :, 2] - score), 2)

    ## 所有参数都带上
    # loss = self.coord * (loss_center + loss_width + loss_prob) + \
    #        self.noobj * (neg_loss_center + neg_loss_width + neg_loss_prob)

    ## 不拟合非responsible的width和center
    loss = self.coord * (loss_center + loss_width) + loss_prob + \
           self.noobj * (neg_loss_prob)

    loss = loss.sum(-1).mean()
    return loss
