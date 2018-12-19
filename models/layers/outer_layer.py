import torch
import torch.nn as nn
import torch.nn.functional as F
from ptsemseg.models.unet import unet

from torch.autograd import Variable
from gluoncvth.models.pspnet import PSP


class OuterNet(nn.Module):
  def __init__(self, x_size, y_size, hidden_size, dropout_rate=0, normalize=True,
               c_max_len=64):
    super(OuterNet, self).__init__()
    self.normalize = normalize
    self.hidden_size = hidden_size
    self.dropout_rate = dropout_rate
    self.c_max_len = c_max_len

    # self.w_q = nn.Linear(y_size, x_size, False)
    self.w_s = nn.Linear(x_size, hidden_size, False)
    self.w_e = nn.Linear(x_size, hidden_size, False)

    self.dropout = nn.Dropout(p=self.dropout_rate)

    # self.unet = unet(feature_scale=8, in_channels=1, n_classes=1)
    # self.unet = PSP(1, backbone='resnet50', aux=False, root='~/.gluoncvth/models')

  def forward(self, c, q, x_mask, y_mask):
    batch_size = c.size(0)
    x_len = c.size(1)

    s = F.tanh(self.w_s(c))  # b, T, hidden
    e = F.tanh(self.w_e(c))  # b, T, hidden
    outer_ = torch.bmm(s, e.transpose(1, 2))  # b, T, T

    # outer = outer_.unsqueeze(1)
    # outer = self.unet(outer)
    # outer = outer.squeeze(1)
    # outer += outer_
    outer=outer_

    xx_mask = x_mask.unsqueeze(2).expand(-1, -1, x_len)
    outer.masked_fill_(xx_mask, -float('inf'))
    outer.masked_fill_(xx_mask.transpose(1, 2), -float('inf'))
    outer = F.sigmoid(outer)

    return outer
