import torch
import torch.nn as nn
import torch.nn.functional as F
from ptsemseg.models.unet import unet, MyUNet

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

    self.w_s = nn.Linear(x_size, hidden_size, False)
    self.w_e = nn.Linear(x_size, hidden_size, False)
    #
    self.dropout = nn.Dropout(p=self.dropout_rate)
    #
    # self.unet = MyUNet(feature_scale=8, in_channels=1, n_classes=1)
    # self.seg_net = PSP(1, backbone='resnet50', aux=False, root='~/.gluoncvth/models')
    # self.seg_net = MyPSPNet()
    self.seg_net = nn.Conv2d(1, 1, 1)

    self.w1 = nn.Linear(x_size, hidden_size, False)
    self.w2 = nn.Linear(hidden_size, c_max_len, False)

    self.w_transform = nn.Sequential(
      nn.Linear(x_size, 128, False),
      nn.ReLU(),

      nn.Linear(128, x_size, False),
      nn.Tanh()
    )

    self.w_t = nn.Linear(x_size, x_size, False)

  def forward(self, c, q, x_mask, y_mask):
    batch_size = c.size(0)
    x_len = c.size(1)

    # s = F.tanh(self.w_s(c))  # b, T, hidden
    # e = F.tanh(self.w_e(c))  # b, T, hidden
    # outer = torch.bmm(s, e.transpose(1, 2))  # b, T, T

    # outer = self.dropout(F.tanh(self.w1(c)))
    # outer = self.dropout(self.w2(outer))
    # outer = outer[:, :, :x_len]

    c = self.w_transform(c)
    outer = torch.bmm(self.w_t(c), c.transpose(1, 2))

    outer = outer.unsqueeze(1)
    outer = self.seg_net(outer)
    outer = outer.squeeze(1)

    xx_mask = x_mask.unsqueeze(2).expand(-1, -1, x_len)
    outer.masked_fill_(xx_mask, -float('inf'))
    outer.masked_fill_(xx_mask.transpose(1, 2), -float('inf'))
    outer = F.sigmoid(outer)

    return outer
