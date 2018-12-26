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
    self.seg_net = unet(feature_scale=8, in_channels=1, n_classes=1)
    # self.seg_net = MyUNet(feature_scale=8, in_channels=1, n_classes=1)
    # self.seg_net = PSP(1, backbone='resnet50', aux=False, root='~/.gluoncvth/models')
    # self.seg_net = MyPSPNet()
    # self.seg_net = nn.Conv2d(1, 1, 1)

    self.w1 = nn.Linear(x_size, hidden_size, False)
    self.w2 = nn.Linear(hidden_size, c_max_len, False)

    self.w_transform = nn.Sequential(
      nn.Linear(x_size, 128, False),
      nn.ReLU(),

      nn.Linear(128, x_size, False),
      nn.Tanh()
    )

    self.w_t = nn.Linear(x_size, x_size, False)

    self.w_t1 = nn.Linear(x_size, c_max_len, False)
    self.w_t2 = nn.Linear(x_size, c_max_len, False)

    self.row_embeds = nn.Embedding(c_max_len, 2)
    self.col_embeds = nn.Embedding(c_max_len, 2)

  def forward(self, c, q, x_mask, y_mask):
    batch_size = c.size(0)
    x_len = c.size(1)

    s = F.tanh(self.w_s(c))  # b, T, hidden
    e = F.tanh(self.w_e(c))  # b, T, hidden
    outer = torch.bmm(s, e.transpose(1, 2))  # b, T, T

    ## 2MLP模式
    # outer = self.dropout(F.tanh(self.w1(c)))
    # outer = self.dropout(self.w2(outer))
    # outer = F.tanh(self.w_t(c))
    # outer = outer[:, :, :x_len]

    ## 多维图像模式
    # outer1 = F.tanh(self.w_t1(c))
    # outer1 = outer1[:, :, :x_len]
    #
    # outer2 = F.tanh(self.w_t2(c))
    # outer2 = outer2[:, :, :x_len]
    # outer2 = outer2.transpose(1, 2)
    # outer = torch.cat([outer1.unsqueeze(1), outer2.unsqueeze(1)], 1)

    ## 套一堆全连接
    # c = self.w_transform(c)
    # outer = torch.bmm(self.w_t(c), c.transpose(1, 2))
    #
    # ## pos embedding
    # pos_idx = torch.cuda.LongTensor(list(range(self.c_max_len)))
    # pos_idx = Variable(pos_idx)
    # row = self.row_embeds(pos_idx)
    # col = self.col_embeds(pos_idx).transpose(0, 1)
    #
    # pos_outer = torch.mm(row, col).unsqueeze(0)
    # pos_outer = F.tanh(pos_outer)
    # pos_outer = torch.cat([pos_outer for _ in range(batch_size)], 0)
    #
    # ## 拼接在一起
    # outer_ = torch.cat([outer.unsqueeze(1), pos_outer.unsqueeze(1)], 1)

    outer_ = outer.unsqueeze(1)
    outer_ = self.seg_net(outer_)
    outer_ = outer_.squeeze(1)
    outer = outer_
    # outer += outer_

    xx_mask = x_mask.unsqueeze(2).expand(-1, -1, x_len)
    outer.masked_fill_(xx_mask, -float('inf'))
    outer.masked_fill_(xx_mask.transpose(1, 2), -float('inf'))
    outer = F.sigmoid(outer)

    return outer
