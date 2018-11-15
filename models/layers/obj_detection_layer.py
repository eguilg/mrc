import torch
import torch.nn as nn
import torch.nn.functional as F
from ptsemseg.models.unet import unet
from torch.autograd import Variable


class SimpleConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
    super(SimpleConvBlock, self).__init__()

    self.conv1 = nn.Sequential(
      nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
      nn.BatchNorm1d(out_channels),
      nn.ReLU(),
    )
    self.conv2 = nn.Sequential(
      nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding),
      nn.BatchNorm1d(out_channels),
      nn.ReLU(),
    )

  def forward(self, inputs):
    outputs = self.conv1(inputs)
    outputs = self.conv2(outputs)
    return outputs


class InceptionConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
    super(InceptionConvBlock, self).__init__()

    self.conv1 = nn.Sequential(
      nn.Conv1d(in_channels, out_channels // 4, 1, 1, 0),
      nn.BatchNorm1d(out_channels),
      nn.ReLU(),
    )

    self.conv2 = nn.Sequential(
      nn.Conv1d(out_channels // 4, out_channels // 2, kernel_size, 1, padding),
      nn.BatchNorm1d(out_channels),
      nn.ReLU(),
    )
    self.conv2 = nn.Sequential(
      nn.Conv1d(out_channels // 4, out_channels // 2, kernel_size, 1, padding),
      nn.BatchNorm1d(out_channels),
      nn.ReLU(),
    )

    self.conv2 = nn.Sequential(
      nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding),
      nn.BatchNorm1d(out_channels),
      nn.ReLU(),
    )

  def forward(self, inputs):
    outputs = self.conv1(inputs)
    outputs = self.conv2(outputs)
    return outputs


class SimpleDetectionNet(nn.Module):
  def __init__(self, in_channels, B=1, S=16, feature_scale=4):
    """

    :param in_channels: hidden_size
    :param feature_scale:
    """
    super().__init__()
    self.B = B
    self.S = S

    filters = [64, 128, 256, 512, 1024]
    filters = [int(x / feature_scale) for x in filters]

    # 输入=[batch_size,hidden_size,n],
    self.conv1 = SimpleConvBlock(in_channels, filters[0], kernel_size=7, padding=7 // 2, stride=2)  # 256
    self.maxpool1 = nn.MaxPool1d(kernel_size=2)  # 128

    self.conv2 = SimpleConvBlock(filters[0], filters[1], kernel_size=3, padding=3 // 2)  # 64
    self.maxpool2 = nn.MaxPool1d(kernel_size=2)  # 32

    # FIXME:  for还不会
    self.conv3 = nn.Sequential(
      nn.Conv1d(filters[1], filters[2] // 4, 1, 1, 0),
      nn.BatchNorm1d(filters[2] // 4),
      nn.ReLU(),

      nn.Conv1d(filters[2] // 4, filters[2] // 2, 3, 1, 1),
      nn.BatchNorm1d(filters[2] // 2),
      nn.ReLU(),

      nn.Conv1d(filters[2] // 2, filters[2] // 2, 1, 1, 0),
      nn.BatchNorm1d(filters[2] // 2),
      nn.ReLU(),

      nn.Conv1d(filters[2] // 2, filters[2], 3, 1, 1),
      nn.BatchNorm1d(filters[2]),
      nn.ReLU(),
    )
    self.maxpool3 = nn.MaxPool1d(kernel_size=2)  # 16

    self.fc = nn.Sequential(
      nn.Linear(filters[2] * S, filters[-1]),
      nn.ReLU(True),
      nn.Dropout(),
      nn.Linear(filters[-1], S * B * 3),
    )

  def forward(self, x):
    net = self.conv1(x)
    net = self.maxpool1(net)

    net = self.conv2(net)
    net = self.maxpool2(net)

    net = self.conv3(net)
    net = self.maxpool3(net)

    net = self.fc(net.view(net.size(0), -1))

    net = F.sigmoid(net)
    net = net.view(-1, self.S, self.B * 3)
    return net


class ObjDetectionNet(nn.Module):
  def __init__(self, x_size, y_size, hidden_size, dropout_rate=0, normalize=True):
    super(ObjDetectionNet, self).__init__()

    self.normalize = normalize
    self.hidden_size = hidden_size
    self.dropout_rate = dropout_rate

    self.dropout = nn.Dropout(p=self.dropout_rate)

    self.detection_net = SimpleDetectionNet(hidden_size * 2, feature_scale=8)

  def forward(self, x, y, x_mask, y_mask):
    outer = self.detection_net(x.transpose(1, 2))
    # s = self.dropout(self.w_s(x))  # b, T, hidden
    # e = self.dropout(self.w_e(x))  # b, T, hidden
    #
    # outer = torch.bmm(s, e.transpose(1, 2))  # b, T, T
    # outer = outer.unsqueeze(1)
    #
    # outer = self.unet(outer)
    #
    # outer = outer.squeeze(1)
    # # print(outer.size())
    # x_len = x.size(1)
    # xx_mask = x_mask.unsqueeze(2).expand(-1, -1, x_len)
    # outer.masked_fill_(xx_mask, -float('inf'))
    # outer.masked_fill_(xx_mask.transpose(1, 2), -float('inf'))
    #
    # outer = F.sigmoid(outer)
    return outer
