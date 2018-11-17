import torch
import torch.nn as nn
import torch.nn.functional as F
from ptsemseg.models.unet import unet
from torch.autograd import Variable


class SimpleConv1d(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
    super(SimpleConv1d, self).__init__()

    self.conv1 = nn.Sequential(
      nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
      nn.BatchNorm1d(out_channels),
      nn.ReLU(),
    )

  def forward(self, inputs):
    outputs = self.conv1(inputs)
    return outputs


class VGGConv1d(nn.Module):
  def __init__(self, in_channels, out_channels, times=2, kernel_size=3, stride=1, padding=1):
    super(VGGConv1d, self).__init__()

    layers = []
    for time in range(times):
      if time == 0:
        layers.append(nn.Sequential(
          nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
          nn.BatchNorm1d(out_channels),
          nn.ReLU(),
        ))
      else:
        layers.append(nn.Sequential(
          nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding),
          nn.BatchNorm1d(out_channels),
          nn.ReLU(),
        ))

    self.conv = nn.Sequential(*layers)

  def forward(self, inputs):
    outputs = self.conv(inputs)
    return outputs


class InceptionConv1dBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
    super(InceptionConv1dBlock, self).__init__()

    self.conv1 = nn.Sequential(
      nn.Conv1d(in_channels, out_channels // 2, 1, 1, 0),
      nn.BatchNorm1d(out_channels // 2),
      nn.ReLU(),
    )
    self.conv2 = nn.Sequential(
      nn.Conv1d(out_channels // 2, out_channels, kernel_size, stride, padding),
      nn.BatchNorm1d(out_channels),
      nn.ReLU(),
    )

  def forward(self, inputs):
    outputs = self.conv1(inputs)
    outputs = self.conv2(outputs)

    return outputs


class InceptionConv1d(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, times=1):
    super(InceptionConv1d, self).__init__()

    inception_layers = []
    for time in range(times):
      if time == 0:
        inception_layers.append(InceptionConv1dBlock(in_channels, out_channels // 2, kernel_size, stride, padding))
      else:
        inception_layers.append(
          InceptionConv1dBlock(out_channels // 2, out_channels // 2, kernel_size, stride, padding))

    self.inception_conv = nn.Sequential(*inception_layers)

    self.conv1 = nn.Sequential(
      nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
      nn.BatchNorm1d(out_channels),
      nn.ReLU(),
    )

  def forward(self, inputs):
    outputs = self.inception_conv(inputs)
    outputs = self.conv1(outputs)
    return outputs


class VGGNet1d(nn.Module):
  def __init__(self, in_channels, filters):
    super().__init__()
    layers = []
    for i in range(len(filters)):
      if i == 0:
        layers.append(VGGConv1d(in_channels, filters[0], times=2, stride=1))
      elif i < 2:
        layers.append(VGGConv1d(filters[i - 1], filters[i], times=2))
      else:
        layers.append(VGGConv1d(filters[i - 1], filters[i], times=3))
      layers.append(nn.MaxPool1d(kernel_size=2))

    self.conv_net = nn.Sequential(*layers)

  def forward(self, x):
    net = self.conv_net(x)

    return net


class SimpleNet1d(nn.Module):
  def __init__(self, in_channels, filters):
    super().__init__()
    self.conv1 = VGGConv1d(in_channels, filters[0], kernel_size=7, padding=3, stride=2)  # 256
    self.maxpool1 = nn.MaxPool1d(kernel_size=2)  # 128

    self.conv2 = VGGConv1d(filters[0], filters[1], kernel_size=3, padding=1)  # 64
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

  def forward(self, x):
    net = self.conv1(x)
    net = self.maxpool1(net)

    net = self.conv2(net)
    net = self.maxpool2(net)

    net = self.conv3(net)
    net = self.maxpool3(net)


    return net


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

    # self.conv_net = VGGNet1d(in_channels, filters)  # 收敛的速度很慢，不知道是不是因为网络太大了

    filters=filters[:3]
    self.conv_net = SimpleNet1d(in_channels, filters)

    self.fc = nn.Sequential(
      nn.Linear(filters[-1] * S, 256),
      nn.ReLU(True),
      nn.Dropout(),
      nn.Linear(256, S * B * 3),
    )


  def forward(self, x):
    net = self.conv_net(x)

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
    return outer
