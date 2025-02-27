import torch.nn as nn

from ptsemseg.models.utils import *


class unet(nn.Module):
  def __init__(
      self,
      feature_scale=4,
      n_classes=21,
      is_deconv=True,
      in_channels=3,
      is_batchnorm=True,
  ):
    super(unet, self).__init__()
    self.is_deconv = is_deconv
    self.in_channels = in_channels
    self.is_batchnorm = is_batchnorm
    self.feature_scale = feature_scale

    filters = [64, 128, 256, 512, 1024]
    filters = [int(x / self.feature_scale) for x in filters]

    # downsampling
    self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
    self.maxpool1 = nn.MaxPool2d(kernel_size=2)

    self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
    self.maxpool2 = nn.MaxPool2d(kernel_size=2)

    self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
    self.maxpool3 = nn.MaxPool2d(kernel_size=2)

    self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
    self.maxpool4 = nn.MaxPool2d(kernel_size=2)

    self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

    # upsampling
    self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
    self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
    self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
    self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

    # final conv (without any concat)
    self.final = nn.Conv2d(filters[0], n_classes, 1)

  def forward(self, inputs):
    conv1 = self.conv1(inputs)
    maxpool1 = self.maxpool1(conv1)

    conv2 = self.conv2(maxpool1)
    maxpool2 = self.maxpool2(conv2)

    conv3 = self.conv3(maxpool2)
    maxpool3 = self.maxpool3(conv3)

    conv4 = self.conv4(maxpool3)
    maxpool4 = self.maxpool4(conv4)

    center = self.center(maxpool4)

    # print(inputs.size())
    # print(conv1.size())
    # print(maxpool1.size())
    # print(conv2.size())
    # print(maxpool2.size())
    # print(conv3.size())
    # print(maxpool3.size())
    # print(conv4.size())
    # print(maxpool4.size())
    # print(center.size())

    up4 = self.up_concat4(conv4, center)

    up3 = self.up_concat3(conv3, up4)
    up2 = self.up_concat2(conv2, up3)
    up1 = self.up_concat1(conv1, up2)

    final = self.final(up1)

    return final


class MyUNet(nn.Module):
  def __init__(
      self,
      feature_scale=4,
      n_classes=1,
      is_deconv=True,
      in_channels=1,
      is_batchnorm=True,
  ):
    super(MyUNet, self).__init__()
    self.is_deconv = is_deconv
    self.in_channels = in_channels
    self.is_batchnorm = is_batchnorm
    self.feature_scale = feature_scale

    filters = [64, 128, 256, 512, 1024]
    filters = [int(x / self.feature_scale) for x in filters]

    # downsampling
    self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
    self.maxpool1 = nn.MaxPool2d(kernel_size=2)

    self.conv2 = unetConv2(self.in_channels, filters[1], self.is_batchnorm)
    self.maxpool2 = nn.MaxPool2d(kernel_size=2)

    self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
    self.maxpool3 = nn.MaxPool2d(kernel_size=2)

    self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
    self.maxpool4 = nn.MaxPool2d(kernel_size=2)

    self.center = unetConv2(filters[0], filters[1], self.is_batchnorm)

    self.up_concat4 = MyunetUp(filters[4], filters[3], self.is_deconv)
    self.up_concat3 = MyunetUp(filters[3], filters[2], self.is_deconv)
    self.up_concat2 = MyunetUp(filters[1], 1, self.is_deconv)
    self.up_concat1 = MyunetUp(filters[0], 1, self.is_deconv)

    # final conv (without any concat)
    self.final = nn.Conv2d(1, n_classes, 1)

  def forward(self, inputs):
    conv1 = self.conv1(inputs)
    maxpool1 = self.maxpool1(conv1)
    up1 = self.up_concat1(inputs, maxpool1)

    # conv2 = self.conv2(up1)
    # maxpool2 = self.maxpool2(conv2)
    # up2= self.up_concat2(inputs, maxpool2)

    # conv2 = self.conv2(conv1)
    # # maxpool2 = self.maxpool2(conv2)
    #
    # conv3 = self.conv3(conv2)
    # # maxpool3 = self.maxpool3(conv3)
    #
    # conv4 = self.conv4(conv3)
    # # maxpool4 = self.maxpool4(conv4)
    #
    # center = self.center(conv4)

    # print(inputs.size())
    # print(conv1.size())
    # print(maxpool1.size())
    # print(conv2.size())
    # print(maxpool2.size())
    # print(conv3.size())
    # print(maxpool3.size())
    # print(conv4.size())
    # print(maxpool4.size())
    # print(center.size())

    # up4 = self.up_concat4(conv4, center)

    # up3 = self.up_concat3(conv3, up4)
    # up2 = self.up_concat2(conv2, up3)
    # up1 = self.up_concat1(inputs, maxpool1)

    final = self.final(up1)

    return final
