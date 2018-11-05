import torch
import torch.nn as nn
import torch.nn.functional as F
from ptsemseg.models.unet import unet
from torch.autograd import Variable


class OuterNet(nn.Module):
	def __init__(self, x_size, y_size, hidden_size, dropout_rate=0, normalize=True):
		super(OuterNet, self).__init__()
		self.normalize = normalize
		self.hidden_size = hidden_size
		self.dropout_rate = dropout_rate

		# self.w_q = nn.Linear(y_size, x_size, False)
		self.w_s = nn.Linear(x_size, hidden_size, False)
		self.w_e = nn.Linear(x_size, hidden_size, False)

		self.dropout = nn.Dropout(p=self.dropout_rate)

		self.unet = unet(feature_scale=8, in_channels=1, n_classes=1)

	def forward(self, x, y, x_mask, y_mask):

		# gamma = self.w_q(y)  # b, J, x_size
		# yy_mask = y_mask.unsqueeze(2).expand(-1, -1, self.hidden_size)
		# gamma.data.masked_fill_(yy_mask, -float('inf'))
		# gamma = F.softmax(gamma, 1)  # b, J, hidden

		# q = torch.bmm(y.transpose(1, 2), gamma)  # b, y_size, hidden
		# q = F.softmax(q, 1)  # b, y_size, hidden
		s = self.dropout(self.w_s(x))  # b, T, hidden
		e = self.dropout(self.w_e(x))  # b, T, hidden

		outer = torch.bmm(s, e.transpose(1, 2))  # b, T, T
		outer = outer.unsqueeze(1)

		outer = self.unet(outer)

		outer = outer.squeeze(1)
		# print(outer.size())
		x_len = x.size(1)
		xx_mask = x_mask.unsqueeze(2).expand(-1, -1, x_len)
		outer.masked_fill_(xx_mask, -float('inf'))
		outer.masked_fill_(xx_mask.transpose(1, 2), -float('inf'))

		outer = F.sigmoid(outer)
		return outer