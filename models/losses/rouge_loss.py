import torch
import torch.nn as nn


class RougeLoss(nn.Module):
	""" Rouge Loss """

	def __init__(self, gamma=2):
		super(RougeLoss, self).__init__()
		self.gamma = gamma

	def forward(self, out_s, out_e, delta_rouge):
		out_matix = torch.bmm(out_s.unsqueeze(-1), out_e.unsqueeze(-1).transpose(1, 2)).view(out_s.size(0), -1)
		delta_rouge = delta_rouge.view(delta_rouge.size(0), -1)

		mrt = ((1 - delta_rouge) * out_matix).sum(-1).mean()

		return mrt
