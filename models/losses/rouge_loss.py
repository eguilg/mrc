import torch
import torch.nn as nn


class RougeLoss(nn.Module):
	""" MRT Loss """

	def __init__(self):
		super(RougeLoss, self).__init__()

	def forward(self, out_s, out_e, delta_rouge):
		out_matix = torch.bmm(out_s.unsqueeze(-1), out_e.unsqueeze(-1).transpose(1, 2)).view(out_s.size(0), -1)
		delta_rouge = delta_rouge.view(delta_rouge.size(0), -1)

		loss = delta_rouge * torch.log(out_matix + 1e-30) + (1 - delta_rouge) * torch.log(1 - out_matix + 1e-30)

		return loss
