import torch
import torch.nn as nn
import torch.nn.functional as F


class PointerLoss(nn.Module):
	""" MLE Loss """
	def __init__(self):
		super(PointerLoss, self).__init__()

	def forward(self, out_s, out_e, y_s, y_e):

		loss_s = F.nll_loss(torch.log(out_s + 1e-12), y_s)
		loss_e = F.nll_loss(torch.log(out_e + 1e-12), y_e)

		return loss_s + loss_e
