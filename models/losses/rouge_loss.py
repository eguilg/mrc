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

		# focal loss
		fl_w = torch.pow(torch.abs(out_matix - delta_rouge), self.gamma)  # (p-r)^y
		nll_p = - delta_rouge * torch.log(out_matix + 1e-30)  # -r*log(p)
		nll_n = - (1 - delta_rouge) * torch.log(1 - out_matix + 1e-30)  # -(1-r)*log(1-p)
		fl = fl_w * (nll_p + nll_n)

		delta_mask_p = delta_rouge.gt(0).float()
		delta_mask_n = (1 - delta_rouge).gt(0).float()

		alpha = (1 - delta_rouge).sum(-1) / delta_rouge.ge(0).float().sum(-1)
		fl_p = alpha * (delta_mask_p * fl).sum(-1)
		fl_n = (1 - alpha) * (delta_mask_n * fl).sum(-1)

		fl = (fl_p + fl_n).mean()

		return fl
