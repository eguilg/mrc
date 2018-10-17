import torch
import torch.nn as nn


class RougeLoss(nn.Module):
	""" Rouge Loss """

	def __init__(self):
		super(RougeLoss, self).__init__()

	def forward(self, out_s, out_e, delta_rouge):
		out_matix = torch.bmm(out_s.unsqueeze(-1), out_e.unsqueeze(-1).transpose(1, 2)).view(out_s.size(0), -1)
		delta_rouge = delta_rouge.view(delta_rouge.size(0), -1)

		# batch, len**2
		cross_entropy = -(delta_rouge * torch.log(out_matix) + (1 - delta_rouge) * torch.log(1 - out_matix))


		delta_p_mask = delta_rouge.gt(0).float()
		sum_p = (cross_entropy * delta_p_mask).sum(-1)  # batch
		num_p = delta_p_mask.sum(-1)  # batch
		loss_p = (sum_p / num_p).mean()

		delta_n_mask = delta_rouge.eq(0).float()
		sum_n = (cross_entropy * delta_n_mask).sum(-1)
		num_n = delta_n_mask.sum(-1)
		loss_n = (sum_n / num_n).mean()

		return loss_p + loss_n
