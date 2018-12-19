import torch
import torch.nn as nn
import torch.nn.functional as F


class RougeLoss(nn.Module):
  """ Rouge Loss """

  def __init__(self, gamma=2, alpha=0.01):
    super(RougeLoss, self).__init__()
    self.gamma = gamma
    self.alpha = alpha

  def handcrafted_loss(self, out_matrix, delta_rouge):
    out_matrix = out_matrix.view(out_matrix.size(0), -1)
    delta_rouge = delta_rouge.view(delta_rouge.size(0), -1)

    cross_entropy = -(
        delta_rouge * torch.log(out_matrix + 1e-30) + (1 - delta_rouge) * torch.log(1 - out_matrix + 1e-30))

    delta_p_mask = delta_rouge.gt(0).float()
    sum_p = (cross_entropy * delta_p_mask).sum(-1)  # batch
    num_p = delta_p_mask.sum(-1)  # batch
    loss_p = (sum_p / num_p).mean()

    delta_n_mask = delta_rouge.eq(0).float()
    sum_n = (cross_entropy * delta_n_mask).sum(-1)
    num_n = delta_n_mask.sum(-1)
    loss_n = (sum_n / num_n).mean()

    loss = 10 * loss_p + loss_n
    return loss

  def mse_loss(self, out_matrix, delta_rouge):
    out_matrix = out_matrix.view(out_matrix.size(0), -1)
    delta_rouge = delta_rouge.view(delta_rouge.size(0), -1)

    mse_diff = torch.abs(out_matrix - delta_rouge)
    mse_diff = torch.pow(mse_diff, 2)

    delta_p_mask = delta_rouge.gt(0).float()
    sum_p = (mse_diff * delta_p_mask).sum(-1)  # batch
    num_p = delta_p_mask.sum(-1)  # batch
    loss_p = (sum_p / num_p).mean()

    delta_n_mask = delta_rouge.eq(0).float()
    sum_n = (mse_diff * delta_n_mask).sum(-1)
    num_n = delta_n_mask.sum(-1)
    loss_n = (sum_n / num_n).mean()

    loss = loss_p + loss_n
    return loss

  def focal(self, out_matix, delta_rouge):
    out_matix = out_matix.view(out_matix.size(0), -1)
    delta_rouge = delta_rouge.view(delta_rouge.size(0), -1)

    # focal loss weight
    # celi = torch.where(delta_rouge > out_matix, delta_rouge, 1 - delta_rouge)
    dist = torch.abs(out_matix - delta_rouge)
    fl_w = - torch.log(1 - torch.pow(dist, self.gamma) + 1e-30)  # (p-r)^y

    #  higher weight for high rouge samples
    # w_p = tocch.exp(delta_rouge)

    # nll
    nll_p = - delta_rouge * torch.log(out_matix + 1e-30)  # -r*log(p)
    nll_n = - (1 - delta_rouge) * torch.log(1 - out_matix + 1e-30)  # -(1-r)*log(1-p)

    # focal loss
    fl = fl_w * (nll_p + nll_n)

    # balance pos and neg
    delta_mask_p = delta_rouge.gt(0).float()
    delta_mask_n = (1 - delta_rouge).gt(0).float()
    fl_p = (delta_mask_p * fl).sum(-1) / delta_rouge.sum(-1)
    fl_n = (delta_mask_n * fl).sum(-1) / (1 - delta_rouge).sum(-1)

    fl = (5 * fl_p + fl_n).mean()
    return fl

  def kl_div(self, out_matix, delta_rouge):
    return - F.kl_div(out_matix, delta_rouge)

  def margin_ranking(self, out_matix, delta_rouge):
    out_matix = out_matix.view(out_matix.size(0), -1)
    delta_rouge = delta_rouge.view(delta_rouge.size(0), -1)

    delta_mask_p = delta_rouge.gt(0)
    delta_mask_n = delta_rouge.eq(0)
    total_loss = None
    for i in range(delta_rouge.size(0)):
      out_p = torch.masked_select(out_matix[i], delta_mask_p[i])
      out_n = torch.masked_select(out_matix[i], delta_mask_n[i])
      rouge_p = torch.masked_select(delta_rouge[i], delta_mask_p[i])

      margin_loss = torch.clamp(out_n - self.alpha * out_p.min()[0], 0).mean()

      rouge_p_sorted, sorted_idx = torch.sort(rouge_p, descending=True)
      out_p_sorted = out_p.index_select(0, sorted_idx)

      x1 = out_p_sorted
      x2 = torch.cat([out_p_sorted[1:], out_n.max()[0].unsqueeze(-1)])
      margins = rouge_p_sorted - torch.cat([rouge_p_sorted[1:], self.alpha * rouge_p.min()[0].unsqueeze(-1)])

      ranking_loss = torch.where(x2 - x1 + margins > 0, x2 - x1 + margins, x1 - x1).mean()

      distance = torch.abs(rouge_p_sorted - out_p_sorted).mean()

      # ranking_loss = None
      # for j in range(min(rouge_p_sorted.size(0), 10)):
      # 	margin = rouge_p_sorted[j] - delta_rouge[i]
      # 	y = (margin > 0).float() * 2 - 1
      # 	cri = - y * (out_p_sorted[j] - out_matix[i] - margin)
      # 	loss = torch.where(cri > 0, cri, cri - cri)
      # 	if ranking_loss is None:
      # 		ranking_loss = loss
      # 	else:
      # 		ranking_loss += loss
      # ranking_loss = ranking_loss.mean()

      if total_loss is None:
        total_loss = ranking_loss + 2 * margin_loss + 2 * distance
      else:
        total_loss += ranking_loss + 2 * margin_loss + 2 * distance
    return total_loss / delta_rouge.size(0)

  def forward(self, out_matix, delta_rouge):
    # return self.kl_div(out_matix,delta_rouge)
    # return self.focal(out_matix,delta_rouge)
    # return self.margin_ranking(out_matix, delta_rouge)
    # return self.handcrafted_loss(out_matix, delta_rouge)
    # return self.mse_loss(out_matix, delta_rouge) * 2 + self.margin_ranking(out_matix, delta_rouge)
    return self.mse_loss(out_matix, delta_rouge)
