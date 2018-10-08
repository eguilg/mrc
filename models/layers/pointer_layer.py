import torch
import torch.nn as nn
import torch.nn.functional as F

from .att_layer import NonLinearSeqAttn
from .common_layer import FeedForwardNetwork, SFU


class PointerNetwork(nn.Module):

	def __init__(self, x_size, y_size, hidden_size, dropout_rate=0, cell_type=nn.GRUCell, normalize=True):
		super(PointerNetwork, self).__init__()
		self.normalize = normalize
		self.hidden_size = hidden_size
		self.dropout_rate = dropout_rate
		self.linear = nn.Linear(x_size + y_size, hidden_size, bias=False)
		self.weights = nn.Linear(hidden_size, 1, bias=False)
		self.self_attn = NonLinearSeqAttn(y_size, hidden_size)
		self.cell = cell_type(x_size, y_size)

	def init_hiddens(self, y, y_mask):
		attn = self.self_attn(y, y_mask)
		res = attn.unsqueeze(1).bmm(y).squeeze(1)  # [B, I]
		return res

	def pointer(self, x, state, x_mask):
		x_ = torch.cat([x, state.unsqueeze(1).repeat(1, x.size(1), 1)], 2)
		s0 = F.tanh(self.linear(x_))
		s = self.weights(s0).view(x.size(0), x.size(1))
		s.data.masked_fill_(x_mask.data, -float('inf'))
		a = F.softmax(s, -1)
		res = a.unsqueeze(1).bmm(x).squeeze(1)
		if self.normalize:
			# 0-1 probabilities
			scores = F.softmax(s, -1)
		else:
			scores = a.exp()
		return res, scores

	def forward(self, x, y, x_mask, y_mask):
		hiddens = self.init_hiddens(y, y_mask)
		c, start_scores = self.pointer(x, hiddens, x_mask)
		c_ = F.dropout(c, p=self.dropout_rate, training=self.training)
		hiddens = self.cell(c_, hiddens)
		c, end_scores = self.pointer(x, hiddens, x_mask)
		return start_scores, end_scores


class MemoryAnsPointer(nn.Module):
	def __init__(self, x_size, y_size, hidden_size, hop=3, dropout_rate=0, normalize=True):
		super(MemoryAnsPointer, self).__init__()
		self.normalize = normalize
		self.hidden_size = hidden_size
		self.hop = hop
		self.dropout_rate = dropout_rate
		self.FFNs_start = nn.ModuleList()
		self.SFUs_start = nn.ModuleList()
		self.FFNs_end = nn.ModuleList()
		self.SFUs_end = nn.ModuleList()
		for i in range(self.hop):
			self.FFNs_start.append(FeedForwardNetwork(x_size + y_size + 2 * hidden_size, hidden_size, 1, dropout_rate))
			self.SFUs_start.append(SFU(y_size, 2 * hidden_size))
			self.FFNs_end.append(FeedForwardNetwork(x_size + y_size + 2 * hidden_size, hidden_size, 1, dropout_rate))
			self.SFUs_end.append(SFU(y_size, 2 * hidden_size))

	def forward(self, x, y, x_mask, y_mask):
		z_s = y[:, -1, :].unsqueeze(1)  # [B, 1, I]
		z_e = None
		s = None
		e = None
		p_s = None
		p_e = None

		for i in range(self.hop):
			z_s_ = z_s.repeat(1, x.size(1), 1)  # [B, S, I]
			s = self.FFNs_start[i](torch.cat([x, z_s_, x * z_s_], 2)).squeeze(2)
			s.data.masked_fill_(x_mask.data, -float('inf'))
			p_s = F.softmax(s, dim=1)  # [B, S]
			u_s = p_s.unsqueeze(1).bmm(x)  # [B, 1, I]
			z_e = self.SFUs_start[i](z_s, u_s)  # [B, 1, I]
			z_e_ = z_e.repeat(1, x.size(1), 1)  # [B, S, I]
			e = self.FFNs_end[i](torch.cat([x, z_e_, x * z_e_], 2)).squeeze(2)
			e.data.masked_fill_(x_mask.data, -float('inf'))
			p_e = F.softmax(e, dim=1)  # [B, S]
			u_e = p_e.unsqueeze(1).bmm(x)  # [B, 1, I]
			z_s = self.SFUs_end[i](z_e, u_e)
		if self.normalize:
			#  0-1 probabilities
			p_s = F.softmax(s, dim=1)  # [B, S]
			p_e = F.softmax(e, dim=1)  # [B, S]
		else:
			p_s = s.exp()
			p_e = e.exp()

		return p_s, p_e
