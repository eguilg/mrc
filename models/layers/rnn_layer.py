import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BiRNN(nn.Module):

	def __init__(self, input_size, hidden_size, num_layers, dropout=0, batch_norm=True, rnn_type=nn.LSTM):
		super(BiRNN, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.dropout = dropout
		self.num_layers = num_layers
		self.batch_norm = batch_norm

		self.rnn = rnn_type(
			input_size=input_size,
			hidden_size=hidden_size,
			num_layers=num_layers,
			bidirectional=True,
			dropout=dropout if num_layers > 1 else 0
		)

		if batch_norm:
			self.layer_norm = nn.LayerNorm(input_size)

		self.layer_dropout = nn.Dropout(p=dropout)
		self.init_params()

	def init_params(self):
		""" init layer params """
		for name, param in self.named_parameters():
			if 'weight_ih' in name:
				torch.nn.init.xavier_uniform_(param)
			elif 'weight_hh' in name:
				torch.nn.init.orthogonal_(param)
			elif 'bias' in name:
				torch.nn.init.constant_(param, 0)

	def forward(self, x, x_mask):
		"""

		:param x: (batch_size, seq_len, input_size)
		:param x_mask: (batch_size, seq_len)
		:return: outputs((batch_size, seq_len, hidden_size*2)
		"""
		batch_size, seq_len, input_size = x.shape

		#  batch norm
		if self.batch_norm:
			x = x.contiguous().view(-1, input_size)
			x = self.layer_norm(x)
			x = x.view(batch_size, seq_len, input_size)

		#  dropout input
		x = self.layer_dropout(x)

		#  prepare rnn input
		lengths = x_mask.data.eq(0).long().sum(1).squeeze()
		# idx_org = torch.range(0, batch_size - 1, dtype=torch.int64, device=x.device)
		lengths, idx_sorted = torch.sort(lengths, dim=0, descending=True)
		_, idx_org = torch.sort(idx_sorted, dim=0)

		# Sort x
		x = x.index_select(0, idx_sorted)

		# Pack it up
		rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)

		#  run rnn
		outputs, _ = self.rnn(rnn_input)  # (seq_len, batch_size, hidden_size*2)

		#  trans back output
		outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

		#  (batch_size, seq_len, hidden_size*2)
		outputs = outputs.index_select(0, idx_org)

		max_len = lengths[0].item()
		if max_len < seq_len:
			pad_len = seq_len - max_len
			hidden_size = self.hidden_size * 2
			padding = outputs.new_zeros(batch_size, pad_len, hidden_size, device=outputs.device)
			outputs = torch.cat([outputs, padding], dim=1)  # cat on seq_len dim
			del padding

		# Dropout on output layer
		outputs = self.layer_dropout(outputs)

		return outputs


class StackedBRNN(nn.Module):
	"""Stacked Bi-directional RNNs.

	Differs from standard PyTorch library in that it has the option to save
	and concat the hidden states between layers. (i.e. the output hidden size
	for each sequence input is num_layers * hidden_size).
	"""

	def __init__(self, input_size, hidden_size, num_layers,
				 dropout=0, dropout_output=False, rnn_type=nn.LSTM,
				 concat_layers=False, padding=False):
		super(StackedBRNN, self).__init__()
		self.padding = padding
		self.dropout_output = dropout_output
		self.dropout_rate = dropout
		self.num_layers = num_layers
		self.concat_layers = concat_layers
		self.rnns = nn.ModuleList()
		for i in range(num_layers):
			input_size = input_size if i == 0 else 2 * hidden_size
			self.rnns.append(rnn_type(input_size, hidden_size,
									  num_layers=1,
									  bidirectional=True))

	def forward(self, x, x_mask):
		"""Encode either padded or non-padded sequences.

		Can choose to either handle or ignore variable length sequences.
		Always handle padding in eval.

		Args:
			x: batch * len * hdim
			x_mask: batch * len (1 for padding, 0 for true)
		Output:
			x_encoded: batch * len * hdim_encoded
		"""

		# Pad if we care or if its during eval.
		output = self._forward_padded(x, x_mask)

		return output.contiguous()

	def _forward_unpadded(self, x, x_mask):
		"""Faster encoding that ignores any padding."""
		# Transpose batch and sequence dims
		x = x.transpose(0, 1)

		# Encode all layers
		outputs = [x]
		for i in range(self.num_layers):
			rnn_input = outputs[-1]

			# Apply dropout to hidden input
			if self.dropout_rate > 0:
				rnn_input = F.dropout(rnn_input,
									  p=self.dropout_rate,
									  training=self.training)
			# Forward
			rnn_output = self.rnns[i](rnn_input)[0]
			outputs.append(rnn_output)

		# Concat hidden layers
		if self.concat_layers:
			output = torch.cat(outputs[1:], 2)
		else:
			output = outputs[-1]

		# Transpose back
		output = output.transpose(0, 1)

		# Dropout on output layer
		if self.dropout_output and self.dropout_rate > 0:
			output = F.dropout(output,
							   p=self.dropout_rate,
							   training=self.training)
		return output

	def _forward_padded(self, x, x_mask):
		"""Slower (significantly), but more precise, encoding that handles
		padding.
		"""
		# Compute sorted sequence lengths
		lengths = x_mask.data.eq(0).long().sum(1).squeeze()
		_, idx_sort = torch.sort(lengths, dim=0, descending=True)
		_, idx_unsort = torch.sort(idx_sort, dim=0)

		lengths = list(lengths[idx_sort])
		idx_sort = Variable(idx_sort)
		idx_unsort = Variable(idx_unsort)

		# Sort x
		x = x.index_select(0, idx_sort)

		# Transpose batch and sequence dims
		x = x.transpose(0, 1)

		# Pack it up
		rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)

		# Encode all layers
		outputs = [rnn_input]
		for i in range(self.num_layers):
			rnn_input = outputs[-1]

			# Apply dropout to input
			if self.dropout_rate > 0:
				dropout_input = F.dropout(rnn_input.data,
										  p=self.dropout_rate,
										  training=self.training)
				rnn_input = nn.utils.rnn.PackedSequence(dropout_input,
														rnn_input.batch_sizes)
			outputs.append(self.rnns[i](rnn_input)[0])

		# Unpack everything
		for i, o in enumerate(outputs[1:], 1):
			outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]

		# Concat hidden layers or take final
		if self.concat_layers:
			output = torch.cat(outputs[1:], 2)
		else:
			output = outputs[-1]

		# Transpose and unsort
		output = output.transpose(0, 1)
		output = output.index_select(0, idx_unsort)

		# Pad up to original batch sequence length
		if output.size(1) != x_mask.size(1):
			padding = torch.zeros(output.size(0),
								  x_mask.size(1) - output.size(1),
								  output.size(2)).type(output.data.type())
			output = torch.cat([output, Variable(padding)], 1)

		# Dropout on output layer
		if self.dropout_output and self.dropout_rate > 0:
			output = F.dropout(output,
							   p=self.dropout_rate,
							   training=self.training)
		return output
