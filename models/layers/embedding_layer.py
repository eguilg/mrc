import torch
import torch.nn as nn


class MergedEmbedding(nn.Module):
	def __init__(self, embed_list):
		super(MergedEmbedding, self).__init__()
		self.embedding = nn.ModuleList()
		self.output_dim = 0
		for i, embed in enumerate(embed_list):
			self.embedding.append(nn.Embedding(embed.shape[0],
											   embed.shape[1],
											   padding_idx=0,
											   _weight=torch.Tensor(embed)))
			self.output_dim += self.embedding[-1].embedding_dim

		for p in self.parameters():
			p.requires_grad = False

	def forward(self, x_list):
		outputs = []
		for x, embed in zip(x_list, self.embedding):
			outputs.append(embed(x))
		return torch.cat(outputs, -1)
