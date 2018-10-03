import multiprocessing as mp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import (MaiDirDataSource, MaiDirDataset, Vocab,
					 MaiIndexTransform, MethodBasedBatchSampler)
from models import R_Net
from models.losses import PointerLoss
from trainers import Trainer

RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}
CELL_TYPES = {'lstm': nn.LSTMCell, 'gru': nn.GRUCell, 'rnn': nn.RNNCell}

trainset_roots = [
	'./data/train/gen/train_1/samples_jieba500',
	'./data/train/gen/train_1/samples_pyltp500',
]
SEED = 502

jieba_base_v = Vocab('./data/embed/base_token_vocab_jieba.pkl',
					 './data/embed/base_token_embed_jieba.pkl')
jieba_sgns_v = Vocab('./data/embed/train_sgns_vocab_jieba.pkl',
					 './data/embed/train_sgns_embed_jieba.pkl')
jieba_flag_v = Vocab('./data/embed/base_flag_vocab_jieba.pkl',
					 './data/embed/base_flag_embed_jieba.pkl')

pyltp_base_v = Vocab('./data/embed/base_token_vocab_pyltp.pkl',
					 './data/embed/base_token_embed_pyltp.pkl')
pyltp_sgns_v = Vocab('./data/embed/train_sgns_vocab_pyltp.pkl',
					 './data/embed/train_sgns_embed_pyltp.pkl')
pyltp_flag_v = Vocab('./data/embed/base_flag_vocab_pyltp.pkl',
					 './data/embed/base_flag_embed_pyltp.pkl')


transform = MaiIndexTransform(jieba_base_v, jieba_sgns_v, jieba_flag_v, pyltp_base_v, pyltp_sgns_v, pyltp_flag_v)
train_data_source = MaiDirDataSource(trainset_roots)
train_data_source.split(dev_split=0.1, seed=SEED)

train_loader = DataLoader(
	dataset=MaiDirDataset(train_data_source.train, transform),
	batch_sampler=MethodBasedBatchSampler(train_data_source.train, batch_size=32, seed=SEED),
	num_workers=mp.cpu_count(),
	collate_fn=transform.batchify
)

dev_loader = DataLoader(
	dataset=MaiDirDataset(train_data_source.dev, transform),
	batch_sampler=MethodBasedBatchSampler(train_data_source.dev, batch_size=32, shuffle=False),
	num_workers=mp.cpu_count(),
	collate_fn=transform.batchify
)

embed_lists = {
	'jieba': [jieba_base_v.embeddings, jieba_sgns_v.embeddings, jieba_flag_v.embeddings],
	'pyltp': [pyltp_base_v.embeddings, pyltp_sgns_v.embeddings, pyltp_flag_v.embeddings]
}

model_params = {
	'embed_lists': embed_lists,
	'num_features': 1,
	'dropout': 0.2,
	'hidden_size': 150,
	'rnn_type': nn.LSTM,
	'hop': 3
}

model = R_Net(model_params)
model = model.cuda()
criterion = PointerLoss().cuda()

param_to_update = filter(lambda p: p.requires_grad, model.parameters())
# Optimizer
optimizer = torch.optim.Adadelta(param_to_update, lr=0.5)

# Trainer
trainer = Trainer(model, criterion)

trainer.train(1, train_loader, optimizer)