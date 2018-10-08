import multiprocessing as mp
import time

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import config
from dataset import (MaiDirDataSource, MaiDirDataset, Vocab,
					 MaiIndexTransform, MethodBasedBatchSampler)
from models.losses import PointerLoss
from models.rc_model import RCModel
from utils.osutils import *

rnet1 = config.r_net_1()
rnet2 = config.r_net_2()
mreader1 = config.m_reader_1()
mreader2 = config.m_reader_2()

if __name__ == '__main__':
	cur_cfg = mreader2
	model_dir = './data/models/'
	mkdir_if_missing(model_dir)
	model_path = os.path.join(model_dir, cur_cfg.name + '.state')

	trainset_roots = [
		'./data/train/gen/train_1/samples_jieba500',
		'./data/train/gen/train_1/samples_pyltp500',
	]
	SEED = 502
	EPOCH = 15

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

	model_params = cur_cfg.model_params

	model = RCModel(model_params, embed_lists)
	model = model.cuda()
	criterion = PointerLoss().cuda()

	param_to_update = list(filter(lambda p: p.requires_grad, model.parameters()))
	model_param_num = 0
	for param in list(param_to_update):
		model_param_num += param.nelement()
	print('num_params_except_embedding:%d' % (model_param_num))
	# Optimizer
	optimizer = torch.optim.Adadelta(param_to_update, lr=0.5)

	# # Trainer
	# trainer = Trainer(model, criterion)
	#
	# trainer.train(1, train_loader, optimizer)

	if os.path.isfile(model_path):
		print('load training param, ', model_path)
		state = torch.load(model_path)
		model.load_state_dict(state['cur_model_state'])
		optimizer.load_state_dict(state['cur_opt_state'])
		epoch_list = range(state['cur_epoch'] + 1, state['cur_epoch'] + 1 + EPOCH)
		train_loss_global = state['train_loss']
		global_step = state['cur_step']
	else:
		state = None
		epoch_list = range(EPOCH)
		grade = 1
		train_loss_global = 0

		global_step = 0

	grade = 0
	print_every = 50
	val_every = [1000, 500, 300]
	train_loss_print = 0

	if state is not None:
		if state['best_loss'] < 1.4:
			grade = 2
		elif state['best_loss'] < 1.5:
			grade = 1
		else:
			grade = 0

	for e in epoch_list:
		step = 0
		with tqdm(total=len(train_loader)) as bar:
			for i, batch in enumerate(train_loader):
				inputs, targets = transform.prepare_inputs(batch)

				model.train()
				optimizer.zero_grad()
				s_scores, e_scores = model(*inputs)
				loss_value = criterion(s_scores, e_scores, *targets)
				loss_value.backward()

				clip_grad_norm_(model.parameters(), 5)
				optimizer.step()

				train_loss_global += loss_value.item()
				train_loss_print += loss_value.item()

				step += 1
				global_step += 1

				if global_step % print_every == 0:
					bar.update(min(print_every, step))
					time.sleep(0.01)
					print('Epoch: [{}][{}/{}]\t'
						  'Train Loss {:.3f} ({:.3f})'
						  .format(e, step, len(train_loader),
								  train_loss_print / print_every, train_loss_global / global_step))

					train_loss_print = 0

				if global_step % val_every[grade] == 0:
					print('-'*80)
					print('Evaluating...')
					val_loss = 0
					val_step = 0
					with torch.no_grad():
						model.eval()
						for val_batch in dev_loader:
							# cut, cuda
							inputs, targets = transform.prepare_inputs(val_batch)
							s_scores, e_scores = model(*inputs)
							loss_value = criterion(s_scores, e_scores, *targets)

							val_loss += loss_value.item()
							val_step += 1

					print('Epoch: [{}][{}/{}]\t'
						  'Val Loss {:.3f}'
						  .format(e, step, len(train_loader),
								  val_loss / val_step))
					print('-' * 80)
					if os.path.isfile(model_path):
						state = torch.load(model_path)
					else:
						state = {}

					if state == {} or state['best_loss'] > (val_loss / val_step):
						state['best_model_state'] = model.state_dict()
						state['best_opt_state'] = optimizer.state_dict()
						state['best_loss'] = val_loss / val_step
						state['best_epoch'] = e
						state['best_step'] = global_step

						if state['best_loss'] < 1.4:
							grade = 2
						elif state['best_loss'] < 1.5:
							grade = 1
						else:
							grade = 0

					state['cur_model_state'] = model.state_dict()
					state['cur_opt_state'] = optimizer.state_dict()
					state['cur_epoch'] = e
					state['val_loss'] = val_loss / val_step
					state['cur_step'] = global_step
					state['train_loss'] = train_loss_global

					torch.save(state, model_path)
