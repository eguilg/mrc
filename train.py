import multiprocessing as mp
import time

import torch
import torch.nn as nn
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
rnet3 = config.r_net_3()
mreader1 = config.m_reader_1()
mreader2 = config.m_reader_2()
mreader3 = config.m_reader_3()
bidaf1 = config.bi_daf_1()
bidaf2 = config.bi_daf_2()
bidaf3 = config.bi_daf_3()
slqa1 = config.slqa_1()
slqa2 = config.slqa_2()
slqa3 = config.slqa_3()

# cur_cfg = rnet1
# cur_cfg = rnet2
# cur_cfg = rnet3
# cur_cfg = mreader1
# cur_cfg = mreader2
# cur_cfg = mreader3
# cur_cfg = bidaf1
# cur_cfg = bidaf2
# cur_cfg = bidaf3
# cur_cfg = slqa1
cur_cfg = slqa2
# cur_cfg = slqa3

jieba_only = False

if __name__ == '__main__':
	print(cur_cfg.model_params)
	model_dir = './data/models/'
	mkdir_if_missing(model_dir)
	model_path = os.path.join(model_dir, cur_cfg.name + '.state')

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
	if jieba_only:
		transform = MaiIndexTransform(jieba_base_v, jieba_sgns_v,
									  jieba_flag_v)  # , pyltp_base_v, pyltp_sgns_v, pyltp_flag_v)
		trainset_roots = [
			'./data/train/gen/train_1/samples_jieba500',
			# './data/train/gen/train_1/samples_pyltp500',
		]
	else:
		transform = MaiIndexTransform(jieba_base_v, jieba_sgns_v, jieba_flag_v, pyltp_base_v, pyltp_sgns_v,
									  pyltp_flag_v)
		trainset_roots = [
			'./data/train/gen/train_1/samples_jieba500',
			'./data/train/gen/train_1/samples_pyltp500',
		]

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
		# 'pyltp': []
	}

	model_params = cur_cfg.model_params

	model = RCModel(model_params, embed_lists)
	model = model.cuda()
	criterion_ptr = PointerLoss().cuda()
	criterion_extra = nn.MultiLabelSoftMarginLoss().cuda()

	param_to_update = list(filter(lambda p: p.requires_grad, model.parameters()))
	model_param_num = 0
	for param in list(param_to_update):
		model_param_num += param.nelement()
	print('num_params_except_embedding:%d' % (model_param_num))
	# Optimizer
	optimizer = torch.optim.Adam(param_to_update, lr=1e-4)

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
		global_step = state['cur_step']
	else:
		state = None
		epoch_list = range(EPOCH)
		grade = 1

		global_step = 0

	grade = 0
	print_every = 50
	val_every = [1000, 600, 300]
	ptr_loss_print = 0
	qtype_loss_print = 0
	c_in_a_loss_print = 0
	q_in_a_loss_print = 0

	if state is not None:
		if state['best_loss'] < 1.20:
			grade = 2
		elif state['best_loss'] < 1.30:
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
				s_scores, e_scores, extra_outputs = model(*inputs)
				starts, ends, extra_targets = targets

				q_type_gt, c_in_a_gt, q_in_a_gt = extra_targets
				q_type_pred, c_in_a_pred, q_in_a_pred = extra_outputs

				loss_ptr = criterion_ptr(s_scores, e_scores, starts, ends)
				loss_qtype = criterion_extra(q_type_pred, q_type_gt)
				loss_c_in_a = criterion_extra(c_in_a_pred, c_in_a_gt)
				loss_q_in_a = criterion_extra(q_in_a_pred, q_in_a_gt)

				train_loss = loss_ptr + 0.2 * (loss_qtype + loss_c_in_a + loss_q_in_a)
				train_loss.backward()

				# clip_grad_norm_(param_to_update, 5)
				optimizer.step()
				ptr_loss_print += loss_ptr.item()
				qtype_loss_print += loss_qtype.item()
				c_in_a_loss_print += loss_c_in_a.item()
				q_in_a_loss_print += loss_q_in_a.item()

				step += 1
				global_step += 1

				if global_step % print_every == 0:
					bar.update(min(print_every, step))
					time.sleep(0.01)
					print('Epoch: [{}][{}/{}]\t'
						  'Loss: Ptr {:.4f}\t'
						  'Qtype {:.4f}\t'
						  'CinA {:.4f}\t'
						  'QinA {:.4f}'
						  .format(e, step, len(train_loader),
								  ptr_loss_print / print_every,
								  qtype_loss_print / print_every,
								  c_in_a_loss_print / print_every,
								  q_in_a_loss_print / print_every))

					ptr_loss_print = 0
					qtype_loss_print = 0
					c_in_a_loss_print = 0
					q_in_a_loss_print = 0

				if global_step % val_every[grade] == 0:
					print('-' * 80)
					print('Evaluating...')
					val_loss_total = 0
					val_step = 0
					with torch.no_grad():
						model.eval()
						for val_batch in dev_loader:
							# cut, cuda
							inputs, targets = transform.prepare_inputs(val_batch)
							starts, ends, extra_targets = targets
							s_scores, e_scores, _ = model(*inputs)
							val_loss = criterion_ptr(s_scores, e_scores, starts, ends)

							val_loss_total += val_loss.item()
							val_step += 1

					print('Epoch: [{}][{}/{}]\t'
						  'Val Ptr Loss {:.4f}'
						  .format(e, step, len(train_loader),
								  val_loss_total / val_step))
					print('-' * 80)
					if os.path.isfile(model_path):
						state = torch.load(model_path)
					else:
						state = {}

					if state == {} or state['best_loss'] > (val_loss_total / val_step):
						state['best_model_state'] = model.state_dict()
						state['best_opt_state'] = optimizer.state_dict()
						state['best_loss'] = val_loss_total / val_step
						state['best_epoch'] = e
						state['best_step'] = global_step

						if state['best_loss'] < 1.20:
							grade = 2
						elif state['best_loss'] < 1.30:
							grade = 1
						else:
							grade = 0

					state['cur_model_state'] = model.state_dict()
					state['cur_opt_state'] = optimizer.state_dict()
					state['cur_epoch'] = e
					state['val_loss'] = val_loss_total / val_step
					state['cur_step'] = global_step

					torch.save(state, model_path)
