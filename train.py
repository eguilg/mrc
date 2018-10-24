import multiprocessing as mp
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import config
from dataset import (MaiDirDataSource, MaiDirDataset, Vocab,
					 MaiIndexTransform, MethodBasedBatchSampler)
from models.losses import PointerLoss, RougeLoss
from models.rc_model import RCModel
from utils.osutils import *
from metrics import RougeL

import postprocess

bidaf1 = config.bi_daf_1()
bidaf2 = config.bi_daf_2()
bidaf3 = config.bi_daf_3()

rnet1 = config.r_net_1()
rnet2 = config.r_net_2()
rnet3 = config.r_net_3()

mreader1 = config.m_reader_1()
mreader2 = config.m_reader_2()
mreader3 = config.m_reader_3()

slqa1 = config.slqa_1()
slqa2 = config.slqa_2()
slqa3 = config.slqa_3()

slqa_plus1 = config.slqa_plus_1()
slqa_plus2 = config.slqa_plus_2()
slqa_plus3 = config.slqa_plus_3()


# cur_cfg = bidaf2
# cur_cfg = bidaf3

# cur_cfg = rnet2
# cur_cfg = rnet3

# cur_cfg = mreader2
cur_cfg = mreader3

# cur_cfg = slqa1
# cur_cfg = slqa2
# cur_cfg = slqa3

# cur_cfg = slqa_plus1
# cur_cfg = slqa_plus2
# cur_cfg = slqa_plus3

SEED = 502
EPOCH = 15
use_mrt = False
switch = False
use_data1 = False
if __name__ == '__main__':
	print(cur_cfg.model_params)
	model_dir = './data/models/'
	mkdir_if_missing(model_dir)

	if use_mrt:
		model_name = cur_cfg.name + '_mrt'
	else:
		model_name = cur_cfg.name
	if switch:
		model_name += '_switch'
	if use_data1:
		model_name += '_full_data'
	model_path = os.path.join(model_dir, model_name + '.state')

	jieba_base_v = Vocab('./data/embed/base_token_vocab_jieba.pkl',
						 './data/embed/base_token_embed_jieba.pkl')
	jieba_sgns_v = Vocab('./data/embed/train_sgns_vocab_jieba.pkl',
						 './data/embed/train_sgns_embed_jieba.pkl')
	jieba_flag_v = Vocab('./data/embed/base_flag_vocab_jieba.pkl',
						 './data/embed/base_flag_embed_jieba.pkl')

	if switch:
		pyltp_base_v = Vocab('./data/embed/base_token_vocab_pyltp.pkl',
							 './data/embed/base_token_embed_pyltp.pkl')
		pyltp_sgns_v = Vocab('./data/embed/train_sgns_vocab_pyltp.pkl',
							 './data/embed/train_sgns_embed_pyltp.pkl')
		pyltp_flag_v = Vocab('./data/embed/base_flag_vocab_pyltp.pkl',
							 './data/embed/base_flag_embed_pyltp.pkl')

		transform = MaiIndexTransform(jieba_base_v, jieba_sgns_v, jieba_flag_v, pyltp_base_v, pyltp_sgns_v,
									  pyltp_flag_v)
		trainset1_roots = [
			'./data/train/gen/train_1/samples_jieba500',
			'./data/train/gen/train_1/samples_pyltp500',
		]
		trainset2_roots = [
			'./data/train/gen/train_2/samples_jieba500',
			'./data/train/gen/train_2/samples_pyltp500',
		]

		embed_lists = {
			'jieba': [jieba_base_v.embeddings, jieba_sgns_v.embeddings, jieba_flag_v.embeddings],
			'pyltp': [pyltp_base_v.embeddings, pyltp_sgns_v.embeddings, pyltp_flag_v.embeddings]
		}
	else:
		transform = MaiIndexTransform(jieba_base_v, jieba_sgns_v, jieba_flag_v)
		trainset1_roots = [
			'./data/train/gen/train_1/samples_jieba500'
		]
		trainset2_roots = [
			'./data/train/gen/train_2/samples_jieba500'
		]

		embed_lists = {
			'jieba': [jieba_base_v.embeddings, jieba_sgns_v.embeddings, jieba_flag_v.embeddings],
			'pyltp': []
		}

	train_data1_source = MaiDirDataSource(trainset1_roots)
	train_data2_source = MaiDirDataSource(trainset2_roots)
	train_data2_source.split(dev_split=0.1, seed=SEED)

	data_for_train = train_data2_source.train
	data_for_dev = train_data2_source.dev

	if use_data1:
		data_for_train += train_data1_source.data
		np.random.seed(SEED)
		np.random.shuffle(data_for_train)

	train_loader = DataLoader(
		dataset=MaiDirDataset(data_for_train, transform),
		batch_sampler=MethodBasedBatchSampler(data_for_train, batch_size=32, seed=SEED),
		num_workers=mp.cpu_count(),
		collate_fn=transform.batchify
	)

	dev_loader = DataLoader(
		dataset=MaiDirDataset(data_for_dev, transform),
		batch_sampler=MethodBasedBatchSampler(data_for_dev, batch_size=32, shuffle=False),
		num_workers=mp.cpu_count(),
		collate_fn=transform.batchify
	)

	model_params = cur_cfg.model_params

	model = RCModel(model_params, embed_lists, normalize=(not use_mrt))
	model = model.cuda()
	if use_mrt:
		criterion_main = RougeLoss().cuda()
	else:
		criterion_main = PointerLoss().cuda()
	criterion_extra = nn.MultiLabelSoftMarginLoss().cuda()

	param_to_update = list(filter(lambda p: p.requires_grad, model.parameters()))
	model_param_num = 0
	for param in list(param_to_update):
		model_param_num += param.nelement()
	print('num_params_except_embedding:%d' % (model_param_num))
	# Optimizer
	optimizer = torch.optim.Adam(param_to_update, lr=4e-4)

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
	last_val_step = global_step
	val_every = [1000, 700, 500, 350]
	drop_lr_frq = [1, 2, 3, 5]
	# val_every_min = 350
	# val_every = 1000
	val_no_improve = 0
	ptr_loss_print = 0
	qtype_loss_print = 0
	c_in_a_loss_print = 0
	q_in_a_loss_print = 0
	ans_len_loss_print = 0

	if state is not None:
		if state['best_score'] > 0.90:
			grade = 3
		elif state['best_score'] > 0.89:
			grade = 2
		elif state['best_score'] > 0.88:
			grade = 1
		else:
			grade = 0

	for e in epoch_list:
		step = 0
		with tqdm(total=len(train_loader)) as bar:
			for i, batch in enumerate(train_loader):
				inputs, targets = transform.prepare_inputs(batch, use_mrt)

				model.train()
				optimizer.zero_grad()
				s_scores, e_scores, extra_outputs = model(*inputs)
				starts, ends, extra_targets = targets

				q_type_gt, c_in_a_gt, q_in_a_gt, ans_len_gt, delta_rouge = extra_targets
				q_type_pred, c_in_a_pred, q_in_a_pred, ans_len_logits = extra_outputs
				if isinstance(criterion_main, PointerLoss):
					loss_main = criterion_main(s_scores, e_scores, starts, ends)
				elif isinstance(criterion_main, RougeLoss) and delta_rouge is not None:
					loss_main = criterion_main(s_scores, e_scores, delta_rouge)
				else:
					raise NotImplementedError

				loss_qtype = criterion_extra(q_type_pred, q_type_gt)
				loss_c_in_a = criterion_extra(c_in_a_pred, c_in_a_gt)
				loss_q_in_a = criterion_extra(q_in_a_pred, q_in_a_gt)
				loss_ans_len = F.nll_loss(F.log_softmax(ans_len_logits, dim=-1), ans_len_gt)

				train_loss = loss_main + 0.2 * (loss_qtype + loss_c_in_a + loss_q_in_a + loss_ans_len)
				train_loss.backward()

				# clip_grad_norm_(param_to_update, 5)
				optimizer.step()
				ptr_loss_print += loss_main.item()
				qtype_loss_print += loss_qtype.item()
				c_in_a_loss_print += loss_c_in_a.item()
				q_in_a_loss_print += loss_q_in_a.item()
				ans_len_loss_print += loss_ans_len.item()

				step += 1
				global_step += 1

				if global_step % print_every == 0:
					bar.update(min(print_every, step))
					time.sleep(0.02)
					print('Epoch: [{}][{}/{}]\t'
						  'Loss: Main {:.4f}\t'
						  'Qtype {:.4f}\t'
						  'CinA {:.4f}\t'
						  'QinA {:.4f}\t'
						  'AnsLen {:.4f}'
						  .format(e, step, len(train_loader),
								  ptr_loss_print / print_every,
								  qtype_loss_print / print_every,
								  c_in_a_loss_print / print_every,
								  q_in_a_loss_print / print_every,
								  ans_len_loss_print / print_every))

					ptr_loss_print = 0
					qtype_loss_print = 0
					c_in_a_loss_print = 0
					q_in_a_loss_print = 0
					ans_len_loss_print = 0

				if global_step - last_val_step == val_every[grade]:
					print('-' * 80)
					print('Evaluating...')
					last_val_step = global_step
					val_loss_total = 0
					val_step = 0
					val_sample_num = 0
					val_ans_len_hit = 0
					val_start_hit = 0
					val_end_hit = 0
					rl = RougeL()
					with torch.no_grad():
						model.eval()
						for val_batch in dev_loader:
							# cut, cuda
							inputs, targets = transform.prepare_inputs(val_batch, use_mrt)
							starts, ends, extra_targets = targets
							_, _, _, ans_len_gt, delta_rouge = extra_targets

							s_scores, e_scores, ans_len_logits = model(*inputs)
							ans_len_prob = F.softmax(ans_len_logits, dim=-1)

							ans_len_hit = (torch.max(ans_len_prob, -1)[1] == ans_len_gt).sum().item()
							start_hit = (torch.max(s_scores, -1)[1] == starts).sum().item()
							end_hit = (torch.max(e_scores, -1)[1] == ends).sum().item()

							if isinstance(criterion_main, PointerLoss):
								val_loss = criterion_main(s_scores, e_scores, starts, ends)
							elif isinstance(criterion_main, RougeLoss) and delta_rouge is not None:
								val_loss = criterion_main(s_scores, e_scores, delta_rouge)
							else:
								raise NotImplementedError

							s_scores = s_scores.detach().cpu()
							e_scores = e_scores.detach().cpu()
							batch_pos1, batch_pos2, confidence = model.decode(s_scores, e_scores)
							for pos1, pos2, sample in zip(batch_pos1, batch_pos2, val_batch['raw']):
								gt_ans = sample['answer']
								pred_ans = postprocess.gen_ans(pos1, pos2, sample)
								rl.add_inst(pred_ans, gt_ans)

							val_loss_total += val_loss.item()
							val_ans_len_hit += ans_len_hit
							val_start_hit += start_hit
							val_end_hit += end_hit
							val_sample_num += ans_len_logits.size(0)
							val_step += 1

					rouge_score = rl.get_score()
					print('Val Epoch: [{}][{}/{}]\t'
						  'Loss: Main {:.4f}\t'
						  'Acc: Start {:.4f}\t'
						  'End {:.4f}\t'
						  'AnsLen {:.4f}\t'
						  .format(e, step, len(train_loader),
								  val_loss_total / val_step,
								  val_start_hit / val_sample_num,
								  val_end_hit / val_sample_num,
								  val_ans_len_hit / val_sample_num))
					print('RougeL: {: .4f}'.format(rouge_score))

					print('-' * 80)
					if os.path.isfile(model_path):
						state = torch.load(model_path)
					else:
						state = {}

					if state == {} or state['best_score'] < rouge_score:
						state['best_model_state'] = model.state_dict()
						state['best_opt_state'] = optimizer.state_dict()
						state['best_loss'] = val_loss_total / val_step
						state['best_score'] = rouge_score
						state['best_epoch'] = e
						state['best_step'] = global_step

						if state['best_score'] > 0.90:
							grade = 3
						elif state['best_score'] > 0.89:
							grade = 2
						elif state['best_score'] > 0.88:
							grade = 1
						else:
							grade = 0

						val_no_improve = 0
					else:
						val_no_improve += 1

						if val_no_improve >= drop_lr_frq[grade]:
							print('dropping lr...')
							val_no_improve = 0
							lr_total = 0
							lr_num = 0
							for param_group in optimizer.param_groups:
								if param_group['lr'] > 5e-5:
									param_group['lr'] *= 0.5
								lr_total += param_group['lr']
								lr_num += 1
							print('curr avg lr is {}'.format(lr_total / lr_num))

					state['cur_model_state'] = model.state_dict()
					state['cur_opt_state'] = optimizer.state_dict()
					state['cur_epoch'] = e
					state['val_loss'] = val_loss_total / val_step
					state['val_score'] = rouge_score
					state['cur_step'] = global_step


					torch.save(state, model_path)
