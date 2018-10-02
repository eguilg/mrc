from __future__ import print_function, absolute_import

import time

from losses import PointerLoss
from utils.meters import AverageMeter


class BaseTrainer(object):
	def __init__(self, model, criterion):
		super(BaseTrainer, self).__init__()
		self.model = model
		self.criterion = criterion

	def train(self, epoch, data_loader, optimizer, print_freq=1):
		self.model.train()

		batch_time = AverageMeter()
		data_time = AverageMeter()
		losses = AverageMeter()
		precisions = AverageMeter()

		end = time.time()
		data_loader = iter(data_loader)
		for i in range(len(data_loader)):
			batch = next(data_loader)
			data_time.update(time.time() - end)

			inputs, targets = self._parse_data(batch)
			loss, prec = self._forward(inputs, targets)

			losses.update(loss.data[0], targets[0].size(0))
			precisions.update(prec, targets[0].size(0))

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			batch_time.update(time.time() - end)
			end = time.time()

			if (i + 1) % print_freq == 0:
				print('Epoch: [{}][{}/{}]\t'
					  'Time {:.3f} ({:.3f})\t'
					  'Data {:.3f} ({:.3f})\t'
					  'Loss {:.3f} ({:.3f})\t'
					  'Prec {:.2%} ({:.2%})\t'
					  .format(epoch, i + 1, len(data_loader),
							  batch_time.val, batch_time.avg,
							  data_time.val, data_time.avg,
							  losses.val, losses.avg,
							  precisions.val, precisions.avg))

	def _parse_data(self, batch):
		raise NotImplementedError

	def _forward(self, inputs, targets):
		raise NotImplementedError


class Trainer(BaseTrainer):
	def _parse_data(self, batch):

		x1_keys = [
			'c_base_idx',
			'c_sgns_idx',
			'c_flag_idx'
		]
		x1_f_keys = [
			'c_in_q'
		]
		x1_list = [batch[key].cuda() for key in x1_keys]
		x1_f_list = [batch[key].cuda() for key in x1_f_keys]
		x1_mask = batch['c_mask'].cuda()

		x2_keys = [
			'q_base_idx',
			'q_sgns_idx',
			'q_flag_idx'
		]
		x2_f_keys = [
			'q_in_c'
		]
		x2_list = [batch[key].cuda() for key in x2_keys]
		x2_f_list = [batch[key].cuda() for key in x2_f_keys]
		x2_mask = batch['q_mask'].cuda()

		method = batch['method']

		inputs = [x1_list, x1_f_list, x1_mask, x2_list, x2_f_list, x2_mask, method]
		if 'start' in batch:
			y_start = batch['start'].cuda()
			y_end = batch['end'].cuda()
			targets = [y_start, y_end]
			return inputs, targets
		else:
			return inputs, None

	def _forward(self, inputs, targets):

		s_p, e_p = self.model(*inputs)
		if isinstance(self.criterion, PointerLoss):
			loss = self.criterion(s_p, e_p, *targets)
		else:
			raise ValueError("Unsupported loss:", self.criterion)
		prec = 0
		return loss, prec
