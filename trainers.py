from __future__ import print_function, absolute_import

import time

from dataset.transform import MaiIndexTransform
from models.losses import PointerLoss
from utils.meters import AverageMeter


class BaseTrainer(object):
	def __init__(self, model, criterion):
		super(BaseTrainer, self).__init__()
		self.model = model
		self.criterion = criterion

	def train(self, epoch, train_loader, optimizer, dev_loader=None, print_freq=1):
		self.model.train()

		batch_time = AverageMeter()
		data_time = AverageMeter()
		losses = AverageMeter()
		precisions = AverageMeter()

		end = time.time()
		# data_loader = iter(train_loader)
		for i, batch in enumerate(train_loader):

			# batch = next(data_loader)
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
					  .format(epoch, i + 1, len(train_loader),
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
		return MaiIndexTransform.prepare_inputs(batch)

	def _forward(self, inputs, targets):

		s_p, e_p = self.model(*inputs)
		if isinstance(self.criterion, PointerLoss):
			loss = self.criterion(s_p, e_p, *targets)
		else:
			raise ValueError("Unsupported loss:", self.criterion)
		prec = 0
		return loss, prec
