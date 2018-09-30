import os

from torch.utils.data import Dataset

from utils.serialization import *


class MaiDataSource(object):

	def __init__(self, root_dirs):
		self.data = []  # (aid, qid, method, fpath)
		self.all_aid = set()
		self.dev_split = 0
		self.seed = 502

		self.dev_aid = []
		self.train = []
		self.dev = []
		for rdir in root_dirs:
			if not osp.isdir(rdir):
				print("{} is not a dir".format(rdir))
				continue

			if 'jieba' in osp.basename(rdir):
				method = 'jieba'
			elif 'pyltp' in osp.basename(rdir):
				method = 'pyltp'
			else:
				print("{} method not known".format(osp.basename(rdir)))
				continue

			for fname in os.listdir(rdir):
				fpath = osp.join(rdir, fname)
				if not osp.isfile(fpath):
					continue
				try:
					ids = osp.splitext(fname)[0].split('_')
					aid = ids[0]
					qid = ids[1]

					self.data.append((aid, qid, method, fpath))
					self.all_aid.add(aid)
				except Exception:
					continue
			print("added root dir: {}".format(rdir))
		self.all_aid = list(sorted(list(self.all_aid)))
		self.train = self.data

	def split(self, dev_split=0.1, seed=502):

		self.dev_split = dev_split
		self.seed = seed

		np.random.seed(self.seed)
		np.random.shuffle(self.all_aid)
		self.dev_aid = self.all_aid[: int(self.dev_split * len(self.all_aid))]

		self.dev = list(filter(lambda item: item[0] in self.dev_aid, self.data))
		self.train = list(filter(lambda item: item[0] not in self.dev_aid, self.data))


class MaiDataset(Dataset):
	def __init__(self, data_source, transform):
		self.data_source = data_source
		self.transform = transform

	def __getitem__(self, indices):
		if isinstance(indices, (tuple, list)):
			return [self.__get_single_item__(index) for index in indices]
		return self.__get_single_item__(indices)

	def __len__(self):
		return len(self.data_source)

	def __get_single_item__(self, index):
		aid, qid, method, fpath = self.data_source[index]
		item = read_json(fpath)
		return self.transform(item, method)
