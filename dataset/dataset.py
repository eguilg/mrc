""" data source and data set classes """

import os
import json
from torch.utils.data import Dataset

from utils.serialization import *


class MaiDirDataSource(object):

  def __init__(self, root_dirs, score_thresh=0.75):
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
          if len(ids) == 3 and float(ids[2]) < score_thresh:
            continue
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


class MaiDirDataset(Dataset):
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


# TODO: MaiFileDataset for seed in test

class MaiWindowsDataset(Dataset):
  def __init__(self, data_path, transform, use_rouge, multi_ans=False):

    self.use_rouge = use_rouge
    with open(data_path, encoding='utf-8') as f:
      self.data_source = json.load(f)[:1000]

    self.transformed_data = {}
    self.transform = transform

  def __getitem__(self, indices):
    if isinstance(indices, (tuple, list)):
      return [self.__get_single_item__(index) for index in indices]
    return self.__get_single_item__(indices)

  def __get_single_item__(self, index):
    try:
      key_data = self.transform(self.data_source[index], method='jieba')
    except:
      key_data = self.transform(self.data_source[0], method='jieba')

    return key_data

  def __len__(self):
    return len(self.data_source)


class TitleSummDataset(Dataset):
  def __init__(self, data_path, transform, use_rouge, max_size=None):
    self.use_rouge = use_rouge

    data_source = []
    with open(data_path, encoding='utf-8') as f:
      lines = f.readlines()
      if max_size is not None and max_size > 0:
        lines = lines[:max_size]
      for line in lines:
        data_source.append(json.loads(line))

    self.data_source = data_source
    self.transformed_data = {}
    self.transform = transform

  def __getitem__(self, indices):
    if isinstance(indices, (tuple, list)):
      return [self.__get_single_item__(index) for index in indices]
    return self.__get_single_item__(indices)

  def __get_single_item__(self, index):
    if index in self.transformed_data:
      key_data = self.transformed_data[index]
      return key_data
    else:
      key_data = self.transform(self.data_source[index], method='jieba')
      self.transformed_data[index] = key_data

      return key_data

  def __len__(self):
    return len(self.data_source)


class SquadDataset(Dataset):
  def __init__(self, data_path, transform, use_rouge, max_size=None, c_max_len=384, q_max_len=64):
    self.use_rouge = use_rouge

    data_source = []
    with open(data_path, encoding='utf-8') as f:
      lines = f.readlines()
      if max_size is not None and max_size > 0:
        lines = lines[:max_size]
      for line in lines:
        t = json.loads(line)
        if t['is_impossible']:
          continue
        if t['answer_token_end'] >= c_max_len - 1:
          continue
        t['article_tokens'] = t['article_tokens'][:c_max_len]
        t['question_tokens'] = t['question_tokens'][:q_max_len]
        if len(t['article_tokens']) <= 0:
          print()
        data_source.append(t)

    self.data_source = data_source
    self.transformed_data = {}
    self.transform = transform

  def __getitem__(self, indices):
    if isinstance(indices, (tuple, list)):
      return [self.__get_single_item__(index) for index in indices]
    return self.__get_single_item__(indices)

  def __get_single_item__(self, index):
    if index in self.transformed_data:
      key_data = self.transformed_data[index]
      return key_data
    else:
      key_data = self.transform(self.data_source[index], method='jieba')
      self.transformed_data[index] = key_data

      return key_data

  def __len__(self):
    return len(self.data_source)
