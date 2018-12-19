# encoding:utf-8
import multiprocessing as mp
import matplotlib.pyplot as plt
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import config
from dataset import (TitleSummDataset, Vocab,
                     TitleSummTransform, MethodBasedBatchSampler)
from models.losses import PointerLoss, RougeLoss
from models.losses.obj_detection_loss import ObjDetectionLoss

from models.rc_model import RCModel
from utils.osutils import *
from metrics.rouge import rouge_eval
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu

import postprocess
from title_summ_tokenizer import TitleSummBertTokenizer
from torch.utils.data import Dataset
import json
from pytorch_pretrained_bert import BertModel
from models.title_models import BertForTitleSumm
from pytorch_pretrained_bert.optimization import BertAdam
from torch.utils.data import DataLoader


class TitleSummBertTransform(object):

  def __init__(self, tokenizer, max_len=64):
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __call__(self, sample, index):
    uuid = index
    tokens = ['[CLS]'] + sample['bert_ori_title_tokens'][:self.max_len - 2] + ['[SEP]']
    input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
    input_masks = [1] * len(input_ids)
    segment_ids = [1] * len(input_ids)

    pad_len = self.max_len - len(input_ids)
    input_ids += [0] * pad_len
    input_masks += [0] * pad_len
    segment_ids += [0] * pad_len

    rouge_matrix = np.zeros([self.max_len, self.max_len])
    for start, end, rouge in zip(sample['delta_token_starts'], sample['delta_token_ends'], sample['delta_rouges']):
      start, end = start + 1, end + 1
      if end >= self.max_len - 2:
        continue
      rouge_matrix[start, end] = rouge

    return uuid, input_ids, segment_ids, input_masks, rouge_matrix

  def batchify(self, batch):
    uuid_batch, input_ids_batch, segment_ids_batch, input_masks_batch, rouge_matrix_batch = [], [], [], [], []
    for sample in batch:
      uuid, input_ids, segment_ids, input_masks, rouge_matrix = sample

      uuid_batch.append(uuid)
      input_ids_batch.append(input_ids)
      segment_ids_batch.append(segment_ids)
      input_masks_batch.append(input_masks)
      rouge_matrix_batch.append(rouge_matrix)

    long_tensors = [uuid_batch, input_ids_batch, segment_ids_batch, input_masks_batch]
    uuid_batch, input_ids_batch, segment_ids_batch, input_masks_batch = (torch.tensor(t, dtype=torch.long) for t in
                                                                         long_tensors)

    rouge_matrix_batch = torch.FloatTensor(rouge_matrix_batch)

    return uuid_batch, input_ids_batch, segment_ids_batch, input_masks_batch, rouge_matrix_batch


class TitleSummBertDataset(Dataset):
  def __init__(self, data_path, transform, max_size=None):
    data_source = []
    with open(data_path, encoding='utf-8') as f:
      lines = f.readlines()
      if max_size is not None and max_size > 0:
        lines = lines[:max_size]
      for line in lines:
        sample = json.loads(line)
        sample['bert_ori_title_tokens'] = sample['ori_title_tokens'][:]
        sample['ori_title_tokens'] = [self.clean_token(token) for token in sample['ori_title_tokens']]
        data_source.append(sample)

    self.data_source = data_source
    self.transformed_data = {}
    self.transform = transform

  def clean_token(self, token):
    if token == '[ ]':
      return ' '
    if token.startswith('##'):
      return token[2:]
    return token

  def __len__(self):
    return len(self.data_source)

  def __getitem__(self, indices):
    if isinstance(indices, (tuple, list)):
      return [self.__get_single_item__(index) for index in indices]
    return self.__get_single_item__(indices)

  def __get_single_item__(self, index):
    if index in self.transformed_data:
      key_data = self.transformed_data[index]
      return key_data
    else:
      key_data = self.transform(self.data_source[index], index)
      self.transformed_data[index] = key_data

      return key_data


if __name__ == '__main__':
  data_root_folder = './title_data'
  corpus_file = os.path.join(data_root_folder, 'corpus.txt')
  train_file = os.path.join(data_root_folder, 'preprocessed',
                            'train.preprocessed.extra_rouge.bert-base-multilingual-cased.json')  # fixme
  val_file = os.path.join(data_root_folder, 'preprocessed',
                          'val.preprocessed.extra_rouge.bert-base-multilingual-cased.json')

  model_dir = os.path.join(data_root_folder, 'models')
  mkdir_if_missing(model_dir)

  SEED = 502
  BATCH_SIZE = 8

  show_plt = True
  on_windows = True

  from config.config import MODE_OBJ, MODE_MRT, MODE_PTR

  mode = MODE_MRT
  ms_rouge_eval = Rouge()

  model_path = os.path.join(model_dir, mode + '.bert.state')
  print('Model path:', model_path)

  tokenizer = TitleSummBertTokenizer('title_data/vocab/bert-base-multilingual-cased.vocab')
  transform = TitleSummBertTransform(tokenizer=tokenizer, max_len=64)

  val_dataset = TitleSummBertDataset(val_file, transform, max_size=None)

  val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=transform.batchify, shuffle=False)

  model = BertForTitleSumm.from_pretrained('bert-base-multilingual-cased', cache_dir='bert_ckpts')
  model = model.cuda()
  state = torch.load(model_path)
  model.load_state_dict(state['best_model_state'])

  state = {}

  # model = torch.load(model_path)

  device = torch.device("cuda")
  global_step = 0

  from title_summ_bert_train import evaluate

  device = torch.device("cuda")

  val_loss, val_scores = evaluate(model, val_dataset, val_dataloader, show_plt, device)
  rouge_score = val_scores['rouge_score']
  nltk_bleu_scores = val_scores['nltk_bleus']
  ms_rouges = val_scores['ms_rouges']

  # print('Val Epoch: [{}][{}/{}]\t'
  #       'Loss: Main {:.4f}\t'
  #       .format(epoch, step, len(train_dataloader), val_loss))
  print('Loss: Main {:.4f}\t'
        .format(val_loss))
  print('Rouge-L: {:.4f}'.format(rouge_score))
  print('NLTK Bleu-1: {: .4f}, Bleu-2: {: .4f}, Bleu-4: {: .4f}'
        .format(nltk_bleu_scores[0], nltk_bleu_scores[1], nltk_bleu_scores[2]))
  print('MS Rouge-1: {: .4f}, Rouge-2: {: .4f}, Rouge-L: {: .4f}'
        .format(ms_rouges[0], ms_rouges[1], ms_rouges[2]))
