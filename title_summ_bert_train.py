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
from metrics import RougeL
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
    tokens = ['[CLS]'] + sample['ori_title_tokens'][:self.max_len - 2] + ['[SEP]']
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
        data_source.append(json.loads(line))

    self.data_source = data_source
    self.transformed_data = {}
    self.transform = transform

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

  lr = 5e-3
  SEED = 502
  EPOCH = 5
  BATCH_SIZE = 8

  print_every = 2000
  eval_every = 2000

  show_plt = True
  on_windows = True

  from config.config import MODE_OBJ, MODE_MRT, MODE_PTR

  mode = MODE_MRT
  ms_rouge_eval = Rouge()

  model_path = os.path.join(model_dir, mode + '.bert.state')
  print('Model path:', model_path)

  tokenizer = TitleSummBertTokenizer('title_data/vocab/bert-base-multilingual-cased.vocab')

  transform = TitleSummBertTransform(tokenizer=tokenizer, max_len=64)

  train_dataset = TitleSummBertDataset(train_file, transform, max_size=None)
  val_dataset = TitleSummBertDataset(val_file, transform, max_size=None)

  train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=transform.batchify, shuffle=True)
  val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=transform.batchify, shuffle=True)

  model = BertForTitleSumm.from_pretrained('bert-base-multilingual-cased', cache_dir='bert_ckpts')
  model = model.cuda()
  param_optimizer = list(model.named_parameters())
  param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

  no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
  optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
  ]

  optimizer = BertAdam(optimizer_grouped_parameters,
                       lr=lr)

  device = torch.device("cuda")
  global_step = 0

  train_loss_sum = 0
  for epoch in range(1, EPOCH + 1):
    for step, batch in enumerate(tqdm(train_dataloader), start=1):
      batch = tuple(t.to(device) for t in batch)
      uuid_batch, input_ids_batch, segment_ids_batch, input_masks_batch, rouge_matrix_batch = batch

      loss = model(input_ids_batch, segment_ids_batch, input_masks_batch, rouge_matrix_batch)
      loss.backward()
      train_loss_sum += loss.item()

      lr_this_step = lr
      for param_group in optimizer.param_groups:
        param_group['lr'] = lr_this_step
      optimizer.step()
      optimizer.zero_grad()
      global_step += 1

      if global_step % print_every == 0:
        time.sleep(0.02)
        print('Epoch: [{}][{}/{}]\t'
              'Loss: Main {:.4f}\t'
              .format(epoch, step, len(train_dataloader),
                      train_loss_sum / print_every))
        train_loss_sum = 0
      #
      # if global_step % eval_every == 0:
      #   print('-' * 80)
      #   print('Evaluating...')
      #   with torch.no_grad():
      #     model.eval()
      #     val_loss_sum = 0
      #     for val_batch in val_dataloader:
      #       uuid_batch, input_ids_batch, segment_ids_batch, input_masks_batch, rouge_matrix_batch = batch
      #       loss = model(input_ids_batch, segment_ids_batch, input_masks_batch, rouge_matrix_batch)
      #       val_loss_sum += loss.item()
      #     print('Val Epoch: [{}][{}/{}]\t'
      #           'Loss: Main {:.4f}\t'
      #           .format(epoch, step, len(train_dataloader),
      #                   val_loss_sum / len(val_dataloader)))
      #   torch.save(model, model_path)
    torch.save(model, model_path)
    torch.save(model, model_path + '%d' % epoch)
