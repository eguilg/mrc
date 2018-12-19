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
from pytorch_pretrained_bert import BertModel, BertConfig, BertForQuestionAnswering
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
    uuid_batch, input_ids_batch, segment_ids_batch, input_masks_batch = (
      torch.tensor(t, dtype=torch.long) for t in long_tensors)

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


def evaluate(model, val_dataset, val_dataloader, show_plt, device):
  with torch.no_grad():
    model.eval()
    val_loss_sum = 0
    rouge_scores = []
    ms_rouge_scores = []
    nltk_bleu_scores = []
    for step, batch in enumerate(val_dataloader):
      batch = tuple(t.to(device) for t in batch)
      uuid_batch, input_ids_batch, segment_ids_batch, input_masks_batch, rouge_matrix_batch = batch

      loss, out = model(input_ids_batch, segment_ids_batch, input_masks_batch, rouge_matrix_batch)
      val_loss_sum += loss.item()

      raw_samples = [val_dataset.data_source[int(id)] for id in uuid_batch]
      out_ = out.detach().cpu()
      if show_plt and step % 1000 == 0:
        for o, rouge_matrix in zip(out_, rouge_matrix_batch):
          plt.subplot(121)
          plt.imshow(rouge_matrix, cmap=plt.cm.hot, vmin=0, vmax=1)
          plt.title('Gt Rouge')
          plt.colorbar()

          plt.subplot(122)
          plt.imshow(o, cmap=plt.cm.hot, vmin=0, vmax=1)
          plt.title('Pr Rouge')
          plt.colorbar()
          plt.show()

      batch_pos1, batch_pos2, confidence = model.decode_outer(out_, top_n=5)
      for pos1, pos2, conf, sample in zip(batch_pos1, batch_pos2, confidence, raw_samples):
        ori_title = ''.join(sample['ori_title_tokens'])
        gt_ans = sample['answer']
        pred_anss = []
        for p1, p2, c in zip(pos1, pos2, conf):
          p1, p2 = p1 - 1, p2 - 1  ## 第一个位置是cls, 所以要去掉
          if c < 0.8:
            break
          pred_ans = postprocess.gen_ans(p1, p2, sample, key='ori_title_tokens', post_process=False)
          pred_anss.append((pred_ans, p1))

        pred_anss = [t for t, _ in sorted(pred_anss, key=lambda d: d[1])]
        pred_ans = ''.join(pred_anss)
        rouge_score, _, _ = rouge_eval.calc_score(pred_ans, gt_ans)
        rouge_scores.append(rouge_score)
        try:
          rouge_score = ms_rouge_eval.get_scores(hyps=[' '.join(list(pred_ans))],
                                                 refs=[' '.join(list(gt_ans))])
          ms_rouge_scores.append(rouge_score[0])
        except:
          pass
        bleu1_score = sentence_bleu(references=[list(gt_ans)], hypothesis=list(pred_ans),
                                    weights=(1, 0, 0, 0))
        bleu2_score = sentence_bleu(references=[list(gt_ans)], hypothesis=list(pred_ans),
                                    weights=(0.5, 0.5, 0.0, 0.0))
        bleu4_score = sentence_bleu(references=[list(gt_ans)], hypothesis=list(pred_ans),
                                    weights=(0.25, 0.25, 0.25, 0.25))
        nltk_bleu_scores.append((bleu1_score, bleu2_score, bleu4_score))

    print('Max confidence:', conf[0])
    val_loss = val_loss_sum / len(val_dataloader)
    bleu1_score, bleu2_score, bleu4_score = 0, 0, 0
    if len(nltk_bleu_scores) > 0:
      bleu1_score = np.mean([score[0] for score in nltk_bleu_scores])
      bleu2_score = np.mean([score[1] for score in nltk_bleu_scores])
      bleu4_score = np.mean([score[2] for score in nltk_bleu_scores])

    ms_rouge1, ms_rouge2, ms_rougeL = 0, 0, 0
    if len(ms_rouge_scores) > 0:
      ms_rouge1 = np.mean([score["rouge-1"]['f'] for score in ms_rouge_scores])
      ms_rouge2 = np.mean([score["rouge-2"]['f'] for score in ms_rouge_scores])
      ms_rougeL = np.mean([score["rouge-l"]['f'] for score in ms_rouge_scores])

    score = {
      'rouge_score': np.mean(rouge_scores),
      'nltk_bleus': [bleu1_score, bleu2_score, bleu4_score],
      'ms_rouges': [ms_rouge1, ms_rouge2, ms_rougeL]
    }
  model.train()
  return val_loss, score


if __name__ == '__main__':
  data_root_folder = './title_data'
  train_file = os.path.join(data_root_folder, 'preprocessed',
                            'train.preprocessed.extra_rouge.bert-base-multilingual-cased.json')  # fixme
  val_file = os.path.join(data_root_folder, 'preprocessed',
                          'val.preprocessed.extra_rouge.bert-base-multilingual-cased.json')

  model_dir = os.path.join(data_root_folder, 'models')
  mkdir_if_missing(model_dir)

  lr = 5e-3
  SEED = 502
  EPOCH = 150
  BATCH_SIZE = 64

  show_plt = False
  on_windows = False
  if on_windows:
    BATCH_SIZE = 8

  from config.config import MODE_OBJ, MODE_MRT, MODE_PTR

  mode = MODE_MRT
  ms_rouge_eval = Rouge()

  model_path = os.path.join(model_dir, mode + '.bert.state')
  print('Model path:', model_path)

  tokenizer = TitleSummBertTokenizer('title_data/vocab/bert-base-multilingual-cased.vocab')
  transform = TitleSummBertTransform(tokenizer=tokenizer, max_len=64)
  train_dataset = TitleSummBertDataset(val_file, transform, max_size=None)  # FIXME: 临时在val上做过拟合测试
  val_dataset = TitleSummBertDataset(val_file, transform, max_size=None)
  train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=transform.batchify, shuffle=True)
  val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=transform.batchify, shuffle=False)

  # if not os.path.isfile(model_path):
  #   model = BertForTitleSumm.from_pretrained('bert-base-multilingual-cased', cache_dir='bert_ckpts')
  # else:
  #   model = BertForTitleSumm(BertConfig.from_json_file('bert_ckpts/bert_config.json'))
  model = BertForTitleSumm.from_pretrained('bert-base-multilingual-cased', cache_dir='bert_ckpts')
  model = model.cuda()

  param_optimizer = list(model.named_parameters())
  param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
  no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
  optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
  ]
  optimizer = BertAdam(optimizer_grouped_parameters, lr=lr)

  if os.path.isfile(model_path):
    print('=' * 80)
    print('load training param, ', model_path)
    print('=' * 80)
    state = torch.load(model_path)
    model.load_state_dict(state['best_model_state'])
    optimizer.load_state_dict(state['best_opt_state'])
    epoch_list = range(state['best_epoch'] + 1, state['best_epoch'] + 1 + EPOCH)
    global_step = state['best_step']

    state = {}  ## FIXME:  临时的解决方案
  else:
    state = None
    epoch_list = range(EPOCH)
    global_step = 0

  grade = 0

  if on_windows:
    print_every = 50
    val_every = [50, 70, 50, 35]
  else:
    print_every = 2000
    val_every = [10000, 700, 500, 350]
  drop_lr_frq = 2
  device = torch.device("cuda")

  val_no_improve = 0
  train_loss_sum = 0

  for epoch in range(1, EPOCH + 1):
    with tqdm(total=len(train_dataloader)) as bar:
      for step, batch in enumerate(train_dataloader, start=1):
        model.train()
        optimizer.zero_grad()
        batch = tuple(t.to(device) for t in batch)
        uuid_batch, input_ids_batch, segment_ids_batch, input_masks_batch, rouge_matrix_batch = batch

        loss, _ = model(input_ids_batch, segment_ids_batch, input_masks_batch, rouge_matrix_batch)
        loss.backward()
        train_loss_sum += loss.item()

        lr_this_step = lr
        for param_group in optimizer.param_groups:
          param_group['lr'] = lr_this_step
        optimizer.step()

        global_step += 1
        if global_step % print_every == 0:
          bar.update(min(print_every, step))
          time.sleep(0.02)
          print('Epoch: [{}][{}/{}]\t'
                'Loss: Main {:.4f}\t'
                .format(epoch, step, len(train_dataloader),
                        train_loss_sum / print_every))
          train_loss_sum = 0

        if global_step % val_every[grade] == 0:
          print('-' * 80)
          print('Evaluating...')
          val_loss, val_scores = evaluate(model, val_dataset, val_dataloader, show_plt, device)
          rouge_score = val_scores['rouge_score']
          nltk_bleu_scores = val_scores['nltk_bleus']
          ms_rouges = val_scores['ms_rouges']

          print('Val Epoch: [{}][{}/{}]\t'
                'Loss: Main {:.4f}\t'
                .format(epoch, step, len(train_dataloader), val_loss))
          print('Rouge-L: {:.4f}'.format(rouge_score))
          print('NLTK Bleu-1: {: .4f}, Bleu-2: {: .4f}, Bleu-4: {: .4f}'
                .format(nltk_bleu_scores[0], nltk_bleu_scores[1], nltk_bleu_scores[2]))
          print('MS Rouge-1: {: .4f}, Rouge-2: {: .4f}, Rouge-L: {: .4f}'
                .format(ms_rouges[0], ms_rouges[1], ms_rouges[2]))

          if state is None:
            state = {}
          if state == {} or state['best_score'] < rouge_score:
            state['best_model_state'] = model.state_dict()
            state['best_opt_state'] = optimizer.state_dict()
            state['best_loss'] = val_loss
            state['best_score'] = rouge_score
            state['best_epoch'] = epoch
            state['best_step'] = global_step

            val_no_improve = 0
          else:
            val_no_improve += 1

            if val_no_improve >= int(drop_lr_frq):
              print('dropping lr...')
              val_no_improve = 0
              drop_lr_frq += 1.4
              lr_total = 0
              lr_num = 0
              for param_group in optimizer.param_groups:
                if param_group['lr'] > 2e-5:
                  param_group['lr'] *= 0.5
                lr_total += param_group['lr']
                lr_num += 1
              print('curr avg lr is {}'.format(lr_total / lr_num))

          state['cur_model_state'] = model.state_dict()
          state['cur_opt_state'] = optimizer.state_dict()
          state['cur_epoch'] = epoch
          state['val_loss'] = val_loss
          state['val_score'] = rouge_score
          state['cur_step'] = global_step

          torch.save(state, model_path)
          print('Saved into', model_path)
