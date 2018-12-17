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
cur_cfg = bidaf3
# cur_cfg = slqa_plus3

# cur_cfg = rnet2
# cur_cfg = rnet3

# cur_cfg = mreader2
# cur_cfg = mreader3

# cur_cfg = slqa1
# cur_cfg = slqa2
# cur_cfg = slqa3

# cur_cfg = slqa_plus1
# cur_cfg = slqa_plus2

SEED = 502
EPOCH = 150
BATCH_SIZE = 128

show_plt = False
on_windows = False

from config.config import MODE_OBJ, MODE_MRT, MODE_PTR

mode = MODE_MRT
ms_rouge_eval = Rouge()

if __name__ == '__main__':
  print(cur_cfg.model_params)
  extra_rouge = True
  data_root_folder = './title_data'
  corpus_file = os.path.join(data_root_folder, 'corpus.txt')
  train_file = os.path.join(data_root_folder, 'preprocessed',
                            'train.preprocessed.%sjson' % ('extra_rouge.' if extra_rouge else ''))
  val_file = os.path.join(data_root_folder, 'preprocessed',
                          'val.preprocessed.%sjson' % ('extra_rouge.' if extra_rouge else ''))

  model_dir = os.path.join(data_root_folder, 'models')
  mkdir_if_missing(model_dir)

  if mode == MODE_MRT:
    model_name = cur_cfg.name + '_mrt'
  elif mode == MODE_OBJ:
    model_name = cur_cfg.name + '_obj'
  else:
    model_name = cur_cfg.name

  model_path = os.path.join(model_dir, model_name + '.state')
  print('Model path:', model_path)

  jieba_base_v = Vocab(os.path.join(data_root_folder, 'vocab', 'title_summ.vocab.pkl'),
                       os.path.join(data_root_folder, 'vocab', 'title_summ.emb.pkl'))

  jieba_sgns_v = Vocab(os.path.join(data_root_folder, 'vocab', 'title_summ.vocab.pkl'),
                       os.path.join(data_root_folder, 'vocab', 'title_summ.emb.pkl'))
  jieba_flag_v = Vocab(os.path.join(data_root_folder, 'vocab', 'title_summ.vocab.pkl'),
                       os.path.join(data_root_folder, 'vocab', 'title_summ.emb.pkl'))

  # jieba_sgns_v = Vocab(os.path.join(data_root_folder, 'vocab', 'useless.vocab.pkl'),
  #                      os.path.join(data_root_folder, 'vocab', 'useless.emb.pkl'))
  # jieba_flag_v = Vocab(os.path.join(data_root_folder, 'vocab', 'useless.vocab.pkl'),
  #                      os.path.join(data_root_folder, 'vocab', 'useless.emb.pkl'))

  trainset_roots = [
    os.path.join(data_root_folder, 'val.txt')
  ]

  embed_lists = {
    'jieba': [jieba_base_v.embeddings, jieba_sgns_v.embeddings, jieba_flag_v.embeddings],
    'pyltp': []
  }

  transform = TitleSummTransform(jieba_base_v, jieba_sgns_v, jieba_flag_v)
  train_dataset = TitleSummDataset(train_file, transform, use_rouge=True, max_size=None)  # FIXME:
  dev_dataset = TitleSummDataset(val_file, transform, use_rouge=True, max_size=None)

  num_workers = 0
  # if on_windows:
  #   BATCH_SIZE = 64
  #   num_workers = 0
  # else:
  #   num_workers = max(4, mp.cpu_count())

  train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    # batch_sampler=MethodBasedBatchSampler(data_for_train, batch_size=BATCH_SIZE, seed=SEED),
    num_workers=num_workers,
    collate_fn=transform.batchify,
  )

  dev_loader = DataLoader(
    dataset=dev_dataset,
    batch_size=BATCH_SIZE,
    # batch_sampler=MethodBasedBatchSampler(data_for_dev, batch_size=BATCH_SIZE, shuffle=False),
    num_workers=num_workers,
    collate_fn=transform.batchify
  )

  model_params = cur_cfg.model_params
  model_params['c_max_len'] = 64

  model = RCModel(model_params, embed_lists, mode=mode, emb_trainable=True)
  model = model.cuda()
  if mode == MODE_MRT:
    criterion_main = RougeLoss().cuda()
  elif mode == MODE_OBJ:
    criterion_main = ObjDetectionLoss(B=transform.B, S=transform.S, dynamic_score=False).cuda()
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

  if os.path.isfile(model_path):
    print('load training param, ', model_path)
    if mode == MODE_MRT:
      state = torch.load(model_path)
      model.load_state_dict(state['best_model_state'])
      optimizer.load_state_dict(state['best_opt_state'])
      epoch_list = range(state['best_epoch'] + 1, state['best_epoch'] + 1 + EPOCH)
      global_step = state['best_step']
    elif mode == MODE_OBJ:
      state = torch.load(model_path)
      model.load_state_dict(state['cur_model_state'])
      optimizer.load_state_dict(state['cur_opt_state'])
      epoch_list = range(state['cur_epoch'] + 1, state['cur_epoch'] + 1 + EPOCH)
      global_step = state['cur_step']

      if 'best_score' not in state:
        state['best_score'] = 0
    else:
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
  print_every = 200
  last_val_step = global_step
  if on_windows:
    val_every = [1, 70, 50, 35]
  else:
    val_every = [1000, 700, 500, 350]
  drop_lr_frq = 1
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
        inputs, targets = transform.prepare_inputs(batch, mode != MODE_PTR)

        model.train()
        optimizer.zero_grad()
        out, extra_outputs = model(*inputs)

        if mode == MODE_OBJ:
          widths, centers, scores, extra_targets = targets
        else:
          starts, ends, extra_targets = targets

        q_type_gt, c_in_a_gt, q_in_a_gt, ans_len_gt, delta_rouge = extra_targets
        q_type_pred, c_in_a_pred, q_in_a_pred, ans_len_logits = extra_outputs
        if isinstance(criterion_main, ObjDetectionLoss):
          loss_main = criterion_main(out, widths, centers, scores)
        elif isinstance(criterion_main, RougeLoss) and delta_rouge is not None:
          loss_main = criterion_main(out, delta_rouge)
        elif isinstance(criterion_main, PointerLoss):
          loss_main = criterion_main(*out, starts, ends)
        else:
          raise NotImplementedError

        loss_qtype = criterion_extra(q_type_pred, q_type_gt)
        loss_c_in_a = criterion_extra(c_in_a_pred, c_in_a_gt)
        loss_q_in_a = criterion_extra(q_in_a_pred, q_in_a_gt)
        loss_ans_len = F.nll_loss(F.log_softmax(ans_len_logits, dim=-1), ans_len_gt)

        train_loss = loss_main + 0.2 * (loss_qtype + loss_c_in_a + loss_q_in_a + loss_ans_len)
        train_loss.backward()

        torch.nn.utils.clip_grad_norm_(param_to_update, 5)
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
          # evaluate(show_plt=False)
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
          rl_cina = RougeL()
          ms_rouge_scores = []
          nltk_bleu_scores = []

          with torch.no_grad():
            model.eval()
            for val_batch in dev_loader:

              # cut, cuda
              if mode == MODE_OBJ:
                inputs, targets = transform.prepare_inputs(val_batch, mode != MODE_PTR, obj_eval=(mode == MODE_OBJ))
                starts, ends, widths, centers, scores, extra_targets = targets
              else:
                inputs, targets = transform.prepare_inputs(val_batch, mode != MODE_PTR)
                starts, ends, extra_targets = targets

              _, _, _, ans_len_gt, delta_rouge = extra_targets

              out, ans_len_logits, c_in_a = model(*inputs)

              ans_len_prob = F.softmax(ans_len_logits, dim=-1)

              ans_len_hit = (torch.max(ans_len_prob, -1)[1] == ans_len_gt).sum().item()

              if isinstance(criterion_main, PointerLoss):
                val_loss = criterion_main(*out, starts, ends)
                s_scores, e_scores = out
                out = torch.bmm(s_scores.unsqueeze(-1), e_scores.unsqueeze(1))
                s_scores = s_scores.detach().cpu()
                e_scores = e_scores.detach().cpu()
                out = out.detach().cpu()
                batch_pos1, batch_pos2, confidence = model.decode(s_scores, e_scores)
              elif isinstance(criterion_main, RougeLoss) and delta_rouge is not None:
                val_loss = criterion_main(out, delta_rouge)
                out_ = out.detach().cpu()
                batch_pos1, batch_pos2, confidence = model.decode_outer(out_, top_n=5)
              elif isinstance(criterion_main, ObjDetectionLoss):
                val_loss = criterion_main(out, widths, centers, scores)
                out = out.detach().cpu()
                batch_pos1, batch_pos2, confidence = model.decode_obj_detection(out,
                                                                                B=transform.B,
                                                                                S=transform.S,
                                                                                max_len=512)  ## FIXME: 临时hard coding成512

              else:
                raise NotImplementedError
              starts = starts.detach().cpu()
              ends = ends.detach().cpu()

              if show_plt and val_step % 1000 == 0:
                out = out.detach().cpu()
                for o, sample in zip(out, val_batch['raw']):
                  dim = o.size(0)
                  rouge_matrix = np.zeros([dim, dim])
                  starts = sample['delta_token_starts']
                  ends = sample['delta_token_ends']
                  rouges = sample['delta_rouges']

                  for s, e, r in zip(starts, ends, rouges):
                    rouge_matrix[s, e] = r

                  plt.subplot(121)
                  plt.imshow(rouge_matrix, cmap=plt.cm.hot, vmin=0, vmax=1)
                  plt.title('Gt Rouge')
                  plt.colorbar()

                  plt.subplot(122)
                  plt.imshow(o, cmap=plt.cm.hot, vmin=0, vmax=1)
                  plt.title('Pr Rouge')
                  plt.colorbar()
                  plt.show()

              if mode == MODE_MRT:
                c_in_a = c_in_a.detach().cpu()

                for pos1, pos2, cina, sample, conf in zip(batch_pos1, batch_pos2, c_in_a, val_batch['raw'], confidence):
                  ori_title = ''.join(sample['ori_title_tokens'])
                  gt_ans = sample['answer']

                  c_in_a_pred_ans = ''
                  for idx, ccc in enumerate(cina):
                    if idx >= len(sample['ori_title_tokens']):
                      continue
                    if ccc > 0:
                      c_in_a_pred_ans += sample['ori_title_tokens'][idx]
                  pred_anss = []
                  for p1, p2, c in zip(pos1, pos2, conf):
                    if c < 0.8:
                      break

                    pred_ans = postprocess.gen_ans(p1, p2, sample, key='ori_title_tokens', post_process=False)
                    pred_anss.append((pred_ans, p1))

                  pred_anss = [t for t, _ in sorted(pred_anss, key=lambda d: d[1])]
                  pred_ans = ''.join(pred_anss)
                  rl.add_inst(pred_ans, gt_ans)
                  rl_cina.add_inst(c_in_a_pred_ans, gt_ans)

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

              elif mode == MODE_PTR:
                for pos1, pos2, sample in zip(batch_pos1, batch_pos2, val_batch['raw']):
                  gt_ans = sample['answer']
                  pred_ans = postprocess.gen_ans(pos1, pos2, sample, key='ori_title_tokens', post_process=False)
                  rl.add_inst(pred_ans, gt_ans)

              val_loss_total += val_loss.item()
              val_ans_len_hit += ans_len_hit
              val_start_hit += 0
              val_end_hit += 0
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

          print('RougeL: {: .4f}, RougeL C in Ans: {: .4f} '.format(rouge_score, rl_cina.get_score()))
          if len(nltk_bleu_scores) > 0:
            bleu1_score = np.mean([score[0] for score in nltk_bleu_scores])
            bleu2_score = np.mean([score[1] for score in nltk_bleu_scores])
            bleu4_score = np.mean([score[2] for score in nltk_bleu_scores])
            print(
              'NLTK Bleu-1: {: .4f}, Bleu-2: {: .4f}, Bleu-4: {: .4f}'.format(bleu1_score, bleu2_score, bleu4_score))

          if len(ms_rouge_scores) > 0:
            ms_rouge1 = np.mean([score["rouge-1"]['f'] for score in ms_rouge_scores])
            ms_rouge2 = np.mean([score["rouge-2"]['f'] for score in ms_rouge_scores])
            ms_rougeL = np.mean([score["rouge-l"]['f'] for score in ms_rouge_scores])
            print('MS Rouge-1: {: .4f}, Rouge-2: {: .4f}, Rouge-L: {: .4f}'.format(ms_rouge1, ms_rouge2, ms_rougeL))

          print('-' * 80)
          if state is None:
            state = {}
          # if os.path.isfile(model_path):
          #   state = torch.load(model_path)
          # else:

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

          print('Saving into', model_path)
          state['cur_model_state'] = model.state_dict()
          state['cur_opt_state'] = optimizer.state_dict()
          state['cur_epoch'] = e
          state['val_loss'] = val_loss_total / val_step
          state['val_score'] = rouge_score
          state['cur_step'] = global_step

          torch.save(state, model_path)
