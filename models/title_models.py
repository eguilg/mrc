from pytorch_pretrained_bert.modeling import PreTrainedBertModel
from pytorch_pretrained_bert import BertModel
import torch
import torch.nn as nn
from models.layers.outer_layer import OuterNet
from models.losses.rouge_loss import RougeLoss
import numpy as np


class BertForTitleSumm(PreTrainedBertModel):
  def __init__(self, config):
    super(BertForTitleSumm, self).__init__(config)
    self.bert = BertModel(config)

    # self.out_layer = out_layer
    # self.criterion = criterion

    self.out_layer = OuterNet(
      x_size=768,  # bert的输出shape
      y_size=0,
      hidden_size=128,
      dropout_rate=0.1,
    )
    self.criterion = RougeLoss().cuda()

    self.apply(self.init_bert_weights)

  def forward(self, input_ids, token_type_ids=None, attention_mask=None, rouge_matrix=None):
    sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
    attention_mask = attention_mask.byte()
    out = self.out_layer(sequence_output, sequence_output, attention_mask, attention_mask)

    if rouge_matrix is not None:
      return self.criterion(out, rouge_matrix), out
    else:
      return out

  @staticmethod
  def decode_outer(outer, top_n=1, max_len=None):
    """Take argmax of constrained score_s * score_e.

    Args:
      score_s: independent start predictions
      score_e: independent end predictions
      top_n: number of top scored pairs to take
      max_len: max span length to consider
    """
    pred_s = []
    pred_e = []
    pred_score = []
    max_len = max_len or outer.size(1)
    for i in range(outer.size(0)):
      # Outer product of scores to get full p_s * p_e matrix
      scores = outer[i]

      # Zero out negative length and over-length span scores
      scores.triu_().tril_(max_len - 1)

      # Take argmax or top n
      scores = scores.numpy().copy()
      scores_flat = scores.flatten()
      if top_n == 1:
        idx_sort = [np.argmax(scores_flat)]
      elif len(scores_flat) < top_n:
        idx_sort = np.argsort(-scores_flat)
      else:
        ## FIXME: 针对多答案的临时魔改: 每次选中一个prob最大的答案，然后把临近的±3行3列全都置零
        raw_score = scores.copy()
        idx_sort = []
        for _ in range(top_n):
          idx = np.argmax(scores_flat)
          idx_sort.append(idx)
          s, e = np.unravel_index(idx, scores.shape)
          # s,e=s[0],e[0]
          scores[:s + 1, e:] = 0
          scores[s:e + 1, :] = 0
          scores[:, s:e + 1] = 0
          scores_flat = scores.flatten()
      s_idx, e_idx = np.unravel_index(idx_sort, scores.shape)

      if top_n == 1:
        pred_s.append(s_idx[0])
        pred_e.append(e_idx[0])
        pred_score.append(scores_flat[idx_sort][0])
      elif len(scores_flat) < top_n:
        pred_s.append(s_idx)
        pred_e.append(e_idx)
        pred_score.append(scores_flat[idx_sort])
      else:
        pred_s.append(s_idx)
        pred_e.append(e_idx)
        pred_score.append(raw_score.flatten()[idx_sort])
    return pred_s, pred_e, pred_score
