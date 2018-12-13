""" Reader wrapper """
import numpy as np
import torch
import torch.nn as nn

from . import BACKBONE_TYPES, RNN_TYPES, POINTER_TYPES, OuterNet
from models.layers.obj_detection_layer import ObjDetectionNet
from .layers.embedding_layer import MergedEmbedding
from config.config import MODE_OBJ, MODE_MRT, MODE_PTR


class RCModel(nn.Module):
  def __init__(self, param_dict, embed_lists, normalize=True, mode=MODE_PTR,emb_trainable=False):
    super(RCModel, self).__init__()
    # Store config
    self.param_dict = param_dict
    self.num_features = param_dict['num_features']
    self.num_qtype = param_dict['num_qtype']
    self.hidden_size = param_dict['hidden_size']
    self.dropout = param_dict['dropout']
    self.backbone_kwarg = param_dict['backbone_kwarg']
    self.ptr_kwarg = param_dict['ptr_kwarg']
    self.mode = mode

    try:
      self.backbone_type = BACKBONE_TYPES[param_dict['backbone_type']]
    except KeyError:
      raise KeyError('Wrong backbone type')
    try:
      self.rnn_type = RNN_TYPES[param_dict['rnn_type']]
    except KeyError:
      raise KeyError('Wrong rnn type')
    try:
      self.ptr_type = POINTER_TYPES[param_dict['ptr_type']]
    except KeyError:
      raise KeyError('Wrong pointer type')

    self.merged_embeddings_jieba = MergedEmbedding(embed_lists['jieba'],trainable=emb_trainable)
    self.merged_embeddings_pyltp = MergedEmbedding(embed_lists['pyltp'],trainable=emb_trainable)
    self.merged_embeddings = {
      'jieba': self.merged_embeddings_jieba,
      'pyltp': self.merged_embeddings_pyltp
    }

    self.doc_input_size = self.merged_embeddings['jieba'].output_dim + self.num_features

    self.backbone = self.backbone_type(
      input_size=self.doc_input_size,
      hidden_size=self.hidden_size,
      dropout=self.dropout,
      rnn_type=self.rnn_type,
      **self.backbone_kwarg
    )

    if mode == MODE_MRT:
      self.out_layer = OuterNet(
        x_size=self.backbone.out1_dim,
        y_size=self.backbone.out2_dim,
        hidden_size=self.hidden_size,
        dropout_rate=self.dropout,
        normalize=normalize,
        # **self.ptr_kwarg
      )
    elif mode == MODE_OBJ:
      self.out_layer = ObjDetectionNet(
        x_size=self.backbone.out1_dim,
        y_size=self.backbone.out2_dim,
        hidden_size=self.hidden_size,
        dropout_rate=self.dropout,
        normalize=normalize,
        # **self.ptr_kwarg
      )
    else:
      self.out_layer = self.ptr_type(
        x_size=self.backbone.out1_dim,
        y_size=self.backbone.out2_dim,
        hidden_size=self.hidden_size,
        dropout_rate=self.dropout,
        normalize=normalize,
        **self.ptr_kwarg
      )
    # self.ptr_net = self.outer_net

    self.qtype_net = nn.Linear(
      in_features=2 * self.hidden_size,
      out_features=self.num_qtype,
      bias=True
    )

    self.isin_net = nn.Linear(
      in_features=2 * self.hidden_size,
      out_features=1,
      bias=True
    )

    self.ans_len_net = nn.Linear(
      in_features=2 * self.hidden_size,
      out_features=21,
      bias=True
    )

  def reset_embeddings(self, embed_lists):
    self.merged_embeddings_jieba = MergedEmbedding(embed_lists['jieba'])
    self.merged_embeddings_pyltp = MergedEmbedding(embed_lists['pyltp'])
    self.merged_embeddings = {
      'jieba': self.merged_embeddings_jieba,
      'pyltp': self.merged_embeddings_pyltp
    }
    self.doc_input_size = self.merged_embeddings['jieba'].output_dim + self.num_features

  def forward(self, x1_list, x1_f_list, x1_mask, x2_list, x2_f_list, x2_mask, method):
    """Inputs:
    x1_list = document word indices of different vocabs		list([batch * len_d])
    x1_f_list = document word features indices				list([batch * len_d])
    x1_mask = document padding mask      					[batch * len_d]
    x2_list = question word indices of different vocabs		list([batch * len_q])
    x2_f_list = document word features indices  			list([batch * len_q])
    x2_mask = question padding mask        					[batch * len_q]
    """
    # Embed both document and question
    x1_emb = self.merged_embeddings[method](x1_list)
    x2_emb = self.merged_embeddings[method](x2_list)

    # Combine input
    crnn_input = [x1_emb]
    qrnn_input = [x2_emb]

    # Add manual features
    if self.num_features > 0:
      x1_f = torch.cat([f.unsqueeze(2) for f in x1_f_list], -1)
      x2_f = torch.cat([f.unsqueeze(2) for f in x2_f_list], -1)
      crnn_input.append(x1_f)
      qrnn_input.append(x2_f)

    # cat document
    c = torch.cat(crnn_input, -1)

    # cat question
    q = torch.cat(qrnn_input, -1)

    c, q = self.backbone(c, x1_mask, q, x2_mask)

    out = self.out_layer(c, q, x1_mask, x2_mask)

    ans_len_logits = self.ans_len_net(torch.max(torch.cat([c, q], 1), dim=1)[0])

    if self.training:
      qtype_vec = self.qtype_net(q[:, -1, :])
      c_in_a = self.isin_net(c).squeeze(-1)
      q_in_a = self.isin_net(q).squeeze(-1)
      return out, (qtype_vec, c_in_a, q_in_a, ans_len_logits)
    else:
      return out, ans_len_logits

  @staticmethod
  def decode(score_s, score_e, top_n=1, max_len=None):
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
    max_len = max_len or score_s.size(1)
    for i in range(score_s.size(0)):
      # Outer product of scores to get full p_s * p_e matrix
      scores = torch.ger(score_s[i], score_e[i])

      # Zero out negative length and over-length span scores
      scores.triu_().tril_(max_len - 1)

      # Take argmax or top n
      scores = scores.numpy()
      scores_flat = scores.flatten()
      if top_n == 1:
        idx_sort = [np.argmax(scores_flat)]
      elif len(scores_flat) < top_n:
        idx_sort = np.argsort(-scores_flat)
      else:
        idx = np.argpartition(-scores_flat, top_n)[0:top_n]
        idx_sort = idx[np.argsort(-scores_flat[idx])]

      s_idx, e_idx = np.unravel_index(idx_sort, scores.shape)

      if top_n == 1:
        pred_s.append(s_idx[0])
        pred_e.append(e_idx[0])
        pred_score.append(scores_flat[idx_sort][0])
      elif top_n > 1:
        pred_s.append(s_idx)
        pred_e.append(e_idx)
        pred_score.append(scores_flat[idx_sort])
      else:
        raise ValueError
    del score_s, score_e
    return pred_s, pred_e, pred_score


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
        raw_score=scores.copy()
        idx_sort=[]
        for _ in range(top_n):
          idx=np.argmax(scores_flat)
          idx_sort.append(idx)
          s,e= np.unravel_index(idx, scores.shape)
          # s,e=s[0],e[0]
          scores[:s+1,e:]=0
          scores[s:e+1,:]=0
          scores[:,s:e+1]=0
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
    del outer
    return pred_s, pred_e, pred_score

  @staticmethod
  def decode_obj_detection(out, B, S, max_len):
    pred_centers = out[:, :, 0]
    pred_widths = out[:, :, 1]
    pred_probs = out[:, :, 2]

    pred_S_idx = torch.argmax(pred_probs, dim=-1)

    pred_centers = torch.Tensor([pred_centers[bs, s_idx] for bs, s_idx in enumerate(pred_S_idx)])
    pred_widths = torch.Tensor([pred_widths[bs, s_idx] for bs, s_idx in enumerate(pred_S_idx)])
    pred_probs = torch.Tensor([pred_probs[bs, s_idx] for bs, s_idx in enumerate(pred_S_idx)])

    block_size = max_len // S
    pred_centers = (pred_centers + pred_S_idx.float()) * block_size
    pred_widths *= max_len

    pred_starts = torch.round(pred_centers - pred_widths / 2).long()
    pred_ends = torch.round(pred_centers + pred_widths / 2).long()

    return pred_starts, pred_ends, pred_probs

  @staticmethod
  def decode_cut_ans(score_s, score_e, ans_len_prob, top_n=1, max_len=None):
    """Take argmax of constrained score_s * score_e.

    Args:
      score_s: independent start predictions
      score_e: independent end predictions
      ans_len: ans len probs
      top_n: number of top scored pairs to take
      max_len: max span length to consider
    """
    pred_s = []
    pred_e = []
    pred_score = []
    max_len = max_len or score_s.size(1)
    for i in range(score_s.size(0)):
      # Outer product of scores to get full p_s * p_e matrix
      scores = torch.ger(score_s[i], score_e[i])

      # Zero out negative length and over-length span scores
      scores.triu_()

      #  cut ans
      pred_ans_prob, pred_ans_len = torch.max(ans_len_prob[i], -1)
      pred_ans_prob, pred_ans_len = pred_ans_prob.item(), pred_ans_len.item()

      if 0 < pred_ans_len < 20 and pred_ans_prob >= 0.5:
        ptr_prob = torch.max(scores).item()
        if ptr_prob < 0.6:
          scores.triu_(max(0, pred_ans_len - 2))
          scores.tril_(pred_ans_len + 3)

      scores.tril_(max_len - 1)

      # Take argmax or top n
      scores = scores.numpy()
      scores_flat = scores.flatten()
      if top_n == 1:
        idx_sort = [np.argmax(scores_flat)]
      elif len(scores_flat) < top_n:
        idx_sort = np.argsort(-scores_flat)
      else:
        idx = np.argpartition(-scores_flat, top_n)[0:top_n]
        idx_sort = idx[np.argsort(-scores_flat[idx])]
      s_idx, e_idx = np.unravel_index(idx_sort, scores.shape)
      pred_s.append(s_idx[0])
      pred_e.append(e_idx[0])
      pred_score.append(scores_flat[idx_sort][0])
    del score_s, score_e
    return pred_s, pred_e, pred_score

  @staticmethod
  def decode_candidates(score_s, score_e, candidates, top_n=1, max_len=None):
    """Take argmax of constrained score_s * score_e. Except only consider
    spans that are in the candidates list.
    """
    pred_s = []
    pred_e = []
    pred_score = []
    for i in range(score_s.size(0)):
      # Extract original tokens stored with candidates
      tokens = candidates[i]['input']
      cands = candidates[i]['cands']

      # if not cands:
      # 	# try getting from globals? (multiprocessing in pipeline mode)
      # 	from ..pipeline.wrmcqa import PROCESS_CANDS
      # 	cands = PROCESS_CANDS
      if not cands:
        raise RuntimeError('No candidates given.')

      # Score all valid candidates found in text.
      # Brute force get all ngrams and compare against the candidate list.
      max_len = max_len or len(tokens)
      scores, s_idx, e_idx = [], [], []
      for s, e in tokens.ngrams(n=max_len, as_strings=False):
        span = tokens.slice(s, e).untokenize()
        if span in cands or span.lower() in cands:
          # Match! Record its score.
          scores.append(score_s[i][s] * score_e[i][e - 1])
          s_idx.append(s)
          e_idx.append(e - 1)

      if len(scores) == 0:
        # No candidates present
        pred_s.append([])
        pred_e.append([])
        pred_score.append([])
      else:
        # Rank found candidates
        scores = np.array(scores)
        s_idx = np.array(s_idx)
        e_idx = np.array(e_idx)

        idx_sort = np.argsort(-scores)[0:top_n]
        pred_s.append(s_idx[idx_sort])
        pred_e.append(e_idx[idx_sort])
        pred_score.append(scores[idx_sort])
    del score_s, score_e, candidates
    return pred_s, pred_e, pred_score
