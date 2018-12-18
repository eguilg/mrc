from pytorch_pretrained_bert.modeling import PreTrainedBertModel
from pytorch_pretrained_bert import BertModel
import torch
import torch.nn as nn
from models.layers.outer_layer import OuterNet
from models.losses.rouge_loss import RougeLoss


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
    attention_mask=attention_mask.byte()
    out = self.out_layer(sequence_output, sequence_output, attention_mask, attention_mask)

    if rouge_matrix is not None:
      return self.criterion(out, rouge_matrix)
    else:
      return out
