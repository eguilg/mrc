import torch.nn as nn

from .backbones import R_Net, MnemonicReader, BiDAF
from .layers.pointer_layer import PointerNetwork, MemoryAnsPointer

RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}
CELL_TYPES = {'lstm': nn.LSTMCell, 'gru': nn.GRUCell, 'rnn': nn.RNNCell}
BACKBONE_TYPES = {'r_net': R_Net, 'm_reader': MnemonicReader, 'bi_daf': BiDAF}
POINTER_TYPES = {'ptr': PointerNetwork, 'm_ptr': MemoryAnsPointer}
