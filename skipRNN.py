from __future__ import absolute_import
from __future__ import print_function

import random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from util.misc import *
from util.graph_definition import *

from torch.optim.lr_scheduler import ReduceLROnPlateau

class cellModule(nn.Module):

    def __init__(self, cells, model, hidden_size, output_size):
        super(cellModule, self).__init__()
        self.model = model
        self.rnn = cells
        self.d1 = nn.Linear(hidden_size, output_size)

    def forward(self, input, hx=None):
        if hx is not None:
            output = self.rnn(input, hx)
        else:
            output = self.rnn(input)
        output, hx, updated_state = split_rnn_outputs(self.model, output)
        output = self.d1(output) # Get the last output of the sequence
        return output, hx, updated_state

def skipLSTM(input_size = 1024,
               hidden_size=300,
               output_size=2,
                num_layers = 3):
    cells = create_model(model = 'skip_lstm',
                    input_size = input_size,
                hidden_size = hidden_size,
                    num_layers = num_layers)
    return cellModule(cells, model='skip_lstm', hidden_size=hidden_size, output_size=output_size)
