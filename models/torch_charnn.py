# Copyright (c) 2018-present, Ahmed H. Al-Ghidani.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

__author__ = "Ahmed H. Al-Ghidani"
__copyright__ = "Copyright 2018, The nlpeus Project, https://github.com/AhmedHani/nlpeus"
__license__ = "GPL"
__maintainer__ = "Ahmed H. Al-Ghidani"
__email__ = "ahmed.hani.ibrahim@gmail.com"

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim


class CharRNN(nn.Module):

    def __init__(self, input_size, output_size, hidden_size=512, model="lstm", n_layers=1, device='cpu'):
        super().__init__()

        self.model = model.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True)

        self.decoder = nn.Linear(hidden_size, output_size)

        self.loss = nn.NLLLoss()
        self.optimzer = optim.Adam(self.parameters(), lr=1e-3)

        self.device = device
        torch.device(device)

        self.to(device)

    def forward(self, input):
        input = torch.LongTensor(input).to(self.device)
        batch_size = input.size(0)

        encoded = self.encoder(input)
        output, hidden = self.rnn(encoded, self.init_hidden(batch_size))

        output = self.decoder(hidden[0].view(hidden[0].size(1), hidden[0].size(2)))
        output = F.log_softmax(output, dim=1)

        return output, hidden

    def predict_classes(self, input):
        input = torch.LongTensor(input).to(self.device)

        batch_size = input.size(0)

        encoded = self.encoder(input)
        output, hidden = self.rnn(encoded, self.init_hidden(batch_size))

        output = self.decoder(hidden[0].view(hidden[0].size(1), hidden[0].size(2)))
        output = F.log_softmax(output, dim=1)

        return output.max(1)[1].data.numpy()

    def predict_probs(self, input):
        input = torch.LongTensor(input).to(self.device)
        batch_size = input.size(0)

        encoded = self.encoder(input)
        output, hidden = self.rnn(encoded, self.init_hidden(batch_size))

        output = self.decoder(hidden[0].view(hidden[0].size(1), hidden[0].size(2)))
        output = F.softmax(output, dim=1)

        return output.max(1)[0].data.numpy()

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        return (Variable(weight.new(self.n_layers, batch_size, self.hidden_size).zero_()),
                Variable(weight.new(self.n_layers, batch_size, self.hidden_size).zero_()))

    def loss_function(self, predicted, target):
        target = torch.LongTensor(target)

        return self.loss(predicted, target)

    def optimize(self):
        return self.optimzer.step()

    def calculate_gradient(self, prediction, target):
        self.zero_grad()
        loss = self.loss_function(prediction, target)

        loss_value = loss.item()
        loss.backward()

        return float(loss_value)

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

        return True

    def load_weights(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))

        return True