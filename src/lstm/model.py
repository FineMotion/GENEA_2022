import torch
import torch.nn as nn


class LstmModel(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_size: int):
        super(LstmModel, self).__init__()
        self.highway = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.rnn = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                           num_layers=2, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2*hidden_size, out_features)

    def forward(self, x):
        x = self.highway(x)
        x, _ = self.rnn(x)
        x = self.out(x)
        return x