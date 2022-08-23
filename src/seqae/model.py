import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 40):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, bidirectional=True, batch_first=True)
        self.out = nn.Linear(2*hidden_size, hidden_size)

    def forward(self, x):
        output, hidden = self.gru(x)
        output = self.out(output)
        return torch.tanh(output)


class Decoder(nn.Module):
    def __init__(self, output_dim: int, hidden_size: int = 40):
        super().__init__()
        self.gru = nn.GRU(hidden_size, output_dim, bidirectional=True, batch_first=True)
        self.out = nn.Linear(2*output_dim, output_dim)

    def forward(self, x):
        output, hidden = self.gru(x)
        output = self.out(output)
        return output


