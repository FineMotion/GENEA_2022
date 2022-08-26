import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, frames_count, output_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.hidden = nn.Linear(input_dim * frames_count, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.hidden(x)
        x = torch.relu(x)
        x = self.out(x)
        x = torch.tanh(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_dim, frames_count, output_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.hidden = nn.Linear(output_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, input_dim * frames_count)

    def forward(self, x):
        x = self.hidden(x)
        x = torch.relu(x)
        x = self.out(x)
        x = (torch.tanh(x) + 1) / 2
        return x
