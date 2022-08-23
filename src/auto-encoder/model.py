import torch
import torch.nn as nn

HIDDEN_DIM = 512
INPUT_DIM = 164
OUTPUT_DIM = 60


class Encoder(nn.Module):
    def __init__(self, frames_count=3):
        super(Encoder, self).__init__()
        self.hidden_1 = nn.Linear(INPUT_DIM * frames_count, HIDDEN_DIM)
        self.out = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)

    def forward(self, x):
        x = self.hidden_1(x)
        x = torch.relu(x)
        x = self.out(x)
        x = torch.tanh(x)
        return x


class Decoder(nn.Module):
    def __init__(self, frames_count=3):
        super(Decoder, self).__init__()
        self.hidden_2 = nn.Linear(OUTPUT_DIM, HIDDEN_DIM)
        self.out = nn.Linear(HIDDEN_DIM, INPUT_DIM * frames_count)

    def forward(self, x):
        x = self.hidden_2(x)
        x = torch.relu(x)
        x = self.out(x)
        x = (torch.tanh(x) + 1) / 2
        return x
