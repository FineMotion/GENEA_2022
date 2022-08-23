import torch
import torch.nn as nn

from src.base.modules import LinearWithBatchNorm


class FeedforwardModel(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_size: int):
        super(FeedforwardModel, self).__init__()
        self.highway = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.rnn = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.out = nn.Linear(hidden_size, out_features)

    def forward(self, x):
        x = self.highway(x)
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        x = self.out(x)
        return x


class FeedforwardLargeModel(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_size: int = 150):
        super(FeedforwardLargeModel, self).__init__()
        self.ff_1 = LinearWithBatchNorm(in_features, hidden_size)
        self.ff_2 = LinearWithBatchNorm(hidden_size, hidden_size)
        self.ff_3 = LinearWithBatchNorm(hidden_size, hidden_size)

        self.gru = nn.GRU(hidden_size, hidden_size, 1, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(hidden_size, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ff_1(x)  # batch_size, context_dim, input_dim
        x = self.ff_2(x)  # batch_size, context_dim, hidden_dim
        x = self.ff_3(x)  # batch_size, context_dim, hidden_dim

        x = self.gru(x, None)[0][:, -1, :]    # batch_size, hidden_dim
        x = self.batch_norm(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out(x)
        return x