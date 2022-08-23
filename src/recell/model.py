import torch
import torch.nn as nn
from src.base.modules import LinearWithBatchNorm


class ReCellModel(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_size: int = 150, bottle_neck: int = 40,
                 window_size: int=61):
        super(ReCellModel, self).__init__()
        self.highway = nn.Sequential(
            LinearWithBatchNorm(in_features, hidden_size, context_size=window_size),
            LinearWithBatchNorm(hidden_size, hidden_size, context_size=window_size),
            LinearWithBatchNorm(hidden_size, hidden_size, context_size=window_size)
        )

        self.encoder = nn.GRU(hidden_size, hidden_size, 1, batch_first=True)
        self.hidden_highway = nn.Sequential(
            nn.Linear(out_features, bottle_neck),
            nn.ReLU(),
            nn.Linear(bottle_neck, hidden_size),
            nn.Dropout(0.5)
        )
        self.batch_norm = nn.BatchNorm1d(2*hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(2*hidden_size, out_features)

    def forward(self, x, h):
        x = self.highway(x)
        x = self.encoder(x, None)[0][:, -1, :]  # batch_size, hidden_dim

        h = self.hidden_highway(h)  # batch_size, hidden_dim
        x = torch.cat([x, h], dim=-1)  # batch_size, 2*hidden_dim
        x = self.batch_norm(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out(x)
        return x
