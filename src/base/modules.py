import torch
import torch.nn as nn


class LinearWithBatchNorm(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, context_size: int = 61):
        super(LinearWithBatchNorm, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(context_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.batch_norm(x)
        x = torch.relu(x)
        x = self.dropout(x)
        return x