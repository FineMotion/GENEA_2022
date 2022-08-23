import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        super(Encoder, self).__init__()

        self.highway = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.rnn = nn.GRU(
            hidden_dim, hidden_dim, bidirectional=True, num_layers=num_layers, batch_first=False, dropout=0.2
        )
        self.dropout = nn.Dropout(0.1)
        self.hidden_dim = hidden_dim
        self.num_layer = num_layers

    def forward(self, x):
        x = self.highway(x)
        output, hidden = self.rnn(x)
        return self.dropout(output), self.dropout(hidden)


class Attention(nn.Module):
    def __init__(self, enc_dim: int, dec_dim: int):
        super(Attention, self).__init__()
        self.enc_to_dec = nn.Linear(enc_dim, dec_dim)

    def forward(self, encoder_states, decoder_hidden):
        encoder_states = self.enc_to_dec(encoder_states)
        encoder_states = encoder_states.transpose(0, 1)
        decoder_hidden = decoder_hidden.permute(1, 2, 0)
        weights = torch.matmul(encoder_states, decoder_hidden).squeeze(2)
        scores = torch.softmax(weights, dim=1).unsqueeze(2)
        encoder_states = encoder_states.transpose(1, 2)
        weighted_states = torch.matmul(encoder_states, scores)
        weighted_states = weighted_states.permute(2, 0, 1)
        return weighted_states


class Decoder(nn.Module):
    def __init__(
            self, output_dim: int, hidden_dim: int, enc_dim: int, max_gen: int = 30
    ):
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.enc_dec_linear = nn.Linear(enc_dim, hidden_dim)
        self.rnn = nn.GRU(output_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim * 2, output_dim)
        self.max_gen = max_gen
        self.output_dim = output_dim
        self.attention = Attention(enc_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, encoder_states: torch.Tensor, encoder_hidden, previous_poses=None, real_poses_len: int = None):
        seq_len, batch_size, enc_dim = encoder_states.shape
        dec_hidden = encoder_hidden
        dec_hidden = dec_hidden.view(-1, 2, batch_size, enc_dim // 2)
        dec_hidden = dec_hidden[-1:]
        dec_hidden.transpose(1, 2)
        dec_hidden = dec_hidden.reshape(1, batch_size, -1)
        dec_hidden = self.enc_dec_linear(dec_hidden)

        output, dec_hidden = self.rnn(previous_poses, dec_hidden)
        start_pose = previous_poses[-1:]

        if real_poses_len is not None:
            max_gen_len = real_poses_len
        else:
            max_gen_len = self.max_gen

        poses = []
        for i in range(max_gen_len):
            # dec_hidden = self.dropout(dec_hidden)dec_hidden
            output, dec_hidden = self.rnn(start_pose, dec_hidden)
            attention = self.attention(encoder_states, dec_hidden)
            concat = torch.cat((dec_hidden, attention), dim=2)
            concat = self.dropout(concat)
            start_pose = self.linear(concat)
            poses.append(start_pose)

        poses = torch.cat(poses, dim=0)
        return poses