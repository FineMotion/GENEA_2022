import torch
import torch.nn as nn
# from src.base.modules import LinearWithBatchNorm


class GRUEncoder(nn.Module):
    """

    Args:
        in_features:
        hidden_size:
        dropout:
        encoder_layers:
        bidirectional:
    """
    def __init__(self, in_features: int, hidden_size: int, dropout: float = 0.2,
                 encoder_layers: int = 2, bidirectional: bool = True):

        super(GRUEncoder, self).__init__()
        self.highway = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.rnn = nn.GRU(
            hidden_size, hidden_size, bidirectional=bidirectional, num_layers=encoder_layers, batch_first=True,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x - (batch_size, seq_len, in_features)
        x = self.highway(x)   # (batch_size, seq_len, hidden_size)
        out, hidden = self.rnn(x)
        # out - (batch_size, seq_len, num_directions * hidden_size
        # hidden - (num_directions * num_layers, batch_size, hidden_size)
        return self.dropout(out), self.dropout(hidden)


class Attention(nn.Module):
    def __init__(self, enc_dim: int, dec_dim: int):
        super(Attention, self).__init__()
        self.enc_to_dec = nn.Linear(enc_dim, dec_dim)

    def forward(self, encoder_states, decoder_hidden):
        # encoder_states - batch_size, enc_len, enc_dim
        encoder_states = self.enc_to_dec(encoder_states)  # batch_size, enc_len, dec_dim
        # decoder_hidden - 1, batch_size, dec_dim
        decoder_hidden = decoder_hidden.permute(1, 2, 0)  # batch_size, dec_dim, 1
        weights = torch.matmul(encoder_states, decoder_hidden)  # batch_size, enc_len, 1
        weights = weights.squeeze(2)  # batch_size, enc_len
        scores = torch.softmax(weights, dim=1).unsqueeze(2) # batch_size, enc_len, 1

        encoder_states = encoder_states.transpose(1, 2)  # batch_size, dec_dim, enc_len
        weighted_states = torch.matmul(encoder_states, scores)  # batch_size, dec_dim, 1
        weighted_states = weighted_states.transpose(1, 2)  # batch_size, 1, dec_dim
        return weighted_states


class GRUAttentionDecoder(nn.Module):
    def __init__(self, enc_hidden: int, out_features: int,
                 hidden_dim: int, dropout: float = 0.5, seq_len: int = 30, bottle_neck: int = 40,
                 encoder_directions: int = 2, encoder_layers: int = 2):
        super(GRUAttentionDecoder, self).__init__()
        num_directions = encoder_directions
        enc_dim = enc_hidden * num_directions * encoder_layers
        self.enc_to_dec = nn.Linear(enc_dim, hidden_dim)
        self.out_features = out_features
        self.dec_bottle_neck = nn.Sequential(
            nn.Linear(out_features, bottle_neck),
            nn.ReLU(),
            nn.Linear(bottle_neck, hidden_dim),
            nn.Dropout(dropout)
        )
        self.rnn = nn.GRU(hidden_dim, hidden_dim, bidirectional=False, batch_first=True)
        self.attention = Attention(num_directions * enc_hidden, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim * 2, out_features)
        self.seq_len = seq_len

    def forward(self, encoder_states: torch.Tensor, encoder_hidden, prev_frames):
        # encoder_states - batch_size, seq_len, enc_hidden * num_directions
        # encoder_hidden - num_layers * num_directions, batch_size, enc_hidden

        # step 1. initialize decoder hidden
        batch_size, seq_len, enc_dim = encoder_states.shape
        decoder_hidden = encoder_hidden  # num_layers * num_directions, batch_size, enc_hidden
        decoder_hidden = decoder_hidden.unsqueeze(0)  # 1, num_layers * num_directions, batch_size, enc_hidden
        decoder_hidden = decoder_hidden.transpose(1, 2)  # 1, batch_size, num_layers * num_directions, enc_hidden
        decoder_hidden = decoder_hidden.reshape(1, batch_size, -1)
        # decoder_hidden - 1, batch_size, num_layers * num_directions * enc_hidden
        decoder_hidden = self.enc_to_dec(decoder_hidden)  # decoder_hidden - 1, batch_size, dec_hidden

        # step 2. update decoder hidden by previous frames
        # prev_frames - batch_size, seq_len, out_features
        history = self.dec_bottle_neck(prev_frames[:, :-1, :])
        _, decoder_hidden = self.rnn(history, decoder_hidden)

        # step 3. generate new frames
        # Assumption: encoder and decoder sequences have equal lengths
        # in_frame = torch.zeros(batch_size, 1, self.out_features, device=encoder_states.device)
        in_frame = prev_frames[:, -1:, :]
        out_frames = []

        for _ in range(self.seq_len):
            in_frame = self.dec_bottle_neck(in_frame)
            out, decoder_hidden = self.rnn(in_frame, decoder_hidden)
            attention = self.attention(encoder_states, decoder_hidden)  # batch_size, 1, dec_dim

            # concat out and attention
            concat = torch.cat((out, attention), dim=2)
            concat = self.dropout(concat)
            out_frame = self.out(concat)
            out_frames.append(out_frame)
            in_frame = out_frame
            # project to out features

        out_frames = torch.cat(out_frames, dim=1)  # batch_size, seq_len, out_features
        return out_frames
