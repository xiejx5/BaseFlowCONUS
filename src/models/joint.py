import torch
import torch.nn as nn


class JointLSTM(nn.Module):
    def __init__(self, input_size_sta, input_size_dyn, hidden_size, output_size, num_layers, drop_prob=0.5):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if num_layers == 1:
            self.lstm = nn.LSTM(input_size_dyn + input_size_sta, hidden_size, num_layers, batch_first=True)
        else:
            self.lstm = nn.LSTM(input_size_dyn + input_size_sta, hidden_size, num_layers,
                                dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_size, output_size)
        self.act = nn.ReLU()

    def forward(self, data):
        x_s, x_d = data
        x = torch.cat((x_d, x_s.repeat_interleave(x_d.shape[1], dim=0).reshape(*x_d.shape[:2], -1)), dim=-1)
        # Set initial hidden and cell states

        # Forward propagate LSTM
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x)

        # Decode the hidden state of the last time step
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)

        # use specific activation
        out = self.act(out)
        return out
