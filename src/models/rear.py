import torch
import torch.nn as nn


class RearLSTM(nn.Module):
    def __init__(self, input_size_sta, input_size_dyn, hidden_size, output_size, num_layers, drop_prob=0.5):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if num_layers == 1:
            self.lstm = nn.LSTM(input_size_dyn, hidden_size, num_layers, batch_first=True)
        else:
            self.lstm = nn.LSTM(input_size_dyn, hidden_size, num_layers,
                                dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc_sta = nn.Linear(input_size_sta, input_size_sta)
        self.fc = nn.Linear(hidden_size + input_size_sta, output_size)
        self.act = nn.ReLU()

    def forward(self, data):
        x_s, x_d = data

        # Forward propagate LSTM
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x_d)

        # Decode the hidden state of the last time step
        out = torch.cat([out[:, -1, :], self.fc_sta(x_s)], axis=-1)
        out = self.fc(self.dropout(out))

        # use specific activation
        out = self.act(out)
        return out
