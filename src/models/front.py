import torch.nn as nn


class FrontLSTM(nn.Module):
    def __init__(self, input_size_sta, input_size_dyn, hidden_size, output_size, num_layers, drop_prob=0.5):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.fc_h = nn.Linear(input_size_sta, num_layers * hidden_size)
        self.fc_c = nn.Linear(input_size_sta, num_layers * hidden_size)
        if num_layers == 1:
            self.lstm = nn.LSTM(input_size_dyn, hidden_size, num_layers, batch_first=True)
        else:
            self.lstm = nn.LSTM(input_size_dyn, hidden_size, num_layers,
                                dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_size, output_size)
        self.act = nn.ReLU()

    def forward(self, x_s, x_d):
        # Set initial hidden and cell states
        h0 = self.act(self.fc_h(x_s))
        h0 = h0.reshape(h0.shape[0], self.num_layers, -1).transpose(0, 1).contiguous()

        c0 = self.act(self.fc_c(x_s))
        c0 = c0.reshape(c0.shape[0], self.num_layers, -1).transpose(0, 1).contiguous()

        # Forward propagate LSTM
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x_d, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)

        # use specific activation
        out = self.act(out)
        return out
