import torch.nn as nn


class LSTMCustom(nn.Module):

    def __init__(self, embedding_dim, hidden_dim):
        super(LSTMCustom, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, sequence, hidden):
        lstm_out, _ = self.lstm(sequence, hidden)
        # todo fix size
        fc_out = self.fc(lstm_out)
        return fc_out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(1, batch_size, self.hidden_dim).zero_(),
                  weight.new(1, batch_size, self.hidden_dim).zero_())
        return hidden