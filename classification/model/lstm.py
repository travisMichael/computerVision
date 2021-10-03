import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(1)

lstm = nn.LSTM(3, 3)  # Input dim is 3, output dim is 3
inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5

# initialize the hidden state.
hidden = (torch.randn(1, 1, 3),
          torch.randn(1, 1, 3))
for i in inputs:
    # Step through the sequence one element at a time.
    # after each step, hidden contains the hidden state.
    out, hidden = lstm(i.view(1, 1, -1), hidden)

# alternatively, we can do the entire sequence all at once.
# the first value returned by LSTM is all of the hidden states throughout
# the sequence. the second is just the most recent hidden state
# (compare the last slice of "out" with "hidden" below, they are the same)
# The reason for this is that:
# "out" will give you access to all hidden states in the sequence
# "hidden" will allow you to continue the sequence and backpropagate,
# by passing it as an argument  to the lstm at a later time
# Add the extra 2nd dimension
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
out, hidden = lstm(inputs, hidden)
print(out)
print(hidden)

# --------------------------------------------------------------------------------


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


input_dim = 1
hidden_dim = 20
n_layers = 1
lstm = LSTMCustom(input_dim, hidden_dim)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=0.01)

train_features = [torch.tensor([[[1.]], [[2.]]]), torch.tensor([[[3.]], [[2.]], [[5.]]])]
train_labels = [torch.tensor(3.), torch.tensor(10.)]

# train_data = TensorDataset(torch.from_numpy(train_features), torch.from_numpy(train_labels))
# train_loader = DataLoader(train_data, shuffle=True, batch_size=1)
epochs = 10
counter = 0

lstm.train()
for i in range(epochs):
    h = lstm.init_hidden(1)

    for (input, label) in zip(train_features, train_labels):
        counter += 1
        hidden = tuple([e.data for e in h])
        # inputs, labels = inputs.to(device), labels.to(device)
        lstm.zero_grad()
        output = lstm(input, hidden)
        loss = criterion(output.squeeze(), label.float())
        print(loss)
        loss.backward()
        optimizer.step()


torch.save(lstm.state_dict(), './state_dict.pt')
print("done")
