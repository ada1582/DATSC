# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
from torch.autograd import Variable

class TSCLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, output_dim=1, drop_prob=0.5):
        super(TSCLSTM, self).__init__()
        # input dimensions
        self.input_dim = input_dim
        # hidden dimensions
        self.hidden_dim = hidden_dim
        # number of hidden layers
        self.num_layers = num_layers
        
        # Define the LSTM layer
        # dropout is a good choise when number of layers is > 1, otherewise there is no dropout applied according to pytorch documentation
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, dropout=drop_prob, batch_first=True)
        
        # Define the output layer
        self.fc = nn.Linear(self.hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # One time step
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
         # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
         # Obtaining the last output (used for many to one architecture / Classification)
        out = self.fc(out[:, -1, :]) 
        out = self.sigmoid(out)
        
        return out
        
        
        #new_input = input.view(len(input), self.batch_size, -1)
        #lstm_out, self.hidden = self.lstm(new_input)
        #out = lstm_out[-1].view(self.batch_size, -1)
        
        #out = self.dropout(out)
        #out = self.fc(out)
        #out = self.sigmoid(out)
        
        #out = out.view(batch_size, -1)
        #out = out[:,-1]
        #return out, hidden

if __name__ == '__main__':
    input_dim = 1
    seq_len = 46 # sequence length
    layer_dim = 2  # ONLY CHANGE IS HERE FROM ONE LAYER TO TWO LAYER
    output_dim = 24 # number of classes 

    model = TSCLSTM(input_dim, seq_len, layer_dim, output_dim)

    # JUST PRINTING MODEL & PARAMETERS 
    print(model)
    print(len(list(model.parameters())))
    for i in range(len(list(model.parameters()))):
        print(list(model.parameters())[i].size())
    # Load images as a torch tensor with gradient accumulation abilities
    x = Variable(torch.randn(128, input_dim, seq_len)).view(-1, seq_len, input_dim).requires_grad_()
    print(x.shape)
    y = model(x)
    print(y.size())



        