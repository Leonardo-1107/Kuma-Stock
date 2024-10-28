import torch
import torch.nn as nn
import torch.nn.functional as F


class ICLoss(nn.Module):
    def __init__(self):
        super(ICLoss, self).__init__()

    def forward(self, y_pred, y_true):
        y_pred = y_pred.squeeze()
        y_true = y_true.squeeze()
        
        pred_mean = torch.mean(y_pred)
        true_mean = torch.mean(y_true)
        
        numerator = torch.sum((y_pred - pred_mean) * (y_true - true_mean))
        denominator = torch.sqrt(torch.sum((y_pred - pred_mean)**2) * torch.sum((y_true - true_mean)**2))
        
        ic = numerator / denominator
        
        # To make it a loss function, we want to maximize IC (which means minimizing -IC)
        return -ic


# set a simple gru model for time sequence prediction
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # the shape of input x: [batch_size, seq_len, input_size]
        out, _ = self.gru(x)
        # get the output of the last generation of time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# an improvement of GRU, it applies recurrent layers of RNN, similer to the MLP structure
class DRNNModel(nn.Module):
    """
    hidden_size(int, default 128): the size of the linear net
    num_layers(int, default 2): num of layers of multiple RNN
    dropout_rate(float, default 0.2): the rate of dropout layer
    
    """
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1, dropout_rate = 0.2, unit='GRU', is_attention=False):
        super(DRNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.drop_out = nn.Dropout(p=dropout_rate)
        # Define the recurrent layers
        if unit == 'LSTM':
            self.rnn_layers = nn.ModuleList([
                nn.LSTM(input_size if i == 0 else hidden_size, hidden_size, batch_first=True)
                for i in range(num_layers)
            ])

        elif unit == 'RNN':
            self.rnn_layers = nn.ModuleList([
                nn.RNN(input_size if i == 0 else hidden_size, hidden_size, batch_first=True)
                for i in range(num_layers)
            ])

        else:
            self.rnn_layers = nn.ModuleList([
                nn.GRU(input_size if i == 0 else hidden_size, hidden_size, batch_first=True)
                for i in range(num_layers)
            ])

        self.attention = SelfAttention(hidden_size)
        self.attention_triger = is_attention
        # the output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden states
        batch_size, seq_len, _ = x.size()
        h_0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        out = torch.zeros(batch_size, seq_len, self.fc.out_features).to(x.device)
        
        # Deep-attention-RNN net
        for i, rnn_layer in enumerate(self.rnn_layers):
            x, h_0 = rnn_layer(x, h_0)
            if i==0 & self.attention_triger:
                x = self.attention(x)
            x = self.drop_out(x)            
        
        # Return last steo of the final output     
        out[:, -1, :] = self.fc(x[:, -1, :])
         
        return out[:, -1, :]



class SelfAttention(nn.Module):
    """
    Self attention component
    """
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, hidden_size]
        query = self.W_q(x)
        key = self.W_k(x)
        value = self.W_v(x)
        
        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float32))
        attention_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attention_weights, value)
        
        return out
    

