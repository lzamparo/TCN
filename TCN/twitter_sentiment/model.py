import torch
from torch import nn

class DCNN(nn.Module):

    def __init__(self, embedding_size=60, vocab_size=60000, num_channels=[6,14],
                 kernel_sizes=[7,5]):
        """ DCNN of Denil et al. implemented from their paper
        embedding_size := embedding dimension size
        vocab_size := size of the vocabulary 
        num_channels := list, where ith entry is the number of feature maps in the ith layer 
        kernel_sizes := list, same size as num_channels, size of Conv1D kernel for that layer 
        
        N.B: there is pooling, but size of pooling is determined by the size of the output of the 
        convolutional layers (c.f Denil et al.)
        """
        
        super(DCNN, self).__init__()
        assert(len(num_channels) == len(kernel_sizes))
        self.encoder = nn.Embedding(output_size, input_size)
        #self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)  
        ### fill this with the appropriate DCNN-style model

        self.decoder = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.encoder.weight.data.normal_(0, 0.01)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, input):
        """Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)"""
        emb = self.encoder(input)
        y = self.tcn(emb.transpose(1, 2)).transpose(1, 2)
        y = self.decoder(y)
        return y.contiguous()    

class TCN(nn.Module):

    def __init__(self, input_size, output_size, num_channels,
                 kernel_size=2, dropout=0.3, emb_dropout=0.1, tied_weights=False):
        super(TCN, self).__init__()
        self.encoder = nn.Embedding(output_size, input_size)
        #self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)  ## fill this with the appropriate DCNN-style model

        self.decoder = nn.Linear(num_channels[-1], output_size)
        if tied_weights:
            if num_channels[-1] != input_size:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
            print("Weight tied")
        self.drop = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout
        self.init_weights()

    def init_weights(self):
        self.encoder.weight.data.normal_(0, 0.01)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, input):
        """Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)"""
        emb = self.drop(self.encoder(input))
        y = self.tcn(emb.transpose(1, 2)).transpose(1, 2)
        y = self.decoder(y)
        return y.contiguous()

