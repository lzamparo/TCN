import torch
import numpy as np
from torch import nn
from torch.nn.utils import weight_norm


class DynamicKmaxPooling(nn.Module):
    def __init__(self, k_top, L, l):
        """ 
        k_top := smallest possible number of pooled elements 
        L := number of convolutional layers in the network
        l := the index of this pooling layer in the network """
        super(DynamicKmaxPooling, self).__init__()
        self.k_top = k_top
        self.L = L
        self.l = l
        
    def forward(self, x, dim=2):    
        s = x.size()[2]
        k_ll = ((self.L - self.l) / self.L) * s
        pool_size = round(max(self.k_top, int(np.ceil(k_ll))))
        index = x.topk(pool_size, dim)[1].sort(dim)[0]
        return x.gather(dim, index)
                

class WideConvBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, k_top, L, l):
        super(WideConvBlock, self).__init__()
        padding = kernel_size - 1 
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, 
                                           stride=stride, padding=padding))
        self.pool1 = DynamicKmaxPooling(k_top, L, l)
        self.act1 = nn.Tanh()
        self.net = nn.Sequential(self.conv1, self.pool1, self.act1)
        
    def init_weights(self):
        self.conv1.weight.data.normal(0, 0.01)
        
    def forward(self, x):
        out = self.net(x)
        return out

                
class DCNN(nn.Module):

    def __init__(self, embedding_size=60, vocab_size=60000, num_maps=[6,14],
                 kernel_sizes=[7,5], k_top=4, output_size=3, dropout=0.2, padding_idx=280000):
        """ DCNN of Denil et al. implemented from their paper
        embedding_size := embedding dimension size
        vocab_size := size of the vocabulary 
        num_maps := list, where ith entry is the number of feature maps in the ith layer 
        kernel_sizes := list, same size as num_channels, size of Conv1D kernel for that layer 
        k_top := the smallest size of temporal pooling window
        droput := the amount of dropout to apply to the penultimate layer 
        output_size := cardinality of the label space
        
        N.B: there is pooling, but size of pooling is determined by the size of the output of the 
        convolutional layers (c.f Denil et al.)
        """
        
        super(DCNN, self).__init__()
        assert(len(num_maps) == len(kernel_sizes))
        self.encoder = nn.Embedding(vocab_size, embedding_size,
                                    padding_idx=padding_idx)
     
        layers = []
        L = len(num_maps)
        for i in range(len(num_maps)):
            in_channels = embedding_size if i == 0 else num_maps[i-1]
            out_channels = num_maps[i]
            # n_inputs, n_outputs, kernel_size, stride, k_top, L, l
            layers += [WideConvBlock(n_inputs=in_channels,n_outputs=out_channels, 
                                     kernel_size=kernel_sizes[i], stride=1, k_top=k_top, 
                                     L=L, l=i)]
        
        self.network = nn.Sequential(*layers)
        flat_features = self._find_reshape_size()
        self.droplayer = nn.Dropout(p=dropout)
        self.decoder = nn.Linear(flat_features, output_size)
        self.init_weights()
        
    def _find_reshape_size(self):
        fixture = torch.randn(1,60,42)
        outsize = self.network(fixture)
        h, w = outsize.size()[1:]
        return h * w

    def init_weights(self):
        self.encoder.weight.data.normal_(0, 0.01)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, input):
        """Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)"""
        
        emb = self.encoder(input)
        emb = emb.transpose_(1,2)
        y_hat = self.network(emb)
        y_flat = y_hat.view(y_hat.size(0),-1)
        y_hat = self.decoder(self.droplayer(y_flat)) ## softmax needs to be applied here
        return nn.functional.softmax(y_hat)



