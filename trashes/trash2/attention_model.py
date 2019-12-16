import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import math

class EncoderRNN(nn.Module):
    def __init__(self, input_size,  hidden_size, n_layers=1, dropout=0.5):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.gru = nn.GRU(input_size, hidden_size, n_layers, dropout=self.dropout, batch_first=True,bidirectional=True)

    def forward(self, input_seqs, hidden=None):
        
        self.gru.flatten_parameters()
        outputs, hidden = self.gru(input_seqs, hidden)
#         outputs = outputs[:, :, :self.hidden_size] + outputs[:, :,self.hidden_size:]  # Sum bidirectional outputs
        return outputs, hidden

class Attn(nn.Module):
    def __init__(self,method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs, src_len=None):
        '''
        :param hidden: 
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (B,T,H)
        :param src_len:
            used for masking. NoneType or tensor in shape (B) indicating sequence length
        :return
            attention energies in shape (B,T)
        '''
        this_batch_size = encoder_outputs.size(0)
        max_len = encoder_outputs.size(1)
        H = hidden.repeat(max_len,1,1).transpose(0,1) # [B*T*H]
        attn_energies = self.score(H,encoder_outputs) # compute attention score
#         print(hidden.shape)
        
        if src_len is not None:
            mask = []
            for b in range(src_len.size(0)):
                mask.append([0] * src_len[b].item() + [1] * (encoder_outputs.size(1) - src_len[b].item()))
            mask = cuda_(torch.ByteTensor(mask).unsqueeze(1)) # [B,1,T]
            attn_energies = attn_energies.masked_fill(mask, -1e18)
        
        return F.softmax(attn_energies,dim = -1).unsqueeze(1) # normalize with softmax

    def score(self, hidden, encoder_outputs):
#         print("score hidden :",hidden.shape)
#         print("encoder_outputs :",encoder_outputs.shape)
        energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2))) # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2,1) # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) #[B*1*H]
        energy = torch.bmm(v,energy) # [B*1*T]
        return energy.squeeze(1) #[B*T]

class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, n_layers=1, dropout_p=0.1):
        super(BahdanauAttnDecoderRNN, self).__init__()
        # Define parameters
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        # Define layers
        self.dropout = nn.Dropout(dropout_p)
        self.attn = Attn('concat', hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout_p)
        self.attn_combine = nn.Linear(hidden_size+hidden_size, hidden_size)
#         self.out = nn.Linear(hidden_size, output_size)
    
    def forward(self, word_input, last_hidden, encoder_outputs):
        '''
        :param word_input:
            word input for current time step, in shape (B)
        :param last_hidden:
            last hidden stat of the decoder, in shape (layers*direction*B*H)
        :param encoder_outputs:
            encoder outputs in shape (T*B*H)
        :return
            decoder output 
        Note: we run this one step at a time i.e. you should use a outer loop 
            to process the whole sequence
        Tip(update):
        EncoderRNN may be bidirectional or have multiple layers, so the shape of hidden states can be 
        different from that of DecoderRNN
        You may have to manually guarantee that they have the same dimension outside this function,
        e.g, select the encoder hidden state of the foward/backward pass.
        '''
        # Get the embedding of the current input word (last output word)
        word_input = self.dropout(word_input).squeeze(1).unsqueeze(0)## (1,B,H)
        # Calculate attention weights and apply to encoder outputs
#         print("last hidden:",last_hidden.shape)
        attn_weights = self.attn(last_hidden[-1], encoder_outputs)
#         print("attn weight shape:",attn_weights.shape)
#         context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,V)
        context = attn_weights.bmm(encoder_outputs)
        context = context.transpose(0,1)  # (1,B,2H)

        # Combine embedded input word and attended context, run through RNN
#         print("word input shape:",word_input.shape)
#         print("context shape:",context.shape)
        rnn_input = torch.cat((word_input, context), 2) ## (1,B,3H)
        rnn_input = self.attn_combine(rnn_input) # use it in case your size of rnn_input is different
        self.gru.flatten_parameters()
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,V)->(B,V)
        # context = context.squeeze(0)
        # update: "context" input before final layer can be problematic.
        # output = F.log_softmax(self.out(torch.cat((output, context), 1)))
#         output = F.log_softmax(self.out(output))
        # Return final output, hidden state
        return output, hidden