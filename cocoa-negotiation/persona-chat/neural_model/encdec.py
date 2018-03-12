from itertools import izip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb  # set_trace

from cocoa.pt_model.util import EPS, smart_variable, basic_variable
# from cocoa.model.sequence_embedder import AttentionRNNEmbedder, BoWEmbedder
from preprocess import markers

def add_model_arguments(parser):
    parser.add_argument('--dropout', type=float, default=0, help='Dropout rate')
    parser.add_argument('--attn-method', default='luong', type=str, help='type of attention', choices=['luong', 'dot', 'vinyals'])
    parser.add_argument('--word-embed-size', type=int, default=128, help='Word embedding and hidden state size')
    parser.add_argument('--num-layers', type=int, default=1, help="number of layers to stack the RNN")
    parser.add_argument('--teacher-forcing-ratio', default=0.7, type=float, help='teacher forcing ratio, 0 means no teacher forcing')

    # parser.add_argument('--num-context', default=0, type=int, help='Number of sentences to consider as dialogue context (in addition to the encoder input)')
    # parser.add_argument('--selector', action='store_true', help='Retrieval-based model (candidate selector)')
    # parser.add_argument('--selector-loss', default='binary', choices=['binary'], help='Loss function for the selector (binary: cross-entropy loss of classifying a candidate as the true response)')

class Bid_Attn_Decoder(nn.Module):
    '''
    During bi-directional encoding, we split up the word embedding in half
    and use then perform a forward pass into two directions.  In code,
    this is interpreted as 2 layers at half the size. Based on the way we
    produce the encodings, we need to merge the context vectors together in
    order properly init the hidden state, but then everything else is the same
    '''
    def __init__(self, hidden_size, method, drop_prob=0.1):
        super(Bid_Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = self.hidden_size * 2  # since we concat input and context
        self.dropout = nn.Dropout(drop_prob)
        self.expand_params = False
        # num_layers is removed since decoder always has one layer
        self.gru = nn.GRU(self.input_size, self.hidden_size) # dropout=drop_prob)
        self.attn = Attention(method, self.hidden_size)  # adds W_a matrix
        # we need "* 2" since we concat hidden state and attention context vector

    def create_embedding(self, embedding, vocab_size):
        self.embedding = embedding
        self.out = nn.Linear(self.hidden_size * 2, vocab_size)

    def forward(self, word_input, last_context, prev_hidden, encoder_outputs):
        if (prev_hidden.size()[0] == (2 * word_input.size()[0])):
            prev_hidden = prev_hidden.view(1, 1, -1)

        # Get the embedding of the current input word (i.e. last output word)
        embedded = self.embedding(word_input).view(1, 1, -1)        # 1 x 1 x N
        embedded = self.dropout(embedded)
        embedded = smart_variable(embedded, "var")
        # Combine input word embedding and previous hidden state, run through RNN
        rnn_input = torch.cat((embedded, last_context), dim=2)
        pdb.set_trace()
        rnn_output, current_hidden = self.gru(rnn_input, prev_hidden)

        # Calculate attention from current RNN state and encoder outputs, then apply
        # Drop first dimension to line up with single encoder_output
        decoder_hidden = current_hidden.squeeze(0)    # (1 x 1 x N) --> 1 x N
        attn_weights = self.attn(decoder_hidden, encoder_outputs)  # 1 x 1 x S
         # [1 x (1xS)(SxN)] = [1 x (1xN)] = 1 x 1 x N)   where S is seq_len of encoder
        attn_context = attn_weights.bmm(encoder_outputs.transpose(0,1))

        # Predict next word using the decoder hidden state and context vector
        joined_hidden = torch.cat((current_hidden, attn_context), dim=2).squeeze(0)
        output = F.log_softmax(self.out(joined_hidden), dim=1)  # (1x2N) (2NxV) = 1xV
        return output, attn_context, current_hidden, attn_weights

class Attn_Decoder(nn.Module):
    def __init__(self, hidden_size, method, drop_prob=0.1):
        super(Attn_Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = self.hidden_size * 2  # since we concat input and context
        self.dropout = nn.Dropout(drop_prob)
        self.expand_params = False

        self.gru = nn.GRU(self.input_size, self.hidden_size) # dropout=drop_prob)
        self.attn = Attention(method, self.hidden_size)  # adds W_a matrix

    def create_embedding(self, embedding, vocab_size):
        self.embedding = embedding
        self.out = nn.Linear(self.input_size, vocab_size)
        # we need "* 2" since we concat hidden state and attention context vector

    def forward(self, word_input, last_context, prev_hidden, encoder_outputs):
        # Get the embedding of the current input word (i.e. last output word)
        embedded = self.embedding(word_input).view(1, 1, -1)        # 1 x 1 x N
        embedded = self.dropout(embedded)
        embedded = smart_variable(embedded, "var")
        # Combine input word embedding and previous hidden state, run through RNN
        rnn_input = torch.cat((embedded, last_context), dim=2)
        # pdb.set_trace()
        rnn_output, current_hidden = self.gru(rnn_input, prev_hidden)

        # Calculate attention from current RNN state and encoder outputs, then apply
        # Drop first dimension to line up with single encoder_output
        decoder_hidden = current_hidden.squeeze(0)    # (1 x 1 x N) --> 1 x N
        attn_weights = self.attn(decoder_hidden, encoder_outputs)  # 1 x 1 x S
         # [1 x (1xS)(SxN)] = [1 x (1xN)] = 1 x 1 x N)   where S is seq_len of encoder
        attn_context = attn_weights.bmm(encoder_outputs.transpose(0,1))

        # Predict next word using the decoder hidden state and context vector
        joined_hidden = torch.cat((current_hidden, attn_context), dim=2).squeeze(0)
        output = F.log_softmax(self.out(joined_hidden), dim=1)  # (1x2N) (2NxV) = 1xV
        return output, attn_context, current_hidden, attn_weights

class LSTM_Decoder(nn.Module):
    def __init__(self, hidden_size, n_layers=1):
        super(LSTM_Decoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.expand_params = False

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=n_layers)

    def create_embedding(self, embedding, vocab_size):
        self.embedding = embedding
        self.out = nn.Linear(self.hidden_size, vocab_size)

    def forward(self, word_input, prev_hidden):
        embedded = self.embedding(word_input).view(1, 1, -1)
        embedded = smart_variable(embedded, "var")
        rnn_output, current_hidden = self.lstm(embedded, prev_hidden)
        output = F.log_softmax(self.out(rnn_output[0]), dim=1)
        return output, current_hidden

    def initHidden(self):
        return smart_variable(torch.zeros(1, 1, self.hidden_size))

class Attention(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attention, self).__init__()
        self.attn_method = method
        self.tanh = nn.Tanh()
        # the "_a" stands for the "attention" weight matrix
        if self.attn_method == 'luong':                # h(Wh)
            self.W_a = nn.Linear(hidden_size, hidden_size)
        elif self.attn_method == 'vinyals':            # v_a tanh(W[h_i;h_j])
            self.W_a =  nn.Linear(hidden_size * 2, hidden_size)
            self.v_a = nn.Parameter(torch.FloatTensor(1, hidden_size))
        elif self.attn_method == 'dot':                 # h_j x h_i
            self.W_a = torch.eye(hidden_size) # identity since no extra matrix is needed

    def forward(self, decoder_hidden, encoder_outputs):
        # Create variable to store attention scores           # seq_len = batch_size
        seq_len = len(encoder_outputs)
        attn_scores = smart_variable(torch.zeros(seq_len))    # B (batch_size)
        # Calculate scores for each encoder output
        for i in range(seq_len):           # h_j            h_i
                attn_scores[i] = self.score(decoder_hidden, encoder_outputs[i]).squeeze(0)
        # Normalize scores into weights in range 0 to 1, resize to 1 x 1 x B
        attn_weights = F.softmax(attn_scores, dim=0).unsqueeze(0).unsqueeze(0)
        return attn_weights

    def score(self, h_dec, h_enc):
        W = self.W_a
        if self.attn_method == 'luong':                # h(Wh)
            return h_dec.matmul( W(h_enc).transpose(0,1) )
        elif self.attn_method == 'vinyals':            # v_a tanh(W[h_i;h_j])
            hiddens = torch.cat((h_enc, h_dec), dim=1)
            # Note that W_a[h_i; h_j] is the same as W_1a(h_i) + W_2a(h_j) since
            # W_a is just (W_1a concat W_2a)             (nx2n) = [(nxn);(nxn)]
            return self.v_a.matmul(self.tanh( W(hiddens).transpose(0,1) ))
        elif self.attn_method == 'dot':                # h_j x h_i
            return h_dec.matmul(h_enc.transpose(0,1))

class Bid_Encoder(nn.Module):
    def __init__(self, hidden_size, n_layers=1):
        super(Bid_Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size // 2, \
                num_layers=n_layers, bidirectional=True)

    def create_embedding(self, vocab_size):
        self.embedding = nn.Embedding(vocab_size, self.hidden_size)
        return self.embedding

    def forward(self, word_inputs, hidden):
        seq_len = len(word_inputs)  # now a matrix multiplication
        embedded = self.embedding(word_inputs).view(seq_len, 1, -1)
        embedded = smart_variable(embedded, "var")
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return smart_variable(torch.zeros(2, 1, self.hidden_size // 2))

class GRU_Encoder(nn.Module):
    def __init__(self, hidden_size, n_layers=1):
        super(GRU_Encoder, self).__init__()
        self.hidden_size = hidden_size # dim of object passed into IFOG gates
        self.input_size = hidden_size # serves double duty
        self.gru = nn.GRU(self.input_size, self.hidden_size, num_layers=n_layers)

    def create_embedding(self, vocab_size):
        self.embedding = nn.Embedding(vocab_size, self.hidden_size)
        return self.embedding

    def forward(self, word_inputs, hidden):
        seq_len = len(word_inputs)
        embedded = self.embedding(word_inputs).view(seq_len, 1, -1)
        embedded = smart_variable(embedded, "var")
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return smart_variable(torch.zeros(1, 1, self.hidden_size))

class LSTM_Encoder(nn.Module):
    def __init__(self, hidden_size, n_layers=1):
        super(LSTM_Encoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(hidden_size, hidden_size)

    def create_embedding(self, vocab_size):
        self.embedding = nn.Embedding(vocab_size, self.hidden_size)
        return self.embedding

    def forward(self, word_inputs, hidden):
        embedded = self.embedding(word_inputs).view(1, 1, -1)
        embedded = smart_variable(embedded, "var")
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def initHidden(self):
        hidden = smart_variable(torch.zeros(1, 1, self.hidden_size))
        cell = smart_variable(torch.zeros(1, 1, self.hidden_size))
        return (hidden, cell)