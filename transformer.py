import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformer from scratch
class PositionalEmbedding(nn.Module):
    def __init__(self,seq_len, d_model):
        super().__init__()
        self.T = seq_len
        self.d_model = d_model

    def forward(self):
        even_i = torch.arange(0, self.d_model, 2).float()
        odd_i = torch.arange(1, self.d_model, 2).float() 
        denominator = torch.pow(10000,(even_i/self.d_model))
        pos = torch.arange(0, self.T).float().unsqueeze(1)
        odd_pos = torch.cos(pos/denominator)
        even_pos = torch.sin(pos/denominator)
        stacked = torch.stack([even_pos, odd_pos], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)

        return PE


# we are not using BPE, we are using character level tokenization here
class TransformerEmbedding(nn.Module):
    '''
        It tokenise the sentence and then add token, positional emebedding to it
    '''

    def __init__(self, max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN, dropout_ratio = 0.1):
        super().__init__()
        self.vocab_size = len(language_to_index)             # language_to_index is a dictionary
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.language_to_index = language_to_index
        self.position_encoder = PositionalEmbedding(max_sequence_length, d_model)
        self.dropout = nn.Dropout(dropout_ratio)
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN
    
    def batch_tokenize(self, batch, start_token=True, end_token=True):

        def tokenize(sentence, start_token=True, end_token=True):

            sentence_word_indicies = [self.language_to_index[token] for token in list(sentence)]
            # start token
            if start_token:
                sentence_word_indicies.insert(0, self.language_to_index[self.START_TOKEN])
            # end token
            if end_token:
                sentence_word_indicies.append(self.language_to_index[self.END_TOKEN])
            # padding token
            for _ in range(len(sentence_word_indicies), self.max_sequence_length):
                sentence_word_indicies.append(self.language_to_index[self.PADDING_TOKEN])

            sentence_word_indicies = sentence_word_indicies[0:self.max_sequence_length]

            return torch.tensor(sentence_word_indicies)
            

        tokenized = []
        for sentence_num in range(len(batch)):
           tokenized.append( tokenize(batch[sentence_num], start_token, end_token) )
        tokenized = torch.stack(tokenized)
        return tokenized.to(device)
    
    def forward(self, x,start_token = True, end_token=True): 
        # x: batch of sentences
        x = self.batch_tokenize(x ,start_token, end_token)
        x = self.embedding(x)
        pos = self.position_encoder().to(device)
        x = self.dropout(x + pos)
        return x


class LayerNorm(torch.nn.Module):
    def __init__(self, features, eps=1e-5):
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(features))           # scale parameter
        self.beta = torch.nn.Parameter(torch.zeros(features))           # shift parameter
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = ((x - mean) ** 2).mean(-1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta
    
class MultiHeadSA(nn.Module):
    # constructor
    def __init__(self, n_heads, d_model, input_dim):  
        super().__init__()    
        assert d_model % n_heads == 0 , "Invalid head_size for the given d_model"
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_size = d_model // n_heads
        self.input_dim = input_dim
        self.qkv_proj = nn.Linear(input_dim, 3 * d_model)
        self.linear = nn.Linear(d_model, d_model)
    
    def forward(self, X, mask = None):
        B, T, C = X.shape
        assert C == self.input_dim, "Input dimension does not match the model input dimension"
        qkv = self.qkv_proj(X)                                    # (B,T,3*D)
        qkv = qkv.reshape(B, T, self.n_heads, 3 * self.d_model // self.n_heads)
        qkv = qkv.permute(0,2,1,3)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        if mask is None:
            attention_score = torch.softmax(q @ k.transpose(-2, -1) / (self.head_size ** 0.5), dim=-1)
        else:
            mask = mask.unsqueeze(1)  # for broadcasting
            attention_score = torch.softmax(q @ k.transpose(-2, -1) / (self.head_size ** 0.5) + mask, dim=-1)
        res = attention_score @ v                                       # (B,H,T,head_size)
        res = res.permute(0,2,1,3).reshape(B, T, self.d_model)   
        res = self.linear(res)

        return res               


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_ratio = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x



# Encoder Layer

class EncoderLayer(nn.Module):

    def __init__(self, input_dim, d_model, n_heads, d_ff, dropout_ratio = 0.1):

        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.input_dim = input_dim
        self.multi_head_sa = MultiHeadSA(n_heads, d_model, input_dim)
        self.ln1 = LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.ln2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x, mask = None):

        res = self.multi_head_sa(x, mask)          # multi head attention
        res = self.dropout(res)
        res = self.ln1(res + x)                    # add and norm               
        res2 = self.feed_forward(res)               # feed forward
        res2 = self.dropout(res2)
        out = self.ln2(res2 + res)                  # add and norm
        
        return out
    

# Encoder

class Encoder(nn.Module):
    
    def __init__(self, d_model, ffn, n_heads, drop_ratio , n_layers, max_sequence_length, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        
        super().__init__()
        # embedding size = d_model in transformer paper (hardcoded positional embedding works then)
        self.embedding = TransformerEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN, drop_ratio)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, d_model, n_heads, ffn, drop_ratio) for _ in range(n_layers)])

    def forward(self, x, start_token = True, end_token=True, mask = None):
        x = self.embedding(x, start_token, end_token)
        for layer in self.encoder_layers:
            x = layer(x, mask)
        return x
    


# MultiHead Cross Attention
# between encoder and decoder

class MultiHeadCA(nn.Module):
    
    def __init__(self,d_model, n_heads):
        assert d_model%n_heads == 0, "Invalid head size for the given d_model"
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_size = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.linear = nn.Linear(d_model, d_model)

    def forward(self,x_enc, x_dec,mask = None):
        B, T, C = x_enc.shape               # they usually share same dimension
        kv = self.kv_proj(x_enc)
        q = self.q_proj(x_dec)
        kv = kv.reshape(B, -1, self.n_heads, 2 * self.head_size)
        kv = kv.permute(0,2,1,3)                                  # (B,nH,T,2*head_size)
        q = q.reshape(B, -1, self.n_heads, self.head_size)        # (B,T,nH,head_size)
        q = q.permute(0,2,1,3)
        k, v = torch.chunk(kv, 2, dim=-1)
        if mask is None:
            attention_score = torch.softmax(q @ k.transpose(-2, -1) / (self.head_size ** 0.5), dim=-1)
        else:
            mask = mask.unsqueeze(1)  # for broadcasting
            attention_score = torch.softmax(q @ k.transpose(-2, -1) / (self.head_size ** 0.5) + mask, dim=-1)
        res = attention_score @ v
        res = res.permute(0,2,1,3).reshape(B, T, self.d_model)
        res = self.linear(res)

        return res
    


# Decoder Layer

class Decoder_Layer(nn.Module):

    def __init__(self, input_dim, d_model, n_heads, d_ff, dropout_ratio = 0.1):
        super().__init__()
        self.multi_head_sa1 = MultiHeadSA(n_heads, d_model, input_dim)
        self.ln1 = LayerNorm(d_model)
        self.multi_head_ca = MultiHeadCA(d_model, n_heads)
        self.ln2 = LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.ln3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x_enc, x_dec, self_attention_mask = None, cross_attention_mask = None):

        res = self.multi_head_sa1(x_dec, self_attention_mask)          # masked self attention
        res = self.dropout(res)
        res = self.ln1(res + x_dec)                                      # add and norm
        res2 = self.multi_head_ca(x_enc, res, cross_attention_mask)      # cross attention
        res2 = self.dropout(res2)
        res2 = self.ln2(res2 + res)                                       # add and norm
        res3 = self.feed_forward(res2)                                    # feed forward
        res3 = self.dropout(res3)
        out = self.ln3(res3 + res2)                                       # add and norm

        return out
    
class Decoder(nn.Module):

    def __init__(self, d_model, ffn, n_heads, drop_ratio, n_layers, max_sequence_length, language_to_index,START_TOKEN,END_TOKEN, PADDING_TOKEN):
        
        super().__init__()
        # embedding size = d_model in transformer paper (hardcoded positional embedding works then)
        self.embedding = TransformerEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN, drop_ratio)
        self.decoder_layers = nn.ModuleList([Decoder_Layer(d_model, d_model, n_heads, ffn, drop_ratio) for _ in range(n_layers)])

    def forward(self, x_enc, x_dec, start_token = True, end_token=True, self_attention_mask = None, cross_attention_mask = None):

        x_dec = self.embedding(x_dec, start_token, end_token)
        for layer in self.decoder_layers:
            x_dec = layer(x_enc, x_dec, self_attention_mask, cross_attention_mask)
        return x_dec
    

# Transformer

class Transformer(nn.Module):

    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers,max_sequence_length,
                 language1_to_index, language2_to_index, START_TOKEN,END_TOKEN, PADDING_TOKEN):
        
        super().__init__()
        # language1 to language2
        self.encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers,max_sequence_length,language1_to_index,START_TOKEN,END_TOKEN, PADDING_TOKEN)
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers,max_sequence_length,language2_to_index,START_TOKEN,END_TOKEN, PADDING_TOKEN)
        self.lin_map = nn.Linear(d_model, len(language2_to_index))

    def forward(self,x,y,encoder_self_attention_mask = None, decoder_self_attention_mask = None, decoder_cross_attention_mask = None, enc_start_token= False, enc_end_token = False, dec_start_token = True, dec_end_token=True):

        x = self.encoder(x, start_token=enc_start_token, end_token=enc_end_token, mask = encoder_self_attention_mask)
        out = self.decoder(x, y, self_attention_mask = decoder_self_attention_mask, cross_attention_mask = decoder_cross_attention_mask, start_token=dec_start_token, end_token=dec_end_token)
        out = self.lin_map(out)
        return out
