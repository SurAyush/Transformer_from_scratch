from transformer import Transformer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import numpy as np

# hyper-parameters
batch_size = 384
n_model = 256
n_layers = 2
n_heads = 4
drop_prob = 0.1
max_sequence_length = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Data-Source
eng_data = 'data/eng.txt'
beng_data = 'data/beng.txt'


# Load data
with open(eng_data, mode='r', encoding='utf-8') as file:
    sentences = file.readlines()

sentences = [line.strip() for line in sentences]

with open(beng_data, mode='r', encoding='utf-8') as file:
    sentences_beng = file.readlines()

sentences_beng = [line.strip() for line in sentences_beng]


# Preprocess data
eng_corpus = "".join(sentences)
eng_corpus = eng_corpus.lower()
eng_unique_characters = sorted(set(eng_corpus))
eng_dict = {char: idx for idx, char in enumerate(eng_unique_characters)}
beng_corpus = "".join(sentences_beng)
beng_unique_characters = sorted(set(beng_corpus))
beng_dict = {char: idx for idx, char in enumerate(beng_unique_characters)}

START = '<S>'
END = '<E>'
PAD = END

eng_dict['<S>'] = 68
eng_dict['<E>'] = 69
beng_dict['<S>'] = 4593
beng_dict['<E>'] = 4594

beng_vocab_size = len(beng_dict)
ind2char = {val:k for k,val in beng_dict.items()}

X = []
Y = []

# -2 as we add additional start and end token
for i, sent in enumerate(sentences_beng):
    if not (len(sent) > max_sequence_length-2 or len(sentences[i]) > max_sequence_length-2) :
        X.append(sentences[i].lower())
        Y.append(sent)

# Dataset Class
class TextDataset(Dataset):

    def __init__(self, english_sentences, bengali_sentences):
        self.english_sentences = english_sentences
        self.bengali_sentences = bengali_sentences

    def __len__(self):
        return len(self.english_sentences)

    def __getitem__(self, idx):
        return self.english_sentences[idx], self.bengali_sentences[idx]
    

# Create DataLoader
dataset = TextDataset(X, Y)
train_data_loader = DataLoader(dataset, batch_size)


# Masking
def gen_masks(eng_batch, beng_batch):
    
    num_sentences = len(eng_batch)
    
    M = torch.zeros(num_sentences, max_sequence_length)
    # for masked input, we set the value to -inf
    mask = torch.tril(torch.ones(max_sequence_length, max_sequence_length))
    # Convert to mask with -inf above the diagonal and 0 below
    look_ahead_mask = mask.masked_fill(mask == 0, float('-inf'))  # -inf for the upper triangle

    encoder_padding_mask = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)
    decoder_padding_mask_self_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)
    decoder_padding_mask_cross_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)

    for idx in range(num_sentences):
        eng_sentence_length, beng_sentence_length = len(eng_batch[idx]), len(beng_batch[idx])
        eng_chars_to_padding_mask = np.arange(eng_sentence_length + 1, max_sequence_length)
        beng_chars_to_padding_mask = np.arange(beng_sentence_length + 1, max_sequence_length)
        encoder_padding_mask[idx, :, eng_chars_to_padding_mask] = True
        encoder_padding_mask[idx, eng_chars_to_padding_mask, :] = True
        decoder_padding_mask_self_attention[idx, :, beng_chars_to_padding_mask] = True
        decoder_padding_mask_self_attention[idx, beng_chars_to_padding_mask, :] = True
        decoder_padding_mask_cross_attention[idx, :, eng_chars_to_padding_mask] = True
        decoder_padding_mask_cross_attention[idx, beng_chars_to_padding_mask, :] = True

    encoder_self_attention_mask = torch.where(encoder_padding_mask, -1e9, 0)
    decoder_self_attention_mask =  torch.where(decoder_padding_mask_self_attention, -1e9, 0) + look_ahead_mask
    decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, -1e9, 0)
    
    return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask




# Model

translator_model = Transformer(n_model, 4*n_model, n_heads, drop_prob, n_layers, max_sequence_length, eng_dict, beng_dict, START, END, PAD )
translator_model = translator_model.to(device)


# Loss and Optimizer
criterion = nn.CrossEntropyLoss(ignore_index=beng_dict[PAD],reduction='none')    
for params in translator_model.parameters():
    if params.dim() > 1:
        nn.init.xavier_uniform_(params)           # initializations

optim = torch.optim.Adam(translator_model.parameters(), lr=1e-4)



loss_hist = []
translator_model.train()     # activating dropout layers
num_epochs = 2

for epoch in range(num_epochs):      
    
    print(f"Epoch {epoch}")
    iterator = iter(train_data_loader)
    substeps = len(train_data_loader)
    
    start = time.time()
    
    for batch_num, batch in enumerate(iterator):
        
        eng_batch, beng_batch = batch
        encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = gen_masks(eng_batch, beng_batch)
        
        optim.zero_grad()
        predictions = translator_model(eng_batch, beng_batch, encoder_self_attention_mask.to(device), 
                                     decoder_self_attention_mask.to(device), decoder_cross_attention_mask.to(device))
        

        # actual
        labels = translator_model.decoder.embedding.batch_tokenize(beng_batch, start_token=False, end_token=True)
        
        loss = criterion(predictions.view(-1, beng_vocab_size).to(device), labels.view(-1).to(device)
        ).to(device)

        # accurately find loss
        valid_indicies = torch.where(labels.view(-1) == beng_dict[PAD], False, True)    # false for pad tokens
        loss = loss.sum() / valid_indicies.sum()
        loss.backward()
        optim.step()

        loss_hist.append(loss.item())
        
        if batch_num % 1000 == 0 or batch_num == substeps-1:
            torch.save(translator_model.state_dict(), f'model{epoch+1}-{batch_num}.pth')
            print(f"Iteration {batch_num} : {loss.item()}")

        
    end = time.time()
    print(f"Time required for 1 epochs: {end-start}")
    torch.save(translator_model.state_dict(), f'model{epoch+1}.pth')

print("Training Completed")
torch.save(translator_model.state_dict(), 'model.pth')