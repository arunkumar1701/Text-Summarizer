#!/usr/bin/env python
# coding: utf-8

# # End-to-End Text Summarization on Google Colab
# 
# This notebook implements the training and evaluation of the Text Summarization project using the SAMSUM dataset. 
# **Make sure to enable GPU Runtime: Runtime > Change runtime type > T4 GPU**

# In[5]:


# get_ipython().system('pip install transformers datasets rouge_score deep-translator accelerate torch')


# In[6]:


import os
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from datasets import load_dataset
import logging
import pickle
import random
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# In[ ]:


# --- 1. Data Ingestion & Processing ---

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+', '', text)
    # Remove emojis and special characters (keep alphanumeric and basic punctuation)
    text = re.sub(r'[^\w\s.,!?\']', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

logger.info("Downloading Dataset...")
dataset = load_dataset('knkarthick/samsum')

train_texts = [clean_text(t) for t in dataset['train']['dialogue']]
train_summaries = [clean_text(t) for t in dataset['train']['summary']]
val_texts = [clean_text(t) for t in dataset['validation']['dialogue']]
val_summaries = [clean_text(t) for t in dataset['validation']['summary']]
test_texts = [clean_text(t) for t in dataset['test']['dialogue']]
test_summaries = [clean_text(t) for t in dataset['test']['summary']]


# In[ ]:


# --- 2. Vocabulary & Dataset ---

MAX_VOCAB_SIZE = 20000
MAX_LEN_TEXT = 200
MAX_LEN_SUMMARY = 50
BATCH_SIZE = 64  # Increased for Colab GPU
EMBEDDING_DIM = 256
HIDDEN_DIM = 512

SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'

class Vocabulary:
    def __init__(self):
        self.word2index = {}
        self.index2word = {}
        self.vocab_size = 0

    def build_vocabulary(self, sentence_list):
        counter = Counter()
        for sentence in sentence_list:
            if isinstance(sentence, str):
                counter.update(sentence.split())
        
        self.add_word(PAD_TOKEN)
        self.add_word(SOS_TOKEN)
        self.add_word(EOS_TOKEN)
        self.add_word(UNK_TOKEN)
        
        for word, _ in counter.most_common(MAX_VOCAB_SIZE):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.vocab_size
            self.index2word[self.vocab_size] = word
            self.vocab_size += 1

    def text_to_indices(self, text, max_len):
        tokens = text.split() if isinstance(text, str) else []
        indices = [self.word2index.get(token, self.word2index[UNK_TOKEN]) for token in tokens]
        indices = indices[:max_len-2]
        return indices

logger.info("Building Vocabulary...")
vocab = Vocabulary()
vocab.build_vocabulary(train_texts + train_summaries)
logger.info(f"Vocab Size: {vocab.vocab_size}")

class SumDataset(Dataset):
    def __init__(self, texts, summaries, vocab):
        self.texts = texts
        self.summaries = summaries
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        summary = self.summaries[idx]
        
        src = self.vocab.text_to_indices(text, MAX_LEN_TEXT)
        src = src + [self.vocab.word2index[PAD_TOKEN]] * (MAX_LEN_TEXT - len(src))
        
        trg = [self.vocab.word2index[SOS_TOKEN]] + self.vocab.text_to_indices(summary, MAX_LEN_SUMMARY)
        trg_y = self.vocab.text_to_indices(summary, MAX_LEN_SUMMARY) + [self.vocab.word2index[EOS_TOKEN]]
        
        padding = [self.vocab.word2index[PAD_TOKEN]] * (MAX_LEN_SUMMARY - len(trg) + 1)
        trg = (trg + padding)[:MAX_LEN_SUMMARY]
        padding_y = [self.vocab.word2index[PAD_TOKEN]] * (MAX_LEN_SUMMARY - len(trg_y) + 1)
        trg_y = (trg_y + padding_y)[:MAX_LEN_SUMMARY]
        
        return torch.tensor(src), torch.tensor(trg), torch.tensor(trg_y)

train_dataset = SumDataset(train_texts, train_summaries, vocab)
val_dataset = SumDataset(val_texts, val_summaries, vocab)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)


# In[ ]:


# --- 3. Model Architecture ---

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, bidirectional=True)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear((hid_dim * 2) + hid_dim, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM((hid_dim * 2) + emb_dim, hid_dim)
        self.fc_out = nn.Linear((hid_dim * 2) + hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden, encoder_outputs)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden.unsqueeze(0), cell))
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        return prediction, hidden.squeeze(0), cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        src = src.permute(1, 0)
        trg = trg.permute(1, 0)
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(src)
        
        # Correctly handle bi-directional cell state summation for uni-directional decoder
        cell_fwd = cell[0]
        cell_bwd = cell[1]
        cell = (cell_fwd + cell_bwd).unsqueeze(0)
        
        input = trg[0,:] 
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs


# In[ ]:


# --- 4. Training with Early Stopping --- 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using Device: {DEVICE}")

enc = Encoder(vocab.vocab_size, EMBEDDING_DIM, HIDDEN_DIM, 0.5)
attn = Attention(HIDDEN_DIM)
dec = Decoder(vocab.vocab_size, EMBEDDING_DIM, HIDDEN_DIM, 0.5, attn)
model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2index[PAD_TOKEN])

EPOCHS = 7 # Changed to 7 as requested
PATIENCE = 3 # Early Stopping Patience
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
    
    for i, (src, trg, trg_y) in enumerate(progress_bar):
        src, trg, trg_y = src.to(DEVICE), trg.to(DEVICE), trg_y.to(DEVICE)
        
        optimizer.zero_grad()
        output = model(src, trg)
        
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg_y = trg_y.permute(1,0)[1:].reshape(-1)
        
        loss = criterion(output, trg_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        progress_bar.set_postfix(loss=loss.item())
            
    avg_train_loss = epoch_loss/len(train_loader)
    print(f"\nEpoch {epoch+1} Train Loss: {avg_train_loss:.4f}")
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i, (src, trg, trg_y) in enumerate(val_loader):
            src, trg, trg_y = src.to(DEVICE), trg.to(DEVICE), trg_y.to(DEVICE)
            output = model(src, trg, 0)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg_y = trg_y.permute(1,0)[1:].reshape(-1)
            loss = criterion(output, trg_y)
            val_loss += loss.item()
            
    avg_val_loss = val_loss/len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    
    # Early Stopping & Checkpointing
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "custom_model.pth")
        print("Validation Loss Improved - Model Saved!")
    else:
        patience_counter += 1
        print(f"Validation Loss did not improve. Patience: {patience_counter}/{PATIENCE}")
        
    if patience_counter >= PATIENCE:
        print("Early Stopping Triggered.")
        break

with open("vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)
print("Training Completed.")

