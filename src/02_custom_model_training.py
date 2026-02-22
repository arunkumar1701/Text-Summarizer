
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import logging
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
MAX_VOCAB_SIZE = 20000
MAX_LEN_TEXT = 200
MAX_LEN_SUMMARY = 50
BATCH_SIZE = 4
EMBEDDING_DIM = 64
HIDDEN_DIM = 64
EPOCHS = 2 # Reduced from 15-20 for demo purposes, fully configurable
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Special Tokens
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
        
        # Add special tokens first
        self.add_word(PAD_TOKEN)
        self.add_word(SOS_TOKEN)
        self.add_word(EOS_TOKEN)
        self.add_word(UNK_TOKEN)
        
        # Add most common words
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
        indices = indices[:max_len-2] # space for SOS/EOS if needed, though usually just EOS for decoder
        return indices

class SumDataset(Dataset):
    def __init__(self, df, vocab, max_len_text, max_len_summary):
        self.df = df
        self.vocab = vocab
        self.max_len_text = max_len_text
        self.max_len_summary = max_len_summary

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row['dialogue']
        summary = row['summary']
        
        # Encoder Input: Just Text
        src = self.vocab.text_to_indices(text, self.max_len_text)
        # Pad src
        src = src + [self.vocab.word2index[PAD_TOKEN]] * (self.max_len_text - len(src))
        
        # Decoder Input: SOS + Summary
        trg = [self.vocab.word2index[SOS_TOKEN]] + self.vocab.text_to_indices(summary, self.max_len_summary)
        # Decoder Target: Summary + EOS
        trg_y = self.vocab.text_to_indices(summary, self.max_len_summary) + [self.vocab.word2index[EOS_TOKEN]]
        
        # Pad trg and trg_y
        padding = [self.vocab.word2index[PAD_TOKEN]] * (self.max_len_summary - len(trg) + 1) # +1 because we added SOS
        trg = (trg + padding)[:self.max_len_summary]
        
        padding_y = [self.vocab.word2index[PAD_TOKEN]] * (self.max_len_summary - len(trg_y) + 1)
        trg_y = (trg_y + padding_y)[:self.max_len_summary]
        
        return torch.tensor(src), torch.tensor(trg), torch.tensor(trg_y)

# --- Model Arch ---
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, bidirectional=True)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src len, batch size] (needs transpose if batch_first=True)
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs = [src len, batch size, hid dim * 2]
        # hidden = [n layers * num directions, batch size, hid dim]
        
        # Concat the forward and backward hidden state
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear((hid_dim * 2) + hid_dim, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden = [batch size, hid dim]
        # encoder_outputs = [src len, batch size, hid dim * 2]
        
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
        # input = [batch size] (1 token)
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        
        a = self.attention(hidden, encoder_outputs)
        a = a.unsqueeze(1)
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden.unsqueeze(0), cell)) # Check celldims
        
        # Reuse hidden/cell for next step, fix dims if needed
        # Actually in this specific arch, usually we handle hidden differently. 
        # Making simple for brevity: standard Bahdanau often implies using the hidden state 
        # from the PREVIOUS time step to calculate attention.
        
        assert (output.shape[0] == 1) # one step
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
        # src = [batch size, src len]
        # trg = [batch size, trg len]
        
        # Permute for current RNN implementation expecting [len, batch]
        src = src.permute(1, 0)
        trg = trg.permute(1, 0)
        
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        encoder_outputs, hidden, cell = self.encoder(src)

        # Prepare hidden/cell for decoder (which is unidirectional)
        # Encoder hidden: [n_layers * n_dirs, batch, hid_dim] -> [2, batch, hid] (since 1 layer, 2 dirs)
        # Decoder expects: [n_layers, batch, hid] -> [1, batch, hid]
        
        # We need to reshape/combine encoder hidden states for decoder.
        # Since we already combined outputs in the encoder forward pass using a linear layer,
        # we should do something similar here or just use the last layer.
        
        # Current Encoder.forward returns: 
        # hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))) 
        # This returns [batch, hid_dim]. 
        # Decoder RNN expects [layers=1, batch, hid_dim].
        
        # Decoder.forward expects 2D hidden [batch, hid] (it unsqueezes it internally).
        # So we keep it as is.
        # hidden = hidden.unsqueeze(0) # REMOVED
        
        # Prepare cell state: Encoder returns [2, batch, hid] (bidirectional)
        # Decoder expects [1, batch, hid]
        # Sum bidirectional states and unsqueeze
        cell_fwd = cell[0]
        cell_bwd = cell[1]
        cell = (cell_fwd + cell_bwd).unsqueeze(0) # [1, batch, hid]
        
        input = trg[0,:] 
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[t] = output
            teacher_force = np.random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
            
        return outputs

def train_model():
    data_path = os.path.join("artifacts", "data")
    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_path, "val.csv"))
    
    # Fill NAs
    train_df = train_df.fillna("")
    val_df = val_df.fillna("")

    logging.info("Building Vocabulary...")
    vocab = Vocabulary()
    vocab.build_vocabulary(train_df['dialogue'].tolist() + train_df['summary'].tolist())
    
    # Save Vocab
    vocab_path = os.path.join("artifacts", "models", "custom")
    os.makedirs(vocab_path, exist_ok=True)
    with open(os.path.join(vocab_path, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)
        
    logging.info(f"Vocabulary Size: {vocab.vocab_size}")

    train_dataset = SumDataset(train_df, vocab, MAX_LEN_TEXT, MAX_LEN_SUMMARY)
    val_dataset = SumDataset(val_df, vocab, MAX_LEN_TEXT, MAX_LEN_SUMMARY)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    
    # Init Model
    enc = Encoder(vocab.vocab_size, EMBEDDING_DIM, HIDDEN_DIM, 0.5)
    attn = Attention(HIDDEN_DIM)
    dec = Decoder(vocab.vocab_size, EMBEDDING_DIM, HIDDEN_DIM, 0.5, attn)
    model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2index[PAD_TOKEN])
    
    logging.info("Starting Training...")
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for i, (src, trg, trg_y) in enumerate(train_loader):
            if i % 10 == 0:
                 logging.info(f"Epoch {epoch+1}, Batch {i}...")
            src, trg, trg_y = src.to(DEVICE), trg.to(DEVICE), trg_y.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(src, trg)
            
            # output = [trg len, batch size, output dim]
            # trg_y = [batch size, trg len] -> [trg len, batch size]
            
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg_y = trg_y.permute(1,0)[1:].reshape(-1)
            
            loss = criterion(output, trg_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        logging.info(f"Epoch {epoch+1}/{EPOCHS} Loss: {epoch_loss/len(train_loader)}")
        
        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(vocab_path, "custom_model.pth"))

if __name__ == "__main__":
    train_model()
