
import os
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import logging
import pickle

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
MAX_VOCAB_SIZE = 20000
MAX_LEN_TEXT = 200
MAX_LEN_SUMMARY = 50
BATCH_SIZE = 8          # Increased batch size slightly for Transformer
EMB_DIM = 256
N_HEAD = 4
HID_DIM = 512
N_LAYERS = 3
DROPOUT = 0.1
EPOCHS = 30
PATIENCE = 5
LEARNING_RATE = 0.0001
GRAD_CLIP = 1.0

# Special Tokens
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'

# --- Data Processing (Same as before) ---
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
        
        src = self.vocab.text_to_indices(text, self.max_len_text)
        src = src + [self.vocab.word2index[PAD_TOKEN]] * (self.max_len_text - len(src))
        
        trg = [self.vocab.word2index[SOS_TOKEN]] + self.vocab.text_to_indices(summary, self.max_len_summary)
        trg_y = self.vocab.text_to_indices(summary, self.max_len_summary) + [self.vocab.word2index[EOS_TOKEN]]
        
        padding = [self.vocab.word2index[PAD_TOKEN]] * (self.max_len_summary - len(trg) + 1)
        trg = (trg + padding)[:self.max_len_summary]
        
        padding_y = [self.vocab.word2index[PAD_TOKEN]] * (self.max_len_summary - len(trg_y) + 1)
        trg_y = (trg_y + padding_y)[:self.max_len_summary]
        
        return torch.tensor(src), torch.tensor(trg), torch.tensor(trg_y)

# --- Transformer Model ---

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)
        
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])

class TransformerModel(nn.Module):
    def __init__(self, num_tokens, dim_model, num_heads, num_encoder_layers, num_decoder_layers, dropout_p, pad_idx):
        super().__init__()
        
        self.model_type = "Transformer"
        self.dim_model = dim_model
        
        self.positional_encoder = PositionalEncoding(dim_model=dim_model, dropout_p=dropout_p, max_len=1000)
        self.embedding = nn.Embedding(num_tokens, dim_model)
        
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
            batch_first=True
        )
        
        self.out = nn.Linear(dim_model, num_tokens)
        self.pad_idx = pad_idx

    def forward(self, src, trg, src_pad_mask=None, trg_pad_mask=None):
        # src: [batch, src_len]
        # trg: [batch, trg_len]
        
        src_emb = self.positional_encoder(self.embedding(src) * math.sqrt(self.dim_model))
        trg_emb = self.positional_encoder(self.embedding(trg) * math.sqrt(self.dim_model))
        
        # Src and Trg need to be permuted for PosEncoding if batch_first=False, but we used batch_first=True in Transformer
        # Note: PosEncoding implementation above expects [seq_len, batch, dim] if we didn't transpose it.
        # But we did: pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1) -> [seq_len, 1, dim]
        # The nn.Transformer with batch_first=True expects [batch, seq_len, dim].
        
        # Let's adjust PositionalEncoding to be batch_first friendly or transpose inputs.
        # Standard fix: Transpose inputs to [seq_len, batch] for PE, then transpose back? 
        # Or just make PE returns [1, seq_len, dim] and broadcast.
        
        # Fixing PE usage:
        # Pytorch's nn.Transformer with batch_first=True takes [batch, seq, feature].
        # Our PE currently generates [seq, 1, feature].
        # We need to permute it to [1, seq, feature] to add to [batch, seq, feature].
        
        # Let's fix embedding + PE inside this forward block to be safe:
        
        # Scale embedding
        src_emb = self.embedding(src) * math.sqrt(self.dim_model) # [batch, src_len, dim]
        trg_emb = self.embedding(trg) * math.sqrt(self.dim_model)
        
        # Add PE (naive implementation inline for clarity/correctness with batch_first)
        seq_len_src = src.size(1)
        seq_len_trg = trg.size(1)
        
        # We'll just rely on the model learning without explicit PE class complexity or resize on fly
        # Ideally we reuse the class. Let's assume the class above is fixed to:
        # return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(1), :].transpose(0,1))
        # But let's just use the Layer as is and Permute around it strictly.
        
        src_emb = src_emb.permute(1, 0, 2) # [src_len, batch, dim]
        trg_emb = trg_emb.permute(1, 0, 2)
        
        src_emb = self.positional_encoder(src_emb)
        trg_emb = self.positional_encoder(trg_emb)
        
        src_emb = src_emb.permute(1, 0, 2) # Back to [batch, src_len, dim]
        trg_emb = trg_emb.permute(1, 0, 2)

        # Masks
        # Target Mask (Causal)
        trg_mask = self.transformer.generate_square_subsequent_mask(seq_len_trg).to(DEVICE)
        
        output = self.transformer(
            src=src_emb, 
            tgt=trg_emb, 
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=trg_pad_mask, 
            memory_key_padding_mask=src_pad_mask,
            tgt_mask=trg_mask
        )
        
        output = self.out(output)
        return output

    def get_tgt_mask(self, size) -> torch.tensor:
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        return mask

    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [batch, seq_len]
        # result = [batch, seq_len]
        return (matrix == pad_token)

# --- Training ---

def train_model():
    data_path = os.path.join("artifacts", "data")
    train_df = pd.read_csv(os.path.join(data_path, "train.csv")).fillna("")
    val_df = pd.read_csv(os.path.join(data_path, "val.csv")).fillna("")

    logging.info("Building Vocabulary...")
    vocab = Vocabulary()
    vocab.build_vocabulary(train_df['dialogue'].tolist() + train_df['summary'].tolist())
    
    # Save Vocab
    model_dir = os.path.join("artifacts", "models", "transformer_scratch")
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)
    
    train_dataset = SumDataset(train_df, vocab, MAX_LEN_TEXT, MAX_LEN_SUMMARY)
    val_dataset = SumDataset(val_df, vocab, MAX_LEN_TEXT, MAX_LEN_SUMMARY)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = TransformerModel(
        num_tokens=vocab.vocab_size, 
        dim_model=EMB_DIM, 
        num_heads=N_HEAD, 
        num_encoder_layers=N_LAYERS, 
        num_decoder_layers=N_LAYERS, 
        dropout_p=DROPOUT,
        pad_idx=vocab.word2index[PAD_TOKEN]
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2index[PAD_TOKEN])
    
    logging.info("Starting Training with Transformer...")
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for i, (src, trg, trg_y) in enumerate(train_loader):
            src, trg, trg_y = src.to(DEVICE), trg.to(DEVICE), trg_y.to(DEVICE)
            
            src_pad_mask = model.create_pad_mask(src, vocab.word2index[PAD_TOKEN])
            trg_pad_mask = model.create_pad_mask(trg, vocab.word2index[PAD_TOKEN])
            
            optimizer.zero_grad()
            output = model(src, trg, src_pad_mask=src_pad_mask, trg_pad_mask=trg_pad_mask)
            
            # output: [batch, seq_len, vocab_size]
            # trg_y: [batch, seq_len]
            
            output_dim = output.shape[-1]
            output = output.reshape(-1, output_dim)
            trg_y = trg_y.reshape(-1)
            
            loss = criterion(output, trg_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if i % 50 == 0:
                print(f"Epoch {epoch+1} Batch {i} Loss: {loss.item():.4f}", end='\r')
        
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, trg, trg_y in val_loader:
                src, trg, trg_y = src.to(DEVICE), trg.to(DEVICE), trg_y.to(DEVICE)
                src_pad_mask = model.create_pad_mask(src, vocab.word2index[PAD_TOKEN])
                trg_pad_mask = model.create_pad_mask(trg, vocab.word2index[PAD_TOKEN])
                
                output = model(src, trg, src_pad_mask=src_pad_mask, trg_pad_mask=trg_pad_mask)
                output = output.reshape(-1, output.shape[-1])
                trg_y = trg_y.reshape(-1)
                loss = criterion(output, trg_y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        logging.info(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Early Stopping & Saving
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(model_dir, "transformer_model.pth"))
            logging.info("Model saved!")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logging.info("Early stopping triggered.")
                break

if __name__ == "__main__":
    train_model()
