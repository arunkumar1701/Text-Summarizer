
import os
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Device: {DEVICE}")

# Set seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# --- 1. Configuration ---
MAX_LEN = 512  # Total length (Article + Summary)
BATCH_SIZE = 8
EMB_DIM = 512
N_HEAD = 8
N_LAYERS = 6
DROPOUT = 0.1
EPOCHS = 10
LEARNING_RATE = 0.0001
GRAD_CLIP = 1.0

# --- 2. Data & Tokenizer ---
print("Loading CNN/DailyMail Dataset...")
dataset = load_dataset('cnn_dailymail', '3.0.0')
train_data = dataset['train']
val_token_count = 1000 # Validation subset size

# Create/Load Tokenizer
if not os.path.exists("tokenizer_gpt"):
    os.makedirs("tokenizer_gpt")
    print("Training BPE Tokenizer...")
    # Use a subset for training tokenizer
    with open("temp_train.txt", "w", encoding="utf-8") as f:
        for i in range(min(20000, len(train_data))):
            f.write(train_data[i]['article'] + "\n" + train_data[i]['highlights'] + "\n")
            
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=["temp_train.txt"], vocab_size=30000, min_frequency=2, special_tokens=[
        "<s>", "<pad>", "</s>", "<unk>", "<sep>"
    ])
    tokenizer.save_model("tokenizer_gpt")
else:
    tokenizer = ByteLevelBPETokenizer(
        "tokenizer_gpt/vocab.json",
        "tokenizer_gpt/merges.txt"
    )

SOS_ID = tokenizer.token_to_id("<s>")
EOS_ID = tokenizer.token_to_id("</s>")
PAD_ID = tokenizer.token_to_id("<pad>")
SEP_ID = tokenizer.token_to_id("<sep>")
UNK_ID = tokenizer.token_to_id("<unk>")

print(f"Vocab Size: {tokenizer.get_vocab_size()}")

# --- 3. Dataset Class (The "GPT" Logic) ---
class GPT_SummarizationDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        article = item['article']
        summary = item['highlights']

        # Format: <s> Article <sep> Summary </s>
        # We need to manage lengths so it fits in max_len
        
        # Tokenize (keep some space for special tokens)
        article_ids = self.tokenizer.encode(article).ids
        summary_ids = self.tokenizer.encode(summary).ids
        
        # Budget: Leave space for <s>, <sep>, </s> (3 tokens)
        # If too long, truncate article first
        total_len = len(article_ids) + len(summary_ids) + 3
        if total_len > self.max_len:
            available_for_article = self.max_len - len(summary_ids) - 3
            if available_for_article < 10: # If summary is huge
                # Semantic truncation (naive)
                summary_ids = summary_ids[:self.max_len // 2]
                article_ids = article_ids[:self.max_len // 2 - 3]
            else:
                article_ids = article_ids[:available_for_article]

        # Construct Sequence
        input_ids = [SOS_ID] + article_ids + [SEP_ID] + summary_ids + [EOS_ID]
        
        # Create Mask (0 for pad, 1 for data)
        # In Reference: Mask input (article) as 0 loss, output (summary) as 1 loss?
        # Standard GPT training predicts EVERY token. 
        # But we can mask the loss for the article part if we want "Conditional Generation".
        # Let's start with standard LM training (predict everything) for simplicity, 
        # or implement the reference's "mask loss on article" logic.
        
        # Reference logic: "We also need to create a mask -- with 0s at inputs and 1s at targets"
        # 1 means "calculate loss here", 0 means "ignore"
        loss_mask = [0] * (len(article_ids) + 2) + [1] * (len(summary_ids) + 1) 
        # +2 is <s> and <sep>, +1 is </s>
        
        # Padding
        pad_len = self.max_len - len(input_ids)
        input_ids = input_ids + [PAD_ID] * pad_len
        loss_mask = loss_mask + [0] * pad_len # Don't calculate loss on pads

        return torch.tensor(input_ids), torch.tensor(loss_mask)

train_dataset = GPT_SummarizationDataset(train_data, tokenizer, MAX_LEN)
# Validation subset
val_dataset = GPT_SummarizationDataset(dataset['validation'].select(range(val_token_count)), tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 4. The Decoder-Only Model ---
class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(MAX_LEN, d_model) # Learned Positional Embedding
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers) # "Encoder" in PyTorch = Decoder block (self-attn)
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, mask=None):
        # src: [Batch, Seq]
        b, seq_len = src.shape
        
        # Positions: [0, 1, 2, ... seq_len-1]
        positions = torch.arange(0, seq_len, device=src.device).unsqueeze(0).expand(b, seq_len)
        
        # Embeddings
        x = self.embedding(src) + self.pos_encoder(positions)
        x = self.dropout(x)
        
        # Causal Mask (Prevent looking at future tokens)
        # -inf above diagonal
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(src.device)
        
        # Pass through Transformer
        # We use 'TransformerEncoder' because in PyTorch 'TransformerDecoder' expects cross-attention.
        # A stack of Self-Attention blocks is technically an "Encoder" class in PyTorch, 
        # but with a causal mask, it behaves as a Decoder (GPT).
        output = self.transformer_decoder(x, mask=causal_mask, src_key_padding_mask=mask)
        
        return self.fc_out(output)

model = GPTModel(tokenizer.get_vocab_size(), EMB_DIM, N_HEAD, N_LAYERS, DROPOUT).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(reduction='none') # We apply mask manually

# --- 5. Training Loop ---
print("Starting Training (Decoder-Only)...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for inputs, loss_mask in progress_bar:
        inputs = inputs.to(DEVICE)       # [Batch, Seq]
        loss_mask = loss_mask.to(DEVICE) # [Batch, Seq]
        
        # Inputs to model: Everything except last token
        # Targets: Everything shifted by 1 (predict next token)
        src = inputs[:, :-1]
        tgt = inputs[:, 1:]
        mask_curr = loss_mask[:, 1:] # align mask with targets
        
        # Padding mask for attention (True where we should NOT attend)
        pad_mask = (src == PAD_ID)
        
        optimizer.zero_grad()
        output = model(src, mask=pad_mask) # [Batch, Seq-1, Vocab]
        
        # Reshape for loss
        output = output.reshape(-1, tokenizer.get_vocab_size())
        tgt = tgt.reshape(-1)
        mask_curr = mask_curr.reshape(-1)
        
        # Calculate Loss
        loss_raw = criterion(output, tgt)
        
        # Apply mask (only learn on summary tokens)
        loss = (loss_raw * mask_curr).sum() / (mask_curr.sum() + 1e-9)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} Average Loss: {avg_loss}")
    
    # Save Model
    torch.save(model.state_dict(), f"gpt_summarizer_epoch_{epoch+1}.pth")
    
    # --- Generation Check (Greedy) ---
    model.eval()
    with torch.no_grad():
        # inference on first validation item
        val_item = val_dataset[0]
        # find where <sep> is
        inp_full = val_item[0].tolist()
        try:
            sep_idx = inp_full.index(SEP_ID)
            prompt = inp_full[:sep_idx+1] # <s> ... Article ... <sep>
        except:
             prompt = inp_full[:10]

        curr_seq = torch.tensor([prompt]).to(DEVICE)
        
        print("\nOriginal Summary:", tokenizer.decode(inp_full[sep_idx+1:], skip_special_tokens=True))
        print("Reference (Decoded Prompt):", tokenizer.decode(prompt, skip_special_tokens=True)[:100] + "...")
        print("Calculating generation...")
        
        # Generate 50 tokens
        for _ in range(50):
            out = model(curr_seq)
            next_token_logits = out[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
            if next_token.item() == EOS_ID:
                break
                
            curr_seq = torch.cat([curr_seq, next_token], dim=1)
        
        gen_text = tokenizer.decode(curr_seq[0].tolist()[sep_idx+1:], skip_special_tokens=True)
        print(f"Generated: {gen_text}\n")
