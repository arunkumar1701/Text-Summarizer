
import os
import math
import pandas as pd
import torch
import torch.nn as nn
from rouge_score import rouge_scorer
from tokenizers import ByteLevelBPETokenizer
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Configuration (Must match training) ---
EMB_DIM = 256
N_HEAD = 4
HID_DIM = 512
N_LAYERS = 4
DROPOUT = 0.1
MAX_LEN_TEXT = 256
MAX_LEN_SUMMARY = 64

# --- Model Architecture ---
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
        src_emb = self.embedding(src) * math.sqrt(self.dim_model)
        trg_emb = self.embedding(trg) * math.sqrt(self.dim_model)
        
        src_emb = src_emb.permute(1, 0, 2)
        trg_emb = trg_emb.permute(1, 0, 2)
        
        src_emb = self.positional_encoder(src_emb)
        trg_emb = self.positional_encoder(trg_emb)
        
        src_emb = src_emb.permute(1, 0, 2)
        trg_emb = trg_emb.permute(1, 0, 2)

        trg_mask = self.transformer.generate_square_subsequent_mask(trg.size(1)).to(DEVICE)
        
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
        
    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        return (matrix == pad_token)

# --- Helper to load BPE Tokenizer ---
def load_bpe_tokenizer(path):
    # Expects vocab.json and merges.txt in the path
    return ByteLevelBPETokenizer(
        os.path.join(path, "vocab.json"),
        os.path.join(path, "merges.txt")
    )

def load_model():
    model_dir = os.path.join("artifacts", "models", "transformer_bpe")
    tokenizer_path = os.path.join(model_dir, "tokenizer_bpe")
    model_path = os.path.join(model_dir, "transformer_bpe_model.pth")
    
    if not os.path.exists(model_path):
        logging.error(f"Model not found at {model_path}")
        return None, None
    if not os.path.exists(tokenizer_path):
        logging.error(f"Tokenizer directory not found at {tokenizer_path}")
        return None, None
        
    logging.info("Loading Tokenizer...")
    tokenizer = load_bpe_tokenizer(tokenizer_path)
    
    # Get IDs for special tokens
    pad_id = tokenizer.token_to_id("<pad>")
    
    logging.info("Loading Model...")
    model = TransformerModel(
        num_tokens=tokenizer.get_vocab_size(), 
        dim_model=EMB_DIM, 
        num_heads=N_HEAD, 
        num_encoder_layers=N_LAYERS, 
        num_decoder_layers=N_LAYERS, 
        dropout_p=DROPOUT,
        pad_idx=pad_id
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    return model, tokenizer

# --- Beam Search Inference ---
def beam_search_decode(model, src, tokenizer, max_len=50, beam_size=3):
    model.eval()
    
    sos_id = tokenizer.token_to_id("<s>")
    eos_id = tokenizer.token_to_id("</s>")
    pad_id = tokenizer.token_to_id("<pad>")
    
    src_pad_mask = model.create_pad_mask(src, pad_id)
    
    beams = [(0, [sos_id])] 
    
    for _ in range(max_len):
        candidates = []
        for score, seq in beams:
            if seq[-1] == eos_id:
                candidates.append((score, seq))
                continue
            
            trg_tensor = torch.tensor([seq]).to(DEVICE)
            trg_pad_mask = model.create_pad_mask(trg_tensor, pad_id)
            
            with torch.no_grad():
                output = model(src, trg_tensor, src_pad_mask=src_pad_mask, trg_pad_mask=trg_pad_mask)
            
            logits = output[:, -1, :] 
            probs = torch.log_softmax(logits, dim=1)
            
            topk_probs, topk_indices = torch.topk(probs, beam_size)
            
            for i in range(beam_size):
                token = topk_indices[0][i].item()
                token_prob = topk_probs[0][i].item()
                new_score = score + token_prob
                new_seq = seq + [token]
                candidates.append((new_score, new_seq))
        
        beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_size]
        if all(seq[-1] == eos_id for _, seq in beams):
            break
            
    best_seq = beams[0][1]
    decoded = tokenizer.decode(best_seq, skip_special_tokens=True)
    return decoded

def evaluate():
    data_path = os.path.join("artifacts", "data")
    test_df = pd.read_csv(os.path.join(data_path, "test.csv")).head(20)
    
    model, tokenizer = load_model()
    if not model:
        return

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    results = []
    
    logging.info("Starting BPE Transformer Evaluation...")
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        dialogue = str(row['dialogue'])
        reference = str(row['summary'])
        
        enc_ids = tokenizer.encode(dialogue).ids
        enc_ids = enc_ids[:MAX_LEN_TEXT-2]
        
        src_tensor = torch.tensor(enc_ids).unsqueeze(0).to(DEVICE)
        
        pred_summary = beam_search_decode(model, src_tensor, tokenizer, max_len=MAX_LEN_SUMMARY, beam_size=3)
        
        scores = scorer.score(reference, pred_summary)
        
        results.append({
            "dialogue": dialogue,
            "reference": reference,
            "bpe_transformer_summary": pred_summary,
            "rouge1": scores['rouge1'].fmeasure,
            "rougeL": scores['rougeL'].fmeasure
        })
        
    eval_dir = os.path.join("artifacts", "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(os.path.join(eval_dir, "bpe_transformer_results.csv"), index=False)
    logging.info(f"Saved results to {eval_dir}")

if __name__ == "__main__":
    evaluate()
