
import os
import math
import pandas as pd
import torch
import torch.nn as nn
from rouge_score import rouge_scorer
import pickle
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Re-define Transformer Model (Must match training) ---
# ideally this traverses to a utils file, but for standalone script we paste here
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

# Vocabulary Definitions (Must match training)
class Vocabulary:
    def __init__(self):
        self.word2index = {}
        self.index2word = {}
        self.vocab_size = 0
    # methods not needed for inference if we load pickle objects

# --- Beam Search Inference ---

def beam_search_decode(model, src, vocab, max_len=50, beam_size=3):
    model.eval()
    
    # Src: [1, seq_len]
    src_pad_mask = model.create_pad_mask(src, vocab.word2index['<PAD>'])
    
    # Calculate encoder output once (Optimized) - Standard nn.Transformer doesn't expirence encoder separately easily
    # So we re-run full forward. For true optimization we'd cache memory.
    # Basic Beam Search Implementation:
    
    # Beam = list of (score, sequence)
    # Start with SOS
    beams = [(0, [vocab.word2index['<SOS>']])] # log prob, sequence
    
    for _ in range(max_len):
        candidates = []
        
        for score, seq in beams:
            if seq[-1] == vocab.word2index['<EOS>']:
                candidates.append((score, seq))
                continue
            
            trg_tensor = torch.tensor([seq]).to(DEVICE) # [1, curr_len]
            trg_pad_mask = model.create_pad_mask(trg_tensor, vocab.word2index['<PAD>'])
            
            with torch.no_grad():
                output = model(src, trg_tensor, src_pad_mask=src_pad_mask, trg_pad_mask=trg_pad_mask)
            
            # Get last token probabilities
            logits = output[:, -1, :] # [1, vocab_size]
            probs = torch.log_softmax(logits, dim=1) # [1, vocab_size]
            
            # Top k
            topk_probs, topk_indices = torch.topk(probs, beam_size)
            
            for i in range(beam_size):
                token = topk_indices[0][i].item()
                token_prob = topk_probs[0][i].item()
                new_score = score + token_prob
                new_seq = seq + [token]
                candidates.append((new_score, new_seq))
        
        # Select best k candidates
        beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_size]
        
        # If all beams ended, break
        if all(seq[-1] == vocab.word2index['<EOS>'] for _, seq in beams):
            break
            
    # Return best beam
    best_seq = beams[0][1]
    
    # Decode
    tokens = [vocab.index2word[i] for i in best_seq]
    
    # Remove SOS and EOS
    if '<SOS>' in tokens: tokens.remove('<SOS>')
    if '<EOS>' in tokens: tokens.remove('<EOS>')
    
    return " ".join(tokens)

def load_transformer_model():
    model_dir = os.path.join("artifacts", "models", "transformer_scratch")
    vocab_path = os.path.join(model_dir, "vocab.pkl")
    model_path = os.path.join(model_dir, "transformer_model.pth")
    
    if not os.path.exists(model_path):
        logging.error("Transformer model not found!")
        return None, None
        
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
        
    # Hyperparams must match training
    model = TransformerModel(
        num_tokens=vocab.vocab_size, 
        dim_model=256, 
        num_heads=4, 
        num_encoder_layers=3, 
        num_decoder_layers=3, 
        dropout_p=0.1,
        pad_idx=vocab.word2index['<PAD>']
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    return model, vocab

def evaluate_transformer():
    data_path = os.path.join("artifacts", "data")
    test_df = pd.read_csv(os.path.join(data_path, "test.csv")).head(20)
    
    model, vocab = load_transformer_model()
    if not model:
        return

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    results = []
    
    logging.info("Starting Transformer Evaluation (Beam Search)...")
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        dialogue = row['dialogue']
        reference = row['summary']
        
        tokens = dialogue.split()
        indices = [vocab.word2index.get(t, vocab.word2index['<UNK>']) for t in tokens]
        indices = indices[:198] # Limit length
        src_tensor = torch.tensor(indices).unsqueeze(0).to(DEVICE) # [1, src_len]
        
        pred_summary = beam_search_decode(model, src_tensor, vocab, beam_size=3)
        
        scores = scorer.score(reference, pred_summary)
        
        results.append({
            "dialogue": dialogue,
            "reference": reference,
            "transformer_summary": pred_summary,
            "rouge1": scores['rouge1'].fmeasure,
            "rougeL": scores['rougeL'].fmeasure
        })
        
    eval_dir = os.path.join("artifacts", "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(os.path.join(eval_dir, "transformer_results.csv"), index=False)
    logging.info(f"Saved results to {eval_dir}")

if __name__ == "__main__":
    evaluate_transformer()
