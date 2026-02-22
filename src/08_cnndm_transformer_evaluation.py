
import os
import math
import pandas as pd
import torch
import torch.nn as nn
from rouge_score import rouge_scorer
from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {DEVICE}")

# --- Configuration (Must match training) ---
EMB_DIM = 256
N_HEAD = 4
N_LAYERS = 4
DROPOUT = 0.1
MAX_LEN_TEXT = 512
MAX_LEN_SUMMARY = 128

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
    
    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])

class TransformerModel(nn.Module):
    def __init__(self, num_tokens, dim_model, num_heads, num_encoder_layers, num_decoder_layers, dropout_p, pad_idx):
        super().__init__()
        self.positional_encoder = PositionalEncoding(dim_model=dim_model, dropout_p=dropout_p, max_len=2000)
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.transformer = nn.Transformer(
            d_model=dim_model, nhead=num_heads, num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers, dropout=dropout_p, batch_first=True
        )
        self.out = nn.Linear(dim_model, num_tokens)
        self.pad_idx = pad_idx

    def forward(self, src, trg, src_pad_mask=None, trg_pad_mask=None):
        src_emb = self.embedding(src) * math.sqrt(self.positional_encoder.pos_encoding.shape[2])
        trg_emb = self.embedding(trg) * math.sqrt(self.positional_encoder.pos_encoding.shape[2])
        
        src_emb = self.positional_encoder(src_emb.permute(1,0,2)).permute(1,0,2)
        trg_emb = self.positional_encoder(trg_emb.permute(1,0,2)).permute(1,0,2)

        trg_mask = self.transformer.generate_square_subsequent_mask(trg.size(1)).to(DEVICE)
        output = self.transformer(
            src=src_emb, tgt=trg_emb, src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=trg_pad_mask, memory_key_padding_mask=src_pad_mask, tgt_mask=trg_mask
        )
        return self.out(output)

    def create_pad_mask(self, matrix, pad_token):
        return (matrix == pad_token)

# --- Load Tokenizer ---
def load_or_create_tokenizer():
    """Load existing tokenizer or create a new one from dataset"""
    tokenizer_path = "tokenizer_large"
    
    # Try to load existing tokenizer
    if os.path.exists(os.path.join(tokenizer_path, "vocab.json")):
        logging.info(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = ByteLevelBPETokenizer(
            os.path.join(tokenizer_path, "vocab.json"),
            os.path.join(tokenizer_path, "merges.txt")
        )
    else:
        logging.warning(f"Tokenizer not found at {tokenizer_path}. Creating new tokenizer...")
        logging.info("Downloading CNN/DailyMail dataset...")
        dataset = load_dataset('cnn_dailymail', '3.0.0')
        
        # Save subset for tokenizer training
        if not os.path.exists("data_dump"):
            os.makedirs("data_dump")
        
        with open("data_dump/train_subset.txt", "w", encoding="utf-8") as f:
            for i in range(min(20000, len(dataset['train']))):
                f.write(dataset['train'][i]['article'] + "\n")
                f.write(dataset['train'][i]['highlights'] + "\n")
        
        logging.info("Training new BPE tokenizer...")
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(
            files=["data_dump/train_subset.txt"], 
            vocab_size=30000, 
            min_frequency=2, 
            special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
        )
        
        os.makedirs(tokenizer_path, exist_ok=True)
        tokenizer.save_model(tokenizer_path)
    
    return tokenizer

# --- Beam Search ---
def beam_search_decode(model, src, tokenizer, sos_id, eos_id, pad_id, max_len=128, beam_size=4):
    model.eval()
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
                candidates.append((score + token_prob, seq + [token]))
        
        beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_size]
        if all(seq[-1] == eos_id for _, seq in beams):
            break
    
    return tokenizer.decode(beams[0][1], skip_special_tokens=True)

# --- Main Evaluation ---
def evaluate():
    # Load tokenizer
    tokenizer = load_or_create_tokenizer()
    
    SOS_ID = tokenizer.token_to_id("<s>")
    EOS_ID = tokenizer.token_to_id("</s>")
    PAD_ID = tokenizer.token_to_id("<pad>")
    
    logging.info(f"Vocab size: {tokenizer.get_vocab_size()}")
    
    # Load model
    model_path = "transformer_cnndm_epoch_9.pth"
    
    if not os.path.exists(model_path):
        logging.error(f"Model not found at {model_path}")
        return
    
    logging.info(f"Loading model from {model_path}")
    model = TransformerModel(
        num_tokens=tokenizer.get_vocab_size(),
        dim_model=EMB_DIM,
        num_heads=N_HEAD,
        num_encoder_layers=N_LAYERS,
        num_decoder_layers=N_LAYERS,
        dropout_p=DROPOUT,
        pad_idx=PAD_ID
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    logging.info("Model loaded successfully")
    
    # Load test data
    logging.info("Loading CNN/DailyMail test set...")
    dataset = load_dataset('cnn_dailymail', '3.0.0')
    test_data = dataset['test'].select(range(50))  # Evaluate on 50 samples for speed
    
    # Evaluation
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    results = []
    
    logging.info("Starting evaluation...")
    for idx in tqdm(range(len(test_data)), desc="Evaluating"):
        article = test_data[idx]['article']
        reference = test_data[idx]['highlights']
        
        # Encode
        enc_ids = tokenizer.encode(article).ids[:MAX_LEN_TEXT-2]
        src_tensor = torch.tensor(enc_ids).unsqueeze(0).to(DEVICE)
        
        # Generate summary
        pred_summary = beam_search_decode(
            model, src_tensor, tokenizer, 
            SOS_ID, EOS_ID, PAD_ID, 
            max_len=MAX_LEN_SUMMARY, 
            beam_size=4
        )
        
        # Calculate scores
        scores = scorer.score(reference, pred_summary)
        
        results.append({
            "article": article[:200] + "...",  # Truncate for CSV
            "reference": reference,
            "prediction": pred_summary,
            "rouge1": scores['rouge1'].fmeasure,
            "rouge2": scores['rouge2'].fmeasure,
            "rougeL": scores['rougeL'].fmeasure
        })
        
        # Print first few examples
        if idx < 3:
            logging.info(f"\n--- Example {idx+1} ---")
            logging.info(f"Reference: {reference}")
            logging.info(f"Prediction: {pred_summary}")
            logging.info(f"ROUGE-1: {scores['rouge1'].fmeasure:.4f}")
    
    # Save results
    eval_dir = os.path.join("artifacts", "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    df = pd.DataFrame(results)
    output_path = os.path.join(eval_dir, "cnndm_transformer_results.csv")
    df.to_csv(output_path, index=False)
    
    # Print average scores
    avg_rouge1 = df['rouge1'].mean()
    avg_rouge2 = df['rouge2'].mean()
    avg_rougeL = df['rougeL'].mean()
    
    logging.info(f"\n=== Evaluation Results ===")
    logging.info(f"Average ROUGE-1: {avg_rouge1:.4f}")
    logging.info(f"Average ROUGE-2: {avg_rouge2:.4f}")
    logging.info(f"Average ROUGE-L: {avg_rougeL:.4f}")
    logging.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    evaluate()
