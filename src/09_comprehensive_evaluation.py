
import os
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, load_metric
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import numpy as np
import sys
import pickle
import math
import torch.nn as nn
import importlib.util

# --- Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAMSUM_TEST_SIZE = 50 
ARTIFACTS_DIR = os.path.join(os.getcwd(), 'artifacts')
MODELS_DIR = os.path.join(ARTIFACTS_DIR, 'models')
RESULTS_FILE = os.path.join(ARTIFACTS_DIR, 'evaluation', 'full_comparison.csv')

# --- Dynamic Import Helper ---
def import_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def calculate_rouge(preds, refs):
    try:
        if not preds or not refs: return {}
        scorer = load_metric('rouge')
        scores = scorer.compute(predictions=preds, references=refs)
        return {
            'rouge1': round(scores['rouge1'].mid.fmeasure, 4),
            'rouge2': round(scores['rouge2'].mid.fmeasure, 4),
            'rougeL': round(scores['rougeL'].mid.fmeasure, 4)
        }
    except Exception as e:
        print(f"Error calculating ROUGE: {e}")
        return {}

print(f"Running on: {DEVICE}")

# --- 1. Load Data ---
print("\nStep 1: Loading SAMSum Dataset (Test)...")
try:
    dataset = load_dataset("samsum", split="test")
    test_data = dataset.select(range(SAMSUM_TEST_SIZE))
    articles = [d['dialogue'] for d in test_data]
    references = [d['summary'] for d in test_data]
    results = {'SAMSUM_Reference': references}
except Exception as e:
    print(f"Failed to load dataset: {e}")
    sys.exit(1)

# --- 2. PEGASUS ---
print("\nStep 2: Evaluating Pegasus...")
try:
    pegasus_model_name = "google/pegasus-samsum"
    pegasus_tokenizer = PegasusTokenizer.from_pretrained(pegasus_model_name)
    pegasus_model = PegasusForConditionalGeneration.from_pretrained(pegasus_model_name).to(DEVICE)
    
    pegasus_summaries = []
    for art in tqdm(articles, desc="Pegasus"):
        inputs = pegasus_tokenizer(art, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            summary_ids = pegasus_model.generate(inputs['input_ids'], max_length=150, num_beams=2, early_stopping=True)
        pegasus_summaries.append(pegasus_tokenizer.decode(summary_ids[0], skip_special_tokens=True))
    
    results['Pegasus_Summary'] = pegasus_summaries
    print("Pegasus Scores:", calculate_rouge(pegasus_summaries, references))
except Exception as e:
    print(f"Pegasus Failed: {e}")
    results['Pegasus_Summary'] = ["Error"] * len(articles)

# --- 3. CUSTOM LSTM ---
print("\nStep 3: Evaluating Custom LSTM...")
lstm_path = os.path.join(MODELS_DIR, 'custom', 'custom_model.pth')
vocab_path = os.path.join(MODELS_DIR, 'custom', 'vocab.pkl')

if os.path.exists(lstm_path):
    try:
        custom_mod = import_from_file("custom_mod", os.path.join("src", "02_custom_model_training.py"))
        
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
            
        enc = custom_mod.Encoder(vocab.vocab_size, 64, 64, 0.5) 
        attn = custom_mod.Attention(64)
        dec = custom_mod.Decoder(vocab.vocab_size, 64, 64, 0.5, attn)
        model_lstm = custom_mod.Seq2Seq(enc, dec, DEVICE).to(DEVICE)
        
        model_lstm.load_state_dict(torch.load(lstm_path, map_location=DEVICE))
        
        lstm_summaries = []
        for art in tqdm(articles, desc="LSTM"):
            model_lstm.eval()
            tokens = [vocab.word2index.get(t, vocab.word2index.get('<UNK>', 0)) for t in art.split()]
            if not tokens: tokens = [0]
            src_tensor = torch.LongTensor(tokens).unsqueeze(1).to(DEVICE) # [src_len, 1]

            with torch.no_grad():
                enc_out, hidden, cell = model_lstm.encoder(src_tensor)
                
                # Logic from Seq2Seq.forward
                hidden = torch.tanh(model_lstm.encoder.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))) 
                cell = (cell[0] + cell[1]).unsqueeze(0) 
                
                input_tok = torch.tensor([vocab.word2index.get('<SOS>', 1)]).to(DEVICE)
                
                decoded_words = []
                for _ in range(50):
                    output, hidden, cell = model_lstm.decoder(input_tok, hidden, cell, enc_out)
                    top1 = output.argmax(1)
                    idx = top1.item()
                    if idx == vocab.word2index.get('<EOS>', 2):
                        break
                    decoded_words.append(vocab.index2word.get(idx, ''))
                    input_tok = top1

            lstm_summaries.append(" ".join(decoded_words))
            
        results['LSTM_Summary'] = lstm_summaries
        print("LSTM Scores:", calculate_rouge(lstm_summaries, references))
    except Exception as e:
        print(f"LSTM Failed: {e}")
        import traceback
        traceback.print_exc()
        results['LSTM_Summary'] = ["Error"] * len(articles)
else:
    print("LSTM model file not found.")
    results['LSTM_Summary'] = ["Missing"] * len(articles)

# --- 4. TRANSFORMER (Word) ---
print("\nStep 4: Evaluating Transformer (Scratch)...")
tf_path = os.path.join(MODELS_DIR, 'transformer_scratch', 'transformer_model.pth')
vocab_tf_path = os.path.join(MODELS_DIR, 'transformer_scratch', 'vocab.pkl')

if os.path.exists(tf_path):
    try:
        tf_mod = import_from_file("tf_mod", os.path.join("src", "05_transformer_training.py"))
        
        with open(vocab_tf_path, 'rb') as f:
            vocab_tf = pickle.load(f)
            
        pad_idx = vocab_tf.word2index.get('<PAD>', 0)
        # Init match src.05: dim=256, nhead=4, layers=3, dropout=0.1
        model_tf = tf_mod.TransformerModel(vocab_tf.vocab_size, 256, 4, 3, 3, 0.1, pad_idx).to(DEVICE)
        model_tf.load_state_dict(torch.load(tf_path, map_location=DEVICE))
        
        tf_summaries = []
        for art in tqdm(articles, desc="Transformer"):
            model_tf.eval()
            tokens = [vocab_tf.word2index.get(t, vocab_tf.word2index.get('<UNK>', 3)) for t in art.split()]
            if not tokens: tokens = [3]
            src = torch.LongTensor(tokens).unsqueeze(0).to(DEVICE) # [1, seq]
            
            sos_idx = vocab_tf.word2index.get('<SOS>', 1)
            eos_idx = vocab_tf.word2index.get('<EOS>', 2)
            
            trg_indices = [sos_idx]
            for _ in range(50):
                trg = torch.LongTensor(trg_indices).unsqueeze(0).to(DEVICE)
                src_pad = model_tf.create_pad_mask(src, pad_idx)
                trg_pad = model_tf.create_pad_mask(trg, pad_idx)
                
                with torch.no_grad():
                    output = model_tf(src, trg, src_pad_mask=src_pad, trg_pad_mask=trg_pad)
                
                next_token = output[0, -1, :].argmax().item()
                trg_indices.append(next_token)
                if next_token == eos_idx:
                    break
            
            decoded = [vocab_tf.index2word.get(i, '') for i in trg_indices[1:-1]]
            tf_summaries.append(" ".join(decoded))
            
        results['Transformer_Word_Summary'] = tf_summaries
        print("Transformer Word Scores:", calculate_rouge(tf_summaries, references))
    except Exception as e:
        print(f"Transformer Word Failed: {e}")
        import traceback
        traceback.print_exc() 
        results['Transformer_Word_Summary'] = ["Error"] * len(articles)
else:
    print("Transformer Word model file not found.")
    results['Transformer_Word_Summary'] = ["Missing"] * len(articles)

# --- 5. TRANSFORMER (BPE) ---
print("\nStep 5: Evaluating Transformer (BPE)...")
bpe_path = os.path.join(MODELS_DIR, 'transformer_bpe', 'transformer_cnndm_epoch_5.pth')
if os.path.exists(bpe_path):
    try:
        from tokenizers import ByteLevelBPETokenizer
        
        # Define BPE classes from src.08 locally to avoid more import headaches
        class PositionalEncodingBPE(nn.Module):
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

        class TransformerModelBPE(nn.Module):
            def __init__(self, num_tokens, dim_model, num_heads, num_encoder_layers, num_decoder_layers, dropout_p, pad_idx):
                super().__init__()
                self.positional_encoder = PositionalEncodingBPE(dim_model, dropout_p, 2000)
                self.embedding = nn.Embedding(num_tokens, dim_model)
                self.transformer = nn.Transformer(d_model=dim_model, nhead=num_heads, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dropout=dropout_p, batch_first=True)
                self.out = nn.Linear(dim_model, num_tokens)
                self.pad_idx = pad_idx
            def forward(self, src, trg, src_pad_mask=None, trg_pad_mask=None): 
                pass # Not used for loading

        # Load Tokenizer
        tokenizer_bpe_dir = os.path.join(MODELS_DIR, 'transformer_bpe', 'tokenizer_bpe')
        tokenizer = ByteLevelBPETokenizer(
            os.path.join(tokenizer_bpe_dir, "vocab.json"),
            os.path.join(tokenizer_bpe_dir, "merges.txt")
        )
        pad_id = tokenizer.token_to_id("<pad>")
        
        # Init Model (4 layers)
        model_bpe = TransformerModelBPE(tokenizer.get_vocab_size(), 256, 4, 4, 4, 0.1, pad_id).to(DEVICE)
        
        model_bpe.load_state_dict(torch.load(bpe_path, map_location=DEVICE), strict=False)
        
        def create_pad_mask(matrix, pad_token): return (matrix == pad_token)
        
        bpe_summaries = []
        sos_id = tokenizer.token_to_id("<s>")
        eos_id = tokenizer.token_to_id("</s>")
        
        for art in tqdm(articles, desc="Transformer BPE"):
            enc_ids = tokenizer.encode(art).ids[:510]
            if not enc_ids: enc_ids = [pad_id]
            src = torch.LongTensor(enc_ids).unsqueeze(0).to(DEVICE)
            
            trg_indices = [sos_id]
            for _ in range(50):
                trg = torch.tensor([trg_indices]).to(DEVICE)
                src_pad = create_pad_mask(src, pad_id)
                trg_pad = create_pad_mask(trg, pad_id)
                
                # Manual Forward Logic for Inference (since class above was dummy for loading)
                # Replicating src.08 forward
                src_emb = model_bpe.embedding(src) * math.sqrt(256)
                trg_emb = model_bpe.embedding(trg) * math.sqrt(256)
                src_emb = model_bpe.positional_encoder(src_emb.permute(1,0,2)).permute(1,0,2)
                trg_emb = model_bpe.positional_encoder(trg_emb.permute(1,0,2)).permute(1,0,2)
                trg_mask = model_bpe.transformer.generate_square_subsequent_mask(trg.size(1)).to(DEVICE)
                output = model_bpe.transformer(src=src_emb, tgt=trg_emb, src_key_padding_mask=src_pad, tgt_key_padding_mask=trg_pad, memory_key_padding_mask=src_pad, tgt_mask=trg_mask)
                logits = model_bpe.out(output)
                
                next_token = logits[0, -1, :].argmax().item()
                trg_indices.append(next_token)
                if next_token == eos_id: break
            
            bpe_summaries.append(tokenizer.decode(trg_indices))
            
        results['Transformer_BPE_Summary'] = bpe_summaries
        print("BPE Scores:", calculate_rouge(bpe_summaries, references))

    except Exception as e:
        print(f"BPE Failed: {e}")
        import traceback
        traceback.print_exc()
        results['Transformer_BPE_Summary'] = ["Error"] * len(articles)
else:
    print("BPE model file not found.")
    results['Transformer_BPE_Summary'] = ["Missing"] * len(articles)

# Save
df = pd.DataFrame(results)
df.to_csv(RESULTS_FILE, index=False)
print("Saved comparison to:", RESULTS_FILE)
