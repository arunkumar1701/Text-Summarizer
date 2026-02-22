
import os
import pandas as pd
import torch
import torch.nn as nn
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from rouge_score import rouge_scorer
import pickle
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Re-define/Import Custom Model Classes ---
# (Ideally these should be in a shared module, replicating here for standalone execution)
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

class Vocabulary: # Needed for inference
    def __init__(self):
        self.word2index = {}
        self.index2word = {}
        self.vocab_size = 0
    # ... methods match training ...

def load_custom_model(vocab_path, model_path):
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
        
    enc = Encoder(vocab.vocab_size, 256, 512, 0.5)
    attn = Attention(512)
    dec = Decoder(vocab.vocab_size, 256, 512, 0.5, attn)
    model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        logging.info("Custom model loaded.")
        return model, vocab
    else:
        logging.warning("Custom model checkpoint not found!")
        return None, None

def custom_inference(model, vocab, text, max_len=50):
    model.eval()
    tokens = text.split()
    indices = [vocab.word2index.get(t, vocab.word2index['<UNK>']) for t in tokens]
    src_tensor = torch.tensor(indices).unsqueeze(1).to(DEVICE) # [len, 1]
    
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor)
        # Handle bidirectional cell state: sum forward and backward and reshape for unidirectional decoder
        cell = (cell[0] + cell[1]).unsqueeze(0)
        
    trg_indexes = [vocab.word2index['<SOS>']]
    
    for i in range(max_len):
        trg_tensor = torch.tensor([trg_indexes[-1]]).to(DEVICE)
        
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell, encoder_outputs)
        
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        
        if pred_token == vocab.word2index['<EOS>']:
            break
            
    trg_tokens = [vocab.index2word[i] for i in trg_indexes]
    return " ".join(trg_tokens[1:-1]) # remove SOS/EOS

def evaluate():
    data_path = os.path.join("artifacts", "data")
    test_df = pd.read_csv(os.path.join(data_path, "test.csv")).head(20) # restrict to 20 for speed demo
    
    # 1. Custom Model
    custom_model_dir = os.path.join("artifacts", "models", "custom")
    custom_model, vocab = load_custom_model(os.path.join(custom_model_dir, "vocab.pkl"), 
                                            os.path.join(custom_model_dir, "custom_model.pth"))
    
    # 2. Pegasus Model
    pegasus_dir = os.path.join("artifacts", "models", "pegasus-samsum")
    if not os.path.exists(pegasus_dir):
        pegasus_dir = "google/pegasus-cnn_dailymail"
    
    logging.info(f"Loading Pegasus from {pegasus_dir}")
    pegasus_tokenizer = PegasusTokenizer.from_pretrained(pegasus_dir)
    pegasus_model = PegasusForConditionalGeneration.from_pretrained(pegasus_dir).to(DEVICE)
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    results = []
    
    logging.info("Starting Evaluation...")
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        dialogue = row['dialogue']
        reference = row['summary']
        
        # Custom Pred
        custom_pred = ""
        if custom_model:
            custom_pred = custom_inference(custom_model, vocab, dialogue)
            
        # Pegasus Pred
        inputs = pegasus_tokenizer(dialogue, max_length=512, truncation=True, return_tensors="pt").to(DEVICE)
        summary_ids = pegasus_model.generate(inputs["input_ids"], max_length=128, num_beams=4, length_penalty=2.0)
        pegasus_pred = pegasus_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Calculate Scores
        c_scores = scorer.score(reference, custom_pred) if custom_model else None
        p_scores = scorer.score(reference, pegasus_pred)
        
        results.append({
            "dialogue": dialogue,
            "reference": reference,
            "custom_summary": custom_pred,
            "pegasus_summary": pegasus_pred,
            "custom_rouge1": c_scores['rouge1'].fmeasure if c_scores else 0,
            "pegasus_rouge1": p_scores['rouge1'].fmeasure
        })
        
    # Save Results
    eval_dir = os.path.join("artifacts", "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(os.path.join(eval_dir, "comparison.csv"), index=False)
    logging.info(f"Evaluation completed. saved to {eval_dir}")

if __name__ == "__main__":
    evaluate()
