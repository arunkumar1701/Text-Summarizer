
import os
import re
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_text(text):
    """
    Cleans text by removing special characters, URLs, and normalizing whitespace.
    """
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove emojis and special characters (keep alphanumeric and basic punctuation)
    text = re.sub(r'[^\w\s.,!?\']', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def data_ingestion():
    logging.info("Starting Data Ingestion...")
    
    # 1. Download Dataset
    try:
        logging.info("Downloading SAMSUM dataset from knkarthick/samsum...")
        dataset = load_dataset('knkarthick/samsum')
    except Exception as e:
        logging.error(f"Error downloading dataset: {e}")
        logging.info("Generating DUMMY dataset for demonstration...")
        # Create dummy data
        data = {
            'dialogue': [
                "Hi, how are you? I'm good. How about you? good too.", 
                "Can we meet at 5? Sure, see you there. Bye.",
                "Did you see the match? Yes, it was amazing! What a goal by Messi.",
                "Mom called. What did she say? She wants us to visit. Okay.",
                "Let's go for lunch. Pizza? No, burgers. Fine, burgers."
            ] * 20,
            'summary': [
                "Two people exchange greetings.",
                "They agree to meet at 5 pm.",
                "They discuss the football match.",
                "Mom wants them to visit.",
                "They decide to have burgers for lunch."
            ] * 20,
            'id': [str(i) for i in range(100)]
        }
        train_df = pd.DataFrame(data)
        val_df = pd.DataFrame(data).iloc[:10]
        test_df = pd.DataFrame(data).iloc[:10]
        
        # Mock the split logic for dummy
        full_df = pd.concat([train_df, val_df, test_df])
        
        # Continue with processing
        logging.info("Cleaning data...")
        full_df['dialogue'] = full_df['dialogue'].apply(clean_text)
        full_df['summary'] = full_df['summary'].apply(clean_text)
        
        logging.info("Splitting data into 80/10/10...")
        train_df, temp_df = train_test_split(full_df, test_size=0.2, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        
        logging.info(f"Train size: {len(train_df)}")
        logging.info(f"Val size: {len(val_df)}")
        logging.info(f"Test size: {len(test_df)}")
        
        output_dir = os.path.join("artifacts", "data")
        os.makedirs(output_dir, exist_ok=True)
        
        train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
        
        logging.info(f"Dummy Data saved to {output_dir}")
        return

    # 2. Convert to DataFrame for easier handling
    train_data = pd.DataFrame(dataset['train'])
    val_data = pd.DataFrame(dataset['validation'])
    test_data = pd.DataFrame(dataset['test'])
    
    # Combine to clean and re-split if needed, but SAMSUM is already split. 
    # However, the workflow mentions "Split dataset: 80% train, 10% val, 10% test". 
    # SAMSUM comes splits: Train(14732), Val(818), Test(819). 
    # Let's strictly follow the workflow instruction to clean tokens and split as requested (approx).
    
    # Concatenate all to re-split according to custom logic if strictly needed, 
    # or just use the provided splits if they match the ratio reasonably. 
    # The workflow says "80% train (11,801 samples), 10% validation (1,475 samples), 10% test (1,475 samples)".
    # SAMSUM total ~16k. 
    # 14732+818+819 = 16369.
    # 11801 is approx 72%. 
    # Let's re-split to match the workflow EXACTLY.
    
    full_df = pd.concat([train_data, val_data, test_data])
    
    # 3. Clean Data
    logging.info("Cleaning data...")
    full_df['dialogue'] = full_df['dialogue'].apply(clean_text)
    full_df['summary'] = full_df['summary'].apply(clean_text)
    
    # 4. Split Data (80/10/10)
    logging.info("Splitting data into 80/10/10...")
    train_df, temp_df = train_test_split(full_df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    logging.info(f"Train size: {len(train_df)}")
    logging.info(f"Val size: {len(val_df)}")
    logging.info(f"Test size: {len(test_df)}")
    
    # 5. Save locally
    output_dir = os.path.join("artifacts", "data")
    os.makedirs(output_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    logging.info(f"Data saved to {output_dir}")

if __name__ == "__main__":
    data_ingestion()
