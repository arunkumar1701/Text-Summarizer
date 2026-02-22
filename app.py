import streamlit as st
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from deep_translator import GoogleTranslator
from langdetect import detect
import os
import time

# Set page config FIRST
st.set_page_config(page_title="AI Text Summarizer", page_icon="✨", layout="wide")

# Inject Custom CSS for Glassmorphism
st.markdown("""
<style>
/* Background Image - Modern dark gradient with some abstract shapes or just an elegant image */
.stApp {
    background: url("https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?q=80&w=2564&auto=format&fit=crop") center/cover no-repeat fixed;
}

/* Global Text Colors */
.stApp, html, body {
    color: #ffffff !important;
}
p, h1, h2, h3, h4, h5, h6, label, span, div {
    color: #ffffff !important;
}

/* Glassmorphism for Columns and Header */
div[data-testid="column"], .glass-header {
    background: rgba(20, 20, 30, 0.4) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border-radius: 20px !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    padding: 25px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4) !important;
    margin-bottom: 20px;
}

/* Make markdown wrapper keep elements in flow */
.stMarkdown {
    margin-bottom: 0px !important;
}

/* Inputs / Text Areas styling */
.stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div, .stTextInput input {
    background-color: rgba(0, 0, 0, 0.3) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    color: #ffffff !important;
    border-radius: 12px !important;
}

.stTextArea textarea:focus, .stSelectbox div[data-baseweb="select"] > div:focus-within {
    border-color: rgba(255, 255, 255, 0.7) !important;
    box-shadow: 0 0 15px rgba(255,255,255,0.2) !important;
}

/* Button styling */
.stButton > button {
    background: linear-gradient(135deg, rgba(255,255,255,0.15), rgba(255,255,255,0.05)) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    letter-spacing: 1px !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    background: rgba(255,255,255,0.25) !important;
    border: 1px solid rgba(255, 255, 255, 0.6) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5) !important;
}

/* File Uploader styling */
div[data-testid="stFileUploader"] {
    background-color: rgba(0, 0, 0, 0.2) !important;
    border: 1px dashed rgba(255, 255, 255, 0.3) !important;
    border-radius: 15px;
    padding: 15px;
}
div[data-testid="stFileUploader"]:hover {
    border-color: rgba(255, 255, 255, 0.6) !important;
}

/* Hide top right items for cleaner look */
header[data-testid="stHeader"] {
    background: transparent !important;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Custom Spinner matching glass theme */
.stSpinner > div > div {
    border-color: #ffffff rgba(255,255,255,0.2) rgba(255,255,255,0.2) rgba(255,255,255,0.2) !important;
}

/* Dropdown list items */
li[role="option"] {
    background: rgba(30, 30, 40, 0.95) !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PEGASUS_PATH = os.path.join("artifacts", "models", "pegasus")

@st.cache_resource(show_spinner="Loading Pegasus Model (One-time Setup)...")
def load_pegasus_model():
    path = PEGASUS_PATH if os.path.exists(PEGASUS_PATH) else "google/pegasus-cnn_dailymail"
    tokenizer = PegasusTokenizer.from_pretrained(path)
    model = PegasusForConditionalGeneration.from_pretrained(path).to(DEVICE)
    return tokenizer, model

def summarize(text, tokenizer, model):
    inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt").to(DEVICE)
    # Using specific generation parameters suited for pegasus
    summary_ids = model.generate(
        inputs["input_ids"], 
        max_length=150, 
        min_length=30, 
        num_beams=4, 
        length_penalty=2.0, 
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def translate_with_retry(translator, text, max_retries=3):
    for attempt in range(max_retries):
        try:
            return translator.translate(text)
        except Exception as e:
            if attempt == max_retries - 1:
                st.warning(f"⚠️ Translation service connection failed. Showing original translation.")
                return text
            time.sleep(1.5)

def translate_text(text, target_lang):
    if target_lang == "en":
        return text
    translator = GoogleTranslator(source='auto', target=target_lang)
    return translate_with_retry(translator, text)

def detect_and_translate_to_english(text):
    try:
        lang = detect(text)
    except:
        lang = 'en'
        
    if lang != 'en':
        translator = GoogleTranslator(source=lang, target='en')
        translated = translate_with_retry(translator, text)
        return translated, lang
    return text, 'en'

# Load Model
tokenizer, model = load_pegasus_model()

# Header
st.markdown('<div class="glass-header"><h1 style="text-align: center; margin:0; padding-bottom:5px; font-weight:800; letter-spacing: 1px;">✨ Nexus AI: Text Summarization</h1><p style="text-align: center; color:#dcdcdc; margin:0; font-size: 1.1em;">Powered by State-of-the-Art Pegasus Model & Glassmorphism UI</p></div>', unsafe_allow_html=True)

col1, col2 = st.columns([1.1, 1], gap="large")

with col1:
    st.markdown('<h3>📥 Input Text</h3>', unsafe_allow_html=True)
    input_text = st.text_area("Enter your article or text here:", height=300, placeholder="Paste your long text here to get a concise summary...")
    
    st.markdown('<p style="margin: 10px 0 5px 0; font-size: 0.9em; font-weight: bold;">Or upload a text document:</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=['txt'], label_visibility="collapsed")
    if uploaded_file:
        input_text = str(uploaded_file.read(), 'utf-8')

with col2:
    st.markdown('<h3>⚙️ Settings & Output</h3>', unsafe_allow_html=True)
    
    target_language = st.selectbox("🌐 Target Language", ["English", "Tamil", "Telugu", "Hindi", "French", "Spanish", "German"])
    lang_map = {
        "English": "en", "Tamil": "ta", "Telugu": "te", "Hindi": "hi", 
        "French": "fr", "Spanish": "es", "German": "de"
    }
    
    st.markdown("<br>", unsafe_allow_html=True)
    generate_btn = st.button("🚀 Generate Summary", use_container_width=True)
    
    if generate_btn and input_text:
        with st.spinner("Analyzing and summarizing..."):
            # 1. Detect & Translate to English
            english_text, original_lang = detect_and_translate_to_english(input_text)
            
            # 2. Summarize using Pegasus
            summary_en = summarize(english_text, tokenizer, model)
            
            # 3. Translate back to Target Language
            target_code = lang_map[target_language]
            final_summary = translate_text(summary_en, target_code)
            
            st.success("✅ Summary Generated Successfully!")
            st.markdown(f'<p style="font-size:0.85em; color:#e0e0e0; margin-bottom:5px;"><i>Detected Original Language: {original_lang.upper()}</i></p>', unsafe_allow_html=True)
            st.text_area("Result:", value=final_summary, height=200, label_visibility="collapsed")
            
    elif generate_btn and not input_text:
        st.warning("⚠️ Please enter some text or upload a file first.")
