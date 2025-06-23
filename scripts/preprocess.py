import re
import os
import pandas as pd
import nltk
from langdetect import detect, DetectorFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import emoji
from tqdm import tqdm
import requests
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Setup
eng_stop_words = set(stopwords.words('english'))
DetectorFactory.seed = 0  # Deterministic language detection
tqdm.pandas()

# Load Amharic stopwords (URL with local fallback)
def load_amharic_stop_words(local_path='../data/amharic-stop-words/StopWord-list.txt'):
    stop_words_urls = [
        "https://www.irit.fr/AmharicResources/wp-content/uploads/2021/03/StopWord-list.txt",
        "https://raw.githubusercontent.com/geeztypes/stopwords-am/main/stopwords-am.txt"
    ]
    stop_words = set()
    
    # Try URLs first
    for url in stop_words_urls:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            stop_words.update(line.strip() for line in response.text.splitlines() if line.strip())
            logger.info(f"Loaded stopwords from {url}")
        except requests.RequestException as e:
            logger.warning(f"Failed to load stopwords from {url}: {e}")
    
    # Fallback to local file
    if not stop_words and os.path.exists(local_path):
        try:
            with open(local_path, 'r', encoding='utf-8') as f:
                stop_words.update(line.strip() for line in f if line.strip())
            logger.info(f"Loaded stopwords from {local_path}")
        except Exception as e:
            logger.error(f"Failed to load local stopwords: {e}")
    
    return stop_words

# # Emoji handling
# def extract_emojis(text):
    if not isinstance(text, str):
        return ""
    return ' '.join(emoji.distinct_emoji_list(text))

def remove_emojis(text):
    if not isinstance(text, str):
        return ""
    return emoji.replace_emoji(text, replace='')

# Language detection
def has_amharic(text):
    return any('\u1200' <= char <= '\u137F' for char in text)

def has_english(text):
    return bool(re.search(r'[A-Za-z]', text))

def detect_language(text):
    text = remove_emojis(text)
    if not isinstance(text, str) or not text.strip():
        return 'unknown'
    if has_amharic(text) and has_english(text):
        return 'mixed'
    elif has_amharic(text):
        return 'am'
    elif has_english(text):
        return 'en'
    else:
        return 'unknown'

# Amharic normalization
def normalize_amharic(text):
    if not isinstance(text, str):
        return ""
    rules = [
        (r'[ሃኅኃሐሓኻ]', 'ሀ'), (r'[ሑኁ]', 'ሁ'), (r'[ሒኂ]', 'ሂ'),
        (r'[ሓኃ]', 'ሃ'), (r'[ሔኄ]', 'ሄ'), (r'[ሕኅ]', 'ህ'),
        (r'[ሖኆ]', 'ሆ'), (r'[ኈሠ]', 'ለ'), (r'[ሷዋ]', 'ዋ'),
        (r'[ዩ]', 'ዩ'), (r'[ዪ]', 'ያ'), (r'[ዳድ]', 'ድ'),
        (r'[ጸፀ]', 'ፀ'), (r'[ጹፁ]', 'ፁ'), (r'[ጺፂ]', 'ፂ'),
        (r'[ጻፃ]', 'ፃ'), (r'[ጼፄ]', 'ፄ'), (r'[ጽፅ]', 'ፅ'),
        (r'[ጾፆ]', 'ፆ'),
    ]
    for pattern, replacement in rules:
        text = re.sub(pattern, replacement, text)
    return re.sub(r'\s+', ' ', text.strip())

# Remove punctuation
def remove_punctuation(text):
    punctuation = r'[፡።፤፥፦፧፨፠፣!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~]'
    return re.sub(punctuation, '', text)

# Amharic tokenizer
def amharic_tokenize(text):
    if not isinstance(text, str):
        return []
    return re.findall(r'\b[\u1200-\u137F\w]+\b', text)

# Preprocessing pipelines
def preprocess_amharic(text, stop_words):
    text = normalize_amharic(text)
    text = remove_punctuation(text)
    tokens = amharic_tokenize(text)
    return [t for t in tokens if t not in stop_words]

def preprocess_english(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    return [t for t in tokens if t not in eng_stop_words]

# Mixed language preprocessing
def preprocess_mixed_text(text, am_stop_words):
    if not isinstance(text, str):
        return ''
    text_clean = remove_emojis(text)
    lang = detect_language(text)

    if lang == 'am':
        return ' '.join(preprocess_amharic(text_clean, am_stop_words))
    elif lang == 'en':
        return ' '.join(preprocess_english(text_clean))
    elif lang == 'mixed':
        am_tokens = preprocess_amharic(text_clean, am_stop_words)
        en_tokens = preprocess_english(text_clean)
        return ' '.join(am_tokens + [f"[EN:{t}]" for t in en_tokens])  # Tag English tokens
    else:
        return ''

# Process CSV
def process_csv(input_csv, output_csv):
    logger.info("Loading Amharic stopwords")
    am_stop_words = load_amharic_stop_words()
    
    logger.info(f"Reading input CSV: {input_csv}")
    try:
        df = pd.read_csv(input_csv)
        df['Message Text'] = df['Message Text'].fillna('')
    except FileNotFoundError:
        logger.error(f"Input CSV not found: {input_csv}")
        raise

    logger.info("Detecting language")
    df['Language'] = df['Message Text'].progress_apply(detect_language)
    
    # logger.info("Extracting emojis")
    # df['Emojis'] = df['Message Text'].progress_apply(extract_emojis)
    
    logger.info("Preprocessing text")
    df['Preprocessed Text'] = df['Message Text'].progress_apply(
        lambda x: preprocess_mixed_text(x, am_stop_words)
    )
    df =df.drop(columns=['Message Text'], axis=1)
    
    logger.info(f"Saving output CSV: {output_csv}")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    logger.info(f"✅ Done! Preprocessed data saved to: {output_csv}")
    return df


