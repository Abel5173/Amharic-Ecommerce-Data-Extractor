import re
import os
import pandas as pd
import logging
from transformers import pipeline
import nltk

# -------------------- 🔧 SETUP -------------------- #
nltk.download('punkt', quiet=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -------------------- 📦 PATTERNS -------------------- #
PRICE_PATTERNS = [
    r'ዋጋ[:፡።\-]?\s*\d+(?:[,.\d]+)?\s*ብር?', r'በ\s*\d+(?:[,.\d]+)?\s*ብር', r'\d+\s*ብር'
]

LOCATION_PATTERNS = [
    r'(?:አዲስ\s*አበባ|ቦሌ|መስከረም|ቤተክርስቲያን|ሞል|ሊፍቱ|ህንፃ|ጎን)',  # Common Amharic locations
    r'በ[^\s]+(?:\s+[^\s]+)?',  # በቦሌ ሊፍቱ
    r'[^\s]+(?:\s+[^\s]+)?\s*(አካባቢ|ላይ)'  # ቦሌ አካባቢ
]

PRODUCT_PATTERNS = [
    r'(GROOMING SET|Steam Iron|Hot plate|Bath brush|glass water set|juicer|bottle)',  # EN
    r'(ማሽን|ሼቨር|ቶንዶስ|መቶከሻ|ጫማ|ቀሚስ|የፀጉር ቶንዶስ|ልብስ)'  # AM
]

# -------------------- 🤖 NER MODEL -------------------- #
try:
    ner = pipeline("ner", model="Davlan/bert-base-multilingual-cased-ner-hrl", grouped_entities=True)
    logger.info("✅ HuggingFace NER loaded.")
except Exception as e:
    ner = None
    logger.warning(f"❌ Could not load NER model: {e}")

# -------------------- ✂️ TOKENIZER -------------------- #
def tokenize(text):
    if not isinstance(text, str):
        return []
    return re.findall(r'\b[\u1200-\u137F\w]+\b', text)

# -------------------- 🏷️ BIO TAGGING -------------------- #
def bio_tagging(text, tokens):
    tags = ['O'] * len(tokens)

    def mark_entities(tag, match):
        match_tokens = tokenize(match.group())
        for i in range(len(tokens) - len(match_tokens) + 1):
            if tokens[i:i+len(match_tokens)] == match_tokens:
                tags[i] = f"B-{tag}"
                for j in range(1, len(match_tokens)):
                    tags[i + j] = f"I-{tag}"
                break

    # 🧠 Rule-based tagging
    for pattern in PRICE_PATTERNS:
        for m in re.finditer(pattern, text): mark_entities("PRICE", m)

    for pattern in LOCATION_PATTERNS:
        for m in re.finditer(pattern, text): mark_entities("LOC", m)

    for pattern in PRODUCT_PATTERNS:
        for m in re.finditer(pattern, text): mark_entities("PRODUCT", m)

    # 🤖 Model-based tagging (only locations)
    if ner:
        try:
            model_results = ner(text)
            for ent in model_results:
                word = ent['word'].replace('##', '')
                for i, tok in enumerate(tokens):
                    if tok.startswith(word) and tags[i] == "O":
                        prefix = "B-LOC" if ent['entity_group'] == "LOC" else "O"
                        tags[i] = prefix
                        break
        except Exception as e:
            logger.warning(f"Model NER error: {e}")
    
    return tags

# -------------------- 📄 CoNLL Formatting -------------------- #
def format_conll(tokens, tags):
    return "\n".join(f"{tok}\t{tag}" for tok, tag in zip(tokens, tags)) + "\n"

# -------------------- 📊 Main Processor -------------------- #
def process_conll(csv_path, output_path, limit=50):
    if not os.path.exists(csv_path):
        logger.error(f"CSV not found at {csv_path}")
        return

    df = pd.read_csv(csv_path).fillna('')
    logger.info(f"📂 Loaded {len(df)} messages")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    count = 0
    conll_lines = []

    for _, row in df.iterrows():
        if count >= limit:
            break
        lang = row.get("Language", "")
        text = row.get("Message Text", row.get("Preprocessed Text", ""))
        if not isinstance(text, str) or lang not in ["am", "en", "mixed"]:
            continue
        tokens = tokenize(text)
        tags = bio_tagging(text, tokens)
        conll_lines.append(format_conll(tokens, tags))
        count += 1
        logger.info(f"Processed: {count}/{limit}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(conll_lines))
    logger.info(f"✅ CoNLL data saved: {output_path}")
