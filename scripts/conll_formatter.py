import re
import os
import logging
import pandas as pd
from transformers import pipeline
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define patterns for entity recognition
PRICE_PATTERNS = [
    r'\bዋጋ[:፡።\-]?\s*\d+(?:[,.\d]+)?\s*(ብር|ETB|birr)?\b',
    r'\bዋጋ\d+(?:[,.\d]+)?(?:ਬር|ETB|birr)?\b',
    r'\bበ\s*\d+(?:[,.\d]+)?\s*(ብር|ETB|birr)\b',
    r'\b\d+\s*(ብር|ETB|birr)\b',
    r'\bPrice[:፡]?\s*\d+(?:[,.\d]+)?\s*(ETB|birr)?\b',
    r'\b\d+(?:[,.\d]+)?(?:ਬር|ETB|birr)\b'
]

LOCATION_PATTERNS = [
    r'(አዲስ\s*አበባ|ቦሌ|መስከረም|ቤተክርስቲያን|ለቡ|መገናኛ|ፒያሳ|ሞል|ሊፍቱ|ህንፃ|ዛምሞል|ታሜ|ጋስ)',
    r'አድራሻ\s*[^\s]+(?:\s+[^\s]+)?',
    r'በ\s*(ቦሌ|መስከረም|ለቡ|መገናኛ|ፒያሳ|መድህኒአለም|ቤተክርስቲያን)(?:\s+[^\s]+)?',
    r'(Addis\s*Ababa|Bole|Piasa|Megenagna|Lebu)'
]

PRODUCT_PATTERNS = [
    r'(GROOMING\s*SET|Steam\s*Iron|Hot\s*Plate|Bath\s*Brush|Glass\s*Water\s*Set|Juicer|Bottle|Shaver|Treadmill)',
    r'(ማሽን|ሼበር|ቶንዶስ|መቶከሻ|ጫማ|ቀሚስ|የፀጉር\s*ቶንዶስ|ልብስ|ቦርሳ|ኮት|የልጆች\s*ጫማ|የሴቶች\s*ቀሚስ)',
    r'የ[^\s]+(?:\s+[^\s]+)?\s*(ማሽን|መሣሪያ|ቁሳቁስ)'
]

# Tokenize text
def tokenize(text):
    if not isinstance(text, str):
        return []
    text = re.sub(r'\[EN:[^\]]*\]|\[|\]', '', text).strip()
    price_matches = []
    for pattern in PRICE_PATTERNS:
        matches = re.finditer(pattern, text, re.UNICODE)
        for match in matches:
            price_matches.append((match.start(), match.end(), match.group()))
    price_matches = sorted(price_matches, key=lambda x: x[0], reverse=True)
    for idx, (start, end, match) in enumerate(price_matches):
        placeholder = f"__PRICE_{idx}__"
        text = text[:start] + placeholder + text[end:]
    tokens = re.findall(r'\b[\u1200-\u137F\w]+(?:\s*[\u1200-\u137F\w]+)?\b|[^\s]', text)
    tokens = [t.strip() for t in tokens if t.strip()]
    final_tokens = []
    for token in tokens:
        if token.startswith('__PRICE_') and token.endswith('__'):
            try:
                idx = int(token.replace('__PRICE_', '').replace('__', ''))
                if idx < len(price_matches):
                    final_tokens.append(price_matches[idx][2])
                else:
                    logger.warning(f"Invalid placeholder token index: {token}")
                    final_tokens.append(token)  # Keep the invalid token as is
            except ValueError:
                logger.warning(f"Invalid placeholder token: {token}")
                final_tokens.append(token)  # Keep the invalid token as is
        else:
            final_tokens.append(token)
    return final_tokens

# Load NER model
def load_ner_model():
    try:
        ner_pipeline = pipeline(
            "ner",
            model="Davlan/afro-xlmr-mini",
            tokenizer="Davlan/afro-xlmr-mini",
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1
        )
        logger.info("Loaded afro-xlmr-mini NER model")
        return ner_pipeline
    except Exception as e:
        logger.warning(f"Failed to load NER model: {e}. Falling back to regex.")
        return None

# Label tokens
def label_tokens(tokens, text, ner_pipeline=None):
    labels = ['O'] * len(tokens)
    text_clean = re.sub(r'\[EN:[^\]]*\]|\[|\]', '', text).lower()
    for pattern in PRICE_PATTERNS:
        for match in re.finditer(pattern, text_clean, re.UNICODE | re.IGNORECASE):
            match_text = match.group()
            match_tokens = tokenize(match_text)
            if not match_tokens:
                continue
            for i in range(len(tokens) - len(match_tokens) + 1):
                if tokens[i:i+len(match_tokens)] == match_tokens:
                    labels[i] = 'B-PRICE'
                    for j in range(1, len(match_tokens)):
                        if i + j < len(tokens):
                            labels[i + j] = 'I-PRICE'
                    break
    for i, token in enumerate(tokens):
        for pattern in LOCATION_PATTERNS:
            if re.match(pattern, token, re.UNICODE | re.IGNORECASE):
                labels[i] = 'B-LOC'
                for j in range(1, 3):
                    if i + j < len(tokens) and re.search(r'[\u1200-\u137F]', tokens[i + j]):
                        labels[i + j] = 'I-LOC'
        for pattern in PRODUCT_PATTERNS:
            if re.search(pattern, token, flags=re.UNICODE | re.IGNORECASE):
                labels[i] = 'B-PRODUCT'
                for j in range(1, 3):
                    if i + j < len(tokens) and re.search(r'[\u1200-\u137F\w]', tokens[i + j]):
                        labels[i + j] = 'I-PRODUCT'
    if ner_pipeline:
        try:
            ner_results = ner_pipeline(text_clean)
            token_idx = 0
            for entity in ner_results:
                entity_text = entity['word']
                entity_label = entity.get('entity', entity.get('entity_group', 'O'))
                if 'PRODUCT' in entity_label:
                    target_label = 'B-PRODUCT'
                elif 'LOC' in entity_label:
                    target_label = 'B-LOC'
                elif 'MONEY' in entity_label or 'PRICE' in entity_label:
                    target_label = 'B-PRICE'
                else:
                    continue
                entity_tokens = tokenize(entity_text)
                for et in entity_tokens:
                    while token_idx < len(tokens) and tokens[token_idx] != et:
                        token_idx += 1
                    if token_idx < len(tokens):
                        labels[token_idx] = target_label
                        token_idx += 1
                        for j in range(1, len(entity_tokens)):
                            if token_idx < len(tokens):
                                labels[token_idx] = target_label.replace('B-', 'I-')
                                token_idx += 1
        except Exception as e:
            logger.warning(f"NER model failed for text: {text}. Error: {e}")
    for i in range(1, len(labels)):
        if labels[i].startswith('I-') and labels[i-1] == 'O':
            labels[i] = labels[i].replace('I-', 'B-')
    return labels

# Format CoNLL
def format_conll(tokens, tags):
    return "\n".join(f"{tok}\t{tag}" for tok, tag in zip(tokens, tags)) + "\n"

# Process CoNLL
def process_conll(csv_path, output_path, limit=50):
    if not os.path.exists(csv_path):
        logger.error(f"CSV not found at {csv_path}")
        return
    df = pd.read_csv(csv_path).fillna('')
    logger.info(f"📂 Loaded {len(df)} messages")
    ner_pipeline = load_ner_model()
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
        if not tokens:
            continue
        labels = label_tokens(tokens, text, ner_pipeline)
        conll_lines.append(format_conll(tokens, labels))
        count += 1
        logger.info(f"Processed: {count}/{limit}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(conll_lines))
    logger.info(f"✅ CoNLL data saved: {output_path}")


