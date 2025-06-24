import csv
import json
import re
import os

input_file = "data/processed_media/cleaned_telegram_messages.csv"
output_file = "data/labeled/amharic_labeled.conll.txt"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

def is_amharic(token):
    return bool(re.search(r'[\u1200-\u137F]', token))

def is_english(token):
    return bool(re.match(r'[A-Za-z]', token))

def label_tokens(tokens):
    labels = ['O'] * len(tokens)

    # 1. PRODUCT → first 1–4 English tokens
    for i in range(min(5, len(tokens))):
        if is_english(tokens[i]) and tokens[i].lower() not in ['size', 'price']:
            if labels[i] == 'O':
                labels[i] = 'B-Product' if i == 0 or labels[i-1] == 'O' else 'I-Product'
        else:
            break

    # 2. PRICE → after 'Price' or 'በ', before 'birr' or 'ብር'
    for i, token in enumerate(tokens):
        if token.lower() == "price" or token == "በ":
            if i+1 < len(tokens) and re.match(r'^\d{2,6}$', tokens[i+1]):
                labels[i] = 'B-PRICE'
                labels[i+1] = 'I-PRICE'
                if i+2 < len(tokens) and tokens[i+2].lower() in ['birr', 'ብር']:
                    labels[i+2] = 'I-PRICE'

    # 3. LOCATION → after 'አድራሻ'
    for i, token in enumerate(tokens):
        if token == "አድራሻ":
            labels[i] = 'B-LOC'
            for j in range(1, 4):
                if i+j < len(tokens) and is_amharic(tokens[i+j]):
                    labels[i+j] = 'I-LOC'
                else:
                    break

    # 4. CONTACT_INFO → 09xxxxxxxx or @username
    for i, token in enumerate(tokens):
        if re.match(r'^09\d{8}$', token) or token.startswith("@"):
            labels[i] = 'B-CONTACT_INFO'

    return labels

# 🚀 Read CSV and write CoNLL
with open(input_file, "r", encoding="utf-8") as infile, \
     open(output_file, "w", encoding="utf-8") as outfile:

    reader = csv.DictReader(infile)

    for row in reader:
        try:
            tokens = json.loads(row["tokens"])
        except Exception:
            continue

        labels = label_tokens(tokens)

        for token, label in zip(tokens, labels):
            outfile.write(f"{token} {label}\n")
        outfile.write("\n")

print("✅ Rule-based labeling complete. Saved to:", output_file)