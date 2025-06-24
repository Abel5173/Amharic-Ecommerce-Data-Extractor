import csv
import os
import sys
import json

# Add scripts/ to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from cleaner import clean_amharic_text
from tokenizer import tokenize_amharic_text

input_file = "data/raw_media/telegram_messages.csv"
output_file = "data/processed_media/cleaned_telegram_messages.csv"

# Make sure the output folder exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(input_file, "r", encoding="utf-8") as infile, \
     open(output_file, "w", newline="", encoding="utf-8") as outfile:

    reader = csv.DictReader(infile)
    fieldnames = ["channel", "message_id", "sender_id", "timestamp", "cleaned_text", "tokens"]
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        cleaned = clean_amharic_text(row["text"])
        tokens = tokenize_amharic_text(cleaned)

        writer.writerow({
            "channel": row["channel"],
            "message_id": row["message_id"],
            "sender_id": row["sender_id"],
            "timestamp": row["timestamp"],
            "cleaned_text": cleaned,
            "tokens": json.dumps(tokens, ensure_ascii=False)
        })

print("✅ Cleaned and tokenized data saved to:", output_file)
