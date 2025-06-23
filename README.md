# Amharic E-commerce Data Extractor

This project addresses EthioMart’s challenge of consolidating decentralized e-commerce data from Telegram channels in Ethiopia, where vendors post product listings in Amharic and English. The goal is to create a centralized data hub by ingesting, preprocessing, and analyzing Telegram messages, extracting key entities (Product, Price, Location) using a fine-tuned Named Entity Recognition (NER) model, and developing a vendor lending scorecard for data-driven lending decisions.

---

## Project Objectives

- **Ingest and preprocess** Amharic and English text and image data from Telegram e-commerce channels.
- **Label data** in CoNLL format for NER training.
- **Fine-tune an NER model** to extract Product, Price, and Location entities.
- **Develop a vendor lending scorecard** (planned) based on posting frequency, engagement, and entity consistency.

---

## Tasks Completed

### Task 1: Data Ingestion and Preprocessing

- **Objective:** Collect and preprocess Telegram messages for analysis.
- **Scripts:**
  - `src/core/telegram_scraper.py`: Scrapes messages, images, and metadata using Telethon.
  - `src/utils/preprocess.py`: Normalizes text, tokenizes Amharic/English, detects language, and extracts emojis.
- **Output:**
  - **Raw data:** `data/raw/telegram_data.csv` (columns: Channel Title, Channel Username, Message ID, Message Text, Date, Media Path).
  - **Preprocessed data:** `data/processed/preprocessed_telegram_data.csv` (additional columns: Language, Emojis, Preprocessed Text).

### Task 2: CoNLL Labeling

- **Objective:** Label messages for NER training in CoNLL format.
- **Script:** `src/core/conll_format.py`
- Uses regex (e.g., for prices like `2300 ብር`) and a pre-trained NER model (`Davlan/afro-xlmr-mini`) for initial annotations, followed by manual correction.
- **Output:** `data/labeled/conll_labeled_data.conll` with 50 messages labeled using BIO tags (`B-PRODUCT`, `I-PRODUCT`, `B-PRICE`, `I-PRICE`, `B-LOC`, `I-LOC`, `O`).

### Task 3: NER Fine-Tuning

- **Objective:** Fine-tune an NER model to extract Product, Price, and Location entities.
- **Script:** `src/core/ner_finetune.py`
- Fine-tunes `Davlan/afro-xlmr-base` using Hugging Face’s Trainer API (5 epochs, batch size 8, learning rate 2e-5).
- Evaluates with `seqeval` (precision, recall, F1).
- **Output:**
  - **Model:** `data/models/ner_model/final/` (model weights, tokenizer).
  - **Results:** F1 score 0.0076 due to small dataset (40 training examples).

> **Limitations:** Low F1 score indicates need for more labeled data and hyperparameter tuning.

---

## Directory Structure

```text
Amharic-Ecommerce-Data-Extractor/
├── data/
│   ├── raw/
│   │   └── telegram_data.csv
│   ├── processed/
│   │   └── preprocessed_telegram_data.csv
│   ├── labeled/
│   │   └── conll_labeled_data.conll
│   ├── models/
│   │   └── ner_model/
│   │       └── final/
│   └── README.md
├── docs/
│   ├── interim_report.md
│   └── interim_submission.pdf
├── src/
│   ├── core/
│   │   ├── telegram_scraper.py
│   │   ├── conll_format.py
│   │   └── ner_finetune.py
│   └── utils/
│       └── preprocess.py
├── .gitignore
├── README.md
└── requirements.txt
```

---

## File Descriptions

- `data/raw/telegram_data.csv`: Raw Telegram messages scraped from five channels.
- `data/processed/preprocessed_telegram_data.csv`: Cleaned and tokenized messages.
- `data/labeled/conll_labeled_data.conll`: 50 messages with BIO tags for NER.
- `data/models/ner_model/final/`: Fine-tuned afro-xlmr-base model and tokenizer.
- `docs/interim_report.md`: Interim report detailing Tasks 1-3.
- `docs/interim_submission.pdf`: Submission document for project progress.
- `src/core/telegram_scraper.py`: Scrapes Telegram data using Telethon.
- `src/utils/preprocess.py`: Preprocesses text (tokenization, normalization).
- `src/core/conll_format.py`: Labels messages in CoNLL format.
- `src/core/ner_finetune.py`: Fine-tunes NER model and evaluates performance.
- `requirements.txt`: Lists dependencies (e.g., telethon, transformers, seqeval).
- `.gitignore`: Excludes data files, .venv, and model weights.

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone git@github.com:Abel5173/Amharic-Ecommerce-Data-Extractor.git
   cd Amharic-Ecommerce-Data-Extractor
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Telegram API credentials:**
   - Obtain `api_id` and `api_hash` from [my.telegram.org](https://my.telegram.org).
   - Store in environment variables or update `telegram_scraper.py`.

---

## Usage

### Scrape Telegram Data

```bash
python src/core/telegram_scraper.py
```
*Outputs:* `data/raw/telegram_data.csv`

### Preprocess Data

```bash
python src/utils/preprocess.py
```
*Outputs:* `data/processed/preprocessed_telegram_data.csv`

### Label Data in CoNLL Format

```bash
python src/core/conll_format.py
```
*Outputs:* `data/labeled/conll_labeled_data.conll`

### Fine-Tune NER Model

```bash
python src/core/ner_finetune.py
```
*Outputs:* Model in `data/models/ner_model/final/`, evaluation metrics

---

## Results

- **Task 1:** Successfully scraped ~500 messages from five channels, preprocessed with accurate Amharic tokenization and language detection.
- **Task 2:** Labeled 50 messages in CoNLL format with balanced entity distribution.
- **Task 3:** Fine-tuned afro-xlmr-base model, but achieved low F1 score (0.0076) due to small dataset (40 training examples).

---

## Limitations

- Small labeled dataset limits NER performance.
- Subword tokenization in afro-xlmr-base causes label misalignment for Amharic text.
- Vendor lending scorecard not yet implemented.

---

## Future Work

### Task 3 Improvements
- Augment CoNLL dataset to 200-500 examples using synthetic data (e.g., price variations).
- Fine-tune Davlan/bert-tiny-amharic for better Amharic handling.
- Apply class weights to address label imbalance.

### Task 4: Vendor Lending Scorecard
- Develop `vendor_scorecard.py` to compute scores based on posting frequency, average views, and entity consistency.
- Integrate SHAP/LIME for NER interpretability.

### Model Comparison
- Evaluate afro-xlmr-base, bert-tiny-amharic, and afro-xlmr-mini for performance and efficiency.

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository.
2. Create a branch (`git checkout -b feature/your-feature`).
3. Commit changes with conventional messages (e.g., `feat: add new feature`).
4. Push and open a pull request.

For questions, contact **Abel5173**.
