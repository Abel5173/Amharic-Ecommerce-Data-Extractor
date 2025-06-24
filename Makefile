# Add your Makefile commands here

.PHONY: help setup scrape preprocess label train clean lint

help:
	@echo "Available targets:"
	@echo "  setup      - Create virtual environment and install dependencies"
	@echo "  scrape     - Scrape Telegram data (raw data)"
	@echo "  preprocess - Preprocess scraped data"
	@echo "  label      - Label data in CoNLL format for NER"
	@echo "  train      - Fine-tune NER model"
	@echo "  clean      - Remove generated data files"
	@echo "  lint       - Run flake8 linter on src/ and scripts/"

setup:
	python -m venv .venv && \
	source .venv/bin/activate && \
	pip install -r requirements.txt

scrape:
	python src/core/telegram_scraper.py

preprocess:
	python src/utils/preprocess.py

label:
	python src/core/conll_format.py

train:
	python src/core/ner_finetune.py

clean:
	rm -f data/raw/telegram_data.csv data/processed/preprocessed_telegram_data.csv data/labeled/conll_labeled_data.conll

lint:
	flake8 src/ scripts/
