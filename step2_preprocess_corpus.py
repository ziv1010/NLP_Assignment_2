"""
Step 2: Preprocess Hindi corpus - clean, normalize, tokenize.
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from tqdm import tqdm


def clean_hindi_text(text: str) -> str:
    """Clean a single line of Hindi text."""
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Keep Devanagari, digits, basic punctuation and spaces
    # Devanagari Unicode range: \u0900-\u097F
    text = re.sub(r'[^\u0900-\u097F\s0-9\u0964,?!.\-]', ' ', text)
    # Normalize multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_hindi(text: str) -> list:
    """Simple whitespace tokenization for Hindi (Devanagari is space-delimited)."""
    # Remove punctuation for word-level tokenization
    text = re.sub(r'[\u0964,?!.\-]', ' ', text)
    tokens = text.split()
    # Filter very short tokens
    tokens = [t for t in tokens if len(t) > 1]
    return tokens


def preprocess_corpus():
    print("[Step 2] Preprocessing Hindi corpus...")

    if not os.path.exists(RAW_CORPUS_PATH):
        print(f"  ERROR: Raw corpus not found at {RAW_CORPUS_PATH}")
        print("  Please run step1_download_corpus.py first.")
        return

    with open(RAW_CORPUS_PATH, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    print(f"  Read {len(raw_lines)} raw lines")

    processed = []
    seen = set()
    for line in tqdm(raw_lines, desc="Preprocessing"):
        cleaned = clean_hindi_text(line)
        tokens = tokenize_hindi(cleaned)
        if len(tokens) >= 5:  # skip very short sentences
            joined = " ".join(tokens)
            if joined not in seen:
                seen.add(joined)
                processed.append(joined)

    print(f"  {len(processed)} lines after preprocessing")

    with open(CLEAN_CORPUS_PATH, "w", encoding="utf-8") as f:
        for line in processed:
            f.write(line + "\n")

    file_size_mb = os.path.getsize(CLEAN_CORPUS_PATH) / (1024 * 1024)
    print(f"  Saved to {CLEAN_CORPUS_PATH} ({file_size_mb:.1f} MB)")

    # Print some samples
    print("\n  Sample preprocessed lines:")
    for i in range(min(3, len(processed))):
        print(f"    [{i}] {processed[i][:100]}...")


if __name__ == "__main__":
    preprocess_corpus()
