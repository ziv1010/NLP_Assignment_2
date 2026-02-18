"""
Step 6: Prepare BBC Hindi classification dataset.
Loads CSV, cleans text, creates train/val/test splits, builds vocabulary.
"""
import sys, os, re, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm


def clean_hindi_text(text: str) -> str:
    """Clean Hindi text for classification."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\u0900-\u097F\s0-9\u0964,?!.\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text: str) -> list:
    """Tokenize Hindi text."""
    text = re.sub(r'[\u0964,?!.\-]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 1]


def prepare_data():
    print("[Step 6] Preparing BBC Hindi classification dataset...")
    set_seed(RANDOM_SEED)

    # Load CSV
    df = pd.read_csv(BBC_CSV_PATH)
    print(f"  Raw dataset: {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")

    # Clean category names and pick valid classes
    df['Category'] = df['Category'].str.strip()
    available_categories = sorted(c for c in df['Category'].dropna().unique() if c)
    if INCLUDE_SOCIAL_CATEGORY:
        valid_categories = available_categories
    else:
        valid_categories = [c for c in available_categories if c != "\u0938\u094b\u0936\u0932"]

    df = df[df['Category'].isin(valid_categories)].copy()
    print(f"  After filtering valid categories: {len(df)} rows")
    print(f"  Using categories: {valid_categories}")
    print(f"  Category distribution:")
    for cat, count in df['Category'].value_counts().items():
        print(f"    {cat}: {count}")

    # Combine Headline + Content for richer input
    df['text'] = df['Headline'].fillna('') + ' ' + df['Content'].fillna('')
    df['text'] = df['text'].apply(clean_hindi_text)
    df['tokens'] = df['text'].apply(tokenize)
    df['token_count'] = df['tokens'].apply(len)

    # Filter very short documents
    df = df[df['token_count'] >= CLS_MIN_TOKEN_COUNT].copy()
    print(f"  After filtering short docs: {len(df)} rows")

    # Create label encoding
    label2idx = {label: idx for idx, label in enumerate(sorted(valid_categories))}
    idx2label = {idx: label for label, idx in label2idx.items()}
    df['label'] = df['Category'].map(label2idx)

    print(f"  Label mapping: {label2idx}")

    # Train/Val/Test split (70/15/15)
    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=RANDOM_SEED, stratify=df['label']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=RANDOM_SEED, stratify=temp_df['label']
    )

    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Build vocabulary from training data
    word_counter = Counter()
    for tokens in train_df['tokens']:
        word_counter.update(tokens)

    # Keep words with min frequency
    min_freq = CLS_VOCAB_MIN_FREQ
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, count in word_counter.most_common():
        if count >= min_freq:
            vocab[word] = len(vocab)

    print(f"  Vocabulary size: {len(vocab)}")

    # Convert tokens to indices
    def tokens_to_indices(tokens, max_len=LSTM_MAX_LEN):
        indices = [vocab.get(t, vocab['<UNK>']) for t in tokens[:max_len]]
        return indices

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df['indices'] = train_df['tokens'].apply(tokens_to_indices)
    val_df['indices'] = val_df['tokens'].apply(tokens_to_indices)
    test_df['indices'] = test_df['tokens'].apply(tokens_to_indices)

    # Save everything
    data = {
        'train': list(zip(train_df['indices'].tolist(), train_df['label'].tolist())),
        'val': list(zip(val_df['indices'].tolist(), val_df['label'].tolist())),
        'test': list(zip(test_df['indices'].tolist(), test_df['label'].tolist())),
        'vocab': vocab,
        'label2idx': label2idx,
        'idx2label': idx2label,
        'category_distribution': df['Category'].value_counts().to_dict(),
    }

    data_path = os.path.join(BBC_PROCESSED_DIR, "processed_data.pkl")
    with open(data_path, "wb") as f:
        pickle.dump(data, f)

    print(f"  Saved processed data to {data_path}")

    # Also save a summary CSV
    summary_path = os.path.join(BBC_PROCESSED_DIR, "dataset_summary.csv")
    summary_rows = [
        {'split': 'train', 'count': len(train_df)},
        {'split': 'val', 'count': len(val_df)},
        {'split': 'test', 'count': len(test_df)},
    ]
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_path, index=False)

    # Token length stats
    print(f"\n  Token length statistics:")
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        lengths = split_df['token_count']
        print(f"    {name}: mean={lengths.mean():.0f}, median={lengths.median():.0f}, "
              f"max={lengths.max()}, min={lengths.min()}")

    return data


if __name__ == "__main__":
    prepare_data()
