"""
Step 1: Download / prepare Hindi corpus for word vector training.
Tries multiple sources in order:
  1. AI4Bharat Sangraha (verified/hin) via HuggingFace
  2. Fallback: use BBC Hindi dataset as corpus (locally available)
"""
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from tqdm import tqdm
import pandas as pd


def extract_text(sample: dict) -> str:
    """Extract text payload from a sample with defensive fallbacks."""
    if not isinstance(sample, dict):
        return ""
    for key in ("text", "content", "document", "body"):
        value = sample.get(key)
        if isinstance(value, str):
            return value.strip()
    return ""


def download_from_sangraha_streaming(num_samples):
    """Download Sangraha using streaming API (preferred for large corpora)."""
    try:
        from datasets import load_dataset
        print("  Trying AI4Bharat Sangraha (streaming=True)...")
        dataset = load_dataset(
            "ai4bharat/sangraha", data_dir="verified/hin",
            split="train", streaming=True
        )
        lines = []
        for i, sample in enumerate(tqdm(dataset, total=num_samples, desc="Sangraha stream")):
            if i >= num_samples:
                break
            text = extract_text(sample)
            if text and len(text.strip()) > 20:
                lines.append(text.strip())
        if len(lines) >= 1000:
            return lines
        print(f"  Streaming returned too few lines: {len(lines)}")
        return None
    except Exception as e:
        print(f"  Streaming load failed: {e}")
        return None


def download_from_sangraha_standard(num_samples):
    """Download Sangraha with the non-streaming API from instructor note."""
    try:
        from datasets import load_dataset
        print("  Trying AI4Bharat Sangraha (standard load_dataset call)...")
        # Instructor-announced pattern:
        # dataset = load_dataset("ai4bharat/sangraha", data_dir="verified/hin")
        dataset = load_dataset("ai4bharat/sangraha", data_dir="verified/hin")
        # Some HF versions expose split as 'train'; keep defensive fallback.
        split_name = "train" if "train" in dataset else next(iter(dataset.keys()))
        split = dataset[split_name]

        lines = []
        for i in tqdm(range(min(num_samples, len(split))), desc="Sangraha standard"):
            text = extract_text(split[i])
            if text and len(text) > 20:
                lines.append(text)
        if len(lines) >= 1000:
            return lines
        print(f"  Standard load returned too few lines: {len(lines)}")
        return None
    except Exception as e:
        print(f"  Standard load failed: {e}")
        return None


def use_bbc_as_corpus(max_samples=None):
    """Use BBC Hindi articles as training corpus.
    This provides ~60MB of quality Hindi text from news articles."""
    print("  Using BBC Hindi articles as corpus...")
    df = pd.read_csv(BBC_CSV_PATH)
    lines = []
    
    for _, row in df.iterrows():
        # Use content (main body of each article)
        content = str(row.get('Content', ''))
        if content and len(content.strip()) > 20:
            # Split long articles into paragraph-sized chunks for better training
            paragraphs = content.strip().split('\n')
            for para in paragraphs:
                para = para.strip()
                if len(para) > 20:
                    lines.append(para)
        
        # Also use headlines as short sentences
        headline = str(row.get('Headline', ''))
        if headline and len(headline.strip()) > 10:
            lines.append(headline.strip())

    if max_samples is not None and max_samples > 0:
        lines = lines[:max_samples]

    print(f"  Extracted {len(lines)} text segments from BBC Hindi dataset")
    return lines


def deduplicate_preserve_order(lines):
    seen = set()
    out = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            out.append(line)
    return out


def download_corpus(num_samples=None):
    if num_samples is None:
        num_samples = get_num_samples()
    min_required = get_min_corpus_lines()

    print(f"[Step 1] Preparing Hindi corpus ({num_samples} target samples, min required: {min_required})...")

    if os.path.exists(RAW_CORPUS_PATH) and not FORCE_REBUILD_CORPUS:
        line_count = sum(1 for _ in open(RAW_CORPUS_PATH, encoding='utf-8'))
        if line_count >= min_required:
            print(f"  Corpus already exists with {line_count} lines (>= {min_required}). Skipping download.")
            return RAW_CORPUS_PATH
        print(f"  Existing corpus has only {line_count} lines (< {min_required}), rebuilding.")

    # Try Sangraha first (streaming preferred for large data)
    lines = download_from_sangraha_streaming(num_samples)
    source = "ai4bharat/sangraha (verified/hin, streaming)"

    if lines is None or len(lines) < 1000:
        lines = download_from_sangraha_standard(num_samples)
        source = "ai4bharat/sangraha (verified/hin, standard)"

    # Fallback: use BBC Hindi dataset
    if lines is None or len(lines) < 1000:
        lines = use_bbc_as_corpus(max_samples=num_samples)
        source = "BBC Hindi fallback"

    if not lines:
        print("  ERROR: Could not prepare corpus!")
        return None

    lines = deduplicate_preserve_order(lines)
    if len(lines) > num_samples:
        lines = lines[:num_samples]
    print(f"  Total lines collected after dedup/filter: {len(lines)}")

    with open(RAW_CORPUS_PATH, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    meta = {
        "source": source,
        "target_num_samples": num_samples,
        "actual_num_lines": len(lines),
    }
    with open(CORPUS_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    file_size_mb = os.path.getsize(RAW_CORPUS_PATH) / (1024 * 1024)
    print(f"  Saved to {RAW_CORPUS_PATH} ({file_size_mb:.1f} MB)")
    print(f"  Source: {source}")
    if len(lines) < min_required:
        print(f"  WARNING: Collected lines ({len(lines)}) are below recommended full-data threshold ({min_required}).")
        print("  If needed, increase CORPUS_NUM_SAMPLES_FULL in config.py.")
    return RAW_CORPUS_PATH


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=None)
    args = parser.parse_args()
    download_corpus(args.num_samples)
