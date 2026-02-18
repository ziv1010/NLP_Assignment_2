"""
Central configuration for the Hindi NLP pipeline.
Toggle SUBSET_MODE to True for quick testing, False for full data.
"""
import os
import random
import numpy as np
import torch

# ──────────────────── Directories ────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CORPUS_DIR = os.path.join(DATA_DIR, "corpus")
BBC_DIR = os.path.join(DATA_DIR, "bbc_hindi")
BBC_PROCESSED_DIR = os.path.join(BBC_DIR, "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PRETRAINED_DIR = os.path.join(MODEL_DIR, "pretrained")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
WV_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "word_vectors")
CLS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "classification")

for d in [CORPUS_DIR, BBC_DIR, BBC_PROCESSED_DIR, MODEL_DIR,
          PRETRAINED_DIR, WV_OUTPUT_DIR, CLS_OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# ──────────────────── Subset / Full Mode ─────────────
SUBSET_MODE = False  # Set to False for full assignment run
FORCE_REBUILD_CORPUS = False

# Corpus download
CORPUS_NUM_SAMPLES_FULL = 200_000   # full mode — 200K sentences
CORPUS_NUM_SAMPLES_SUBSET = 50_000  # subset mode
MIN_CORPUS_LINES_FULL = 150_000
MIN_CORPUS_LINES_SUBSET = 30_000

# ──────────────────── Corpus Paths ───────────────────
RAW_CORPUS_PATH = os.path.join(CORPUS_DIR, "hindi_corpus.txt")
CLEAN_CORPUS_PATH = os.path.join(CORPUS_DIR, "hindi_corpus_clean.txt")
CORPUS_META_PATH = os.path.join(CORPUS_DIR, "corpus_meta.json")

# ──────────────────── BBC Dataset ────────────────────
BBC_CSV_PATH = os.path.join(BASE_DIR, "bbc_hindi_articles_with_categories_cleaned.csv")

# ──────────────────── Word Vector Hyperparams ────────
WV_DIM = 300
WV_WINDOW = 5
WV_MIN_COUNT = 5
WV_EPOCHS = 5
WV_SG = 1          # 1 = skip-gram, 0 = CBOW
WV_WORKERS = 4     # M1 has ~8 performance cores
WV_MAX_TRAIN_LINES_FULL = 400_000
WV_MAX_TRAIN_LINES_SUBSET = 80_000

# ──────────────────── LSTM Hyperparams ───────────────
LSTM_MAX_LEN = 256        # max tokens per document
LSTM_HIDDEN_DIM = 128
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3
LSTM_BIDIRECTIONAL = True
LSTM_BATCH_SIZE = 64
LSTM_LR = 1e-3
LSTM_EPOCHS_FULL = 15
LSTM_EPOCHS_SUBSET = 5
LSTM_PATIENCE = 4         # early stopping patience

# ──────────────────── Classification Data ────────────
INCLUDE_SOCIAL_CATEGORY = True   # include "सोशल" if present in dataset
CLS_MIN_TOKEN_COUNT = 10
CLS_VOCAB_MIN_FREQ = 2

# ──────────────────── Pretrained Vectors (Common Crawl Hindi FastText) ──
# Source: https://fasttext.cc/docs/en/crawl-vectors.html
# Binary format (supports OOV via subword info):
PRETRAINED_BIN_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.hi.300.bin.gz"
PRETRAINED_BIN_GZ = os.path.join(PRETRAINED_DIR, "cc.hi.300.bin.gz")
PRETRAINED_BIN_PATH = os.path.join(PRETRAINED_DIR, "cc.hi.300.bin")
# Text format (lighter, word vectors only):
PRETRAINED_VEC_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.hi.300.vec.gz"
PRETRAINED_VEC_GZ = os.path.join(PRETRAINED_DIR, "cc.hi.300.vec.gz")
PRETRAINED_VEC_PATH = os.path.join(PRETRAINED_DIR, "cc.hi.300.vec")
PRETRAINED_LOAD_LIMIT = 400_000   # load top-N vectors for memory/performance tradeoff

# ──────────────────── Model Paths ────────────────────
FASTTEXT_MODEL_PATH = os.path.join(MODEL_DIR, "hindi_fasttext.model")
WORD2VEC_MODEL_PATH = os.path.join(MODEL_DIR, "hindi_word2vec.model")
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_classifier.pt")

# ──────────────────── Reproducibility / Device ───────
RANDOM_SEED = 42

def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def get_num_samples():
    return CORPUS_NUM_SAMPLES_SUBSET if SUBSET_MODE else CORPUS_NUM_SAMPLES_FULL

def get_min_corpus_lines():
    return MIN_CORPUS_LINES_SUBSET if SUBSET_MODE else MIN_CORPUS_LINES_FULL

def get_lstm_epochs():
    return LSTM_EPOCHS_SUBSET if SUBSET_MODE else LSTM_EPOCHS_FULL

def get_wv_max_train_lines():
    return WV_MAX_TRAIN_LINES_SUBSET if SUBSET_MODE else WV_MAX_TRAIN_LINES_FULL

print(f"[config] SUBSET_MODE={SUBSET_MODE}  DEVICE={DEVICE}")
