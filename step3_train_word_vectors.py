"""
Step 3: Train Word2Vec AND FastText word vectors on preprocessed Hindi corpus.
Both models are trained for comparative analysis.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from gensim.models import Word2Vec, FastText
from tqdm import tqdm
import time


class CorpusIterator:
    """Iterate over preprocessed corpus file, yielding tokenized sentences."""
    def __init__(self, filepath, max_lines=None, total_lines=None):
        self.filepath = filepath
        self.max_lines = max_lines
        self.total_lines = total_lines
        self.stride = 1
        if self.max_lines and self.total_lines and self.total_lines > self.max_lines:
            # Spread sampling across the corpus instead of only taking the first chunk.
            self.stride = max(1, self.total_lines // self.max_lines)

    def __iter__(self):
        yielded = 0
        with open(self.filepath, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if self.stride > 1 and (idx % self.stride != 0):
                    continue
                tokens = line.strip().split()
                if tokens:
                    yield tokens
                    yielded += 1
                    if self.max_lines and yielded >= self.max_lines:
                        break


def train_fasttext(corpus_iter):
    print("\n[Step 3a] Training FastText model...")
    start = time.time()

    model = FastText(
        sentences=corpus_iter,
        vector_size=WV_DIM,
        window=WV_WINDOW,
        min_count=WV_MIN_COUNT,
        sg=WV_SG,
        epochs=WV_EPOCHS,
        workers=WV_WORKERS,
    )

    elapsed = time.time() - start
    print(f"  FastText training done in {elapsed:.1f}s")
    print(f"  Vocabulary size: {len(model.wv)}")

    model.save(FASTTEXT_MODEL_PATH)
    print(f"  Saved FastText model to {FASTTEXT_MODEL_PATH}")
    return model


def train_word2vec(corpus_iter):
    print("\n[Step 3b] Training Word2Vec model...")
    start = time.time()

    model = Word2Vec(
        sentences=corpus_iter,
        vector_size=WV_DIM,
        window=WV_WINDOW,
        min_count=WV_MIN_COUNT,
        sg=WV_SG,
        epochs=WV_EPOCHS,
        workers=WV_WORKERS,
    )

    elapsed = time.time() - start
    print(f"  Word2Vec training done in {elapsed:.1f}s")
    print(f"  Vocabulary size: {len(model.wv)}")

    model.save(WORD2VEC_MODEL_PATH)
    print(f"  Saved Word2Vec model to {WORD2VEC_MODEL_PATH}")
    return model


def train_all():
    if not os.path.exists(CLEAN_CORPUS_PATH):
        print(f"  ERROR: Clean corpus not found at {CLEAN_CORPUS_PATH}")
        print("  Please run step2_preprocess_corpus.py first.")
        return

    set_seed(RANDOM_SEED)
    line_count = sum(1 for _ in open(CLEAN_CORPUS_PATH, "r", encoding="utf-8"))
    max_train_lines = get_wv_max_train_lines()
    effective_lines = min(line_count, max_train_lines)
    print(f"[Step 3] Available preprocessed lines: {line_count:,}")
    if line_count > max_train_lines:
        print(f"[Step 3] Training on filtered large subset: {effective_lines:,} lines "
              f"(system-friendly setting from config.py)")
    else:
        print(f"[Step 3] Training on full preprocessed corpus: {effective_lines:,} lines")

    ft_model = train_fasttext(CorpusIterator(CLEAN_CORPUS_PATH, max_train_lines, line_count))
    w2v_model = train_word2vec(CorpusIterator(CLEAN_CORPUS_PATH, max_train_lines, line_count))

    # Quick sanity check
    test_words = ["\u092d\u093e\u0930\u0924", "\u0939\u093f\u0902\u0926\u0940", "\u0938\u0930\u0915\u093e\u0930", "\u0926\u0947\u0936", "\u0932\u094b\u0917"]
    print("\n  Quick sanity check - similar words:")
    for word in test_words:
        try:
            ft_sim = ft_model.wv.most_similar(word, topn=3)
            w2v_sim = w2v_model.wv.most_similar(word, topn=3)
            print(f"\n  '{word}':")
            print(f"    FastText: {[w for w, s in ft_sim]}")
            print(f"    Word2Vec: {[w for w, s in w2v_sim]}")
        except KeyError:
            print(f"    '{word}' not in vocabulary")

    return ft_model, w2v_model


if __name__ == "__main__":
    train_all()
