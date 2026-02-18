"""
Step 5: Compare custom-trained word vectors (FastText & Word2Vec) with
pretrained Common Crawl Hindi FastText vectors in both .vec and .bin format.

Produces:
  - Vocabulary overlap stats
  - Word similarity comparisons
  - Nearest neighbor comparisons
  - Word analogy tests
  - t-SNE visualizations
  - Comprehensive comparison report
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
from gensim.models import Word2Vec, FastText, KeyedVectors
from gensim.models.fasttext import load_facebook_vectors
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import json

# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 Font setup for Devanagari \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# Try to find a Devanagari-capable font
def setup_devanagari_font():
    """Try various approaches to render Devanagari in matplotlib."""
    candidates = [
        "Devanagari MT", "Devanagari Sangam MN", "Kohinoor Devanagari",
        "Noto Sans Devanagari", "Mangal", "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for c in candidates:
        if c in available:
            # Use fallback stack so Latin labels/legend remain readable.
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = ["DejaVu Sans", c, "Arial Unicode MS"]
            print(f"  Using font: {c}")
            return True
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial Unicode MS"]
    # Fallback - transliterate labels are handled separately.
    print("  WARNING: No Devanagari font found - using fallback fonts")
    return False


WORD_ASCII = {
    "राजा": "raja",
    "रानी": "rani",
    "भारत": "bharat",
    "देश": "desh",
    "पानी": "pani",
    "जल": "jal",
    "बड़ा": "bada",
    "छोटा": "chhota",
    "अच्छा": "accha",
    "बुरा": "bura",
    "लड़का": "ladka",
    "लड़की": "ladki",
    "सूरज": "suraj",
    "चांद": "chaand",
    "डॉक्टर": "doctor",
    "अस्पताल": "aspatal",
    "खाना": "khana",
    "पीना": "peena",
    "स्कूल": "school",
    "शिक्षा": "shiksha",
}


def pair_label_ascii(w1, w2):
    return f"{WORD_ASCII.get(w1, w1)}-{WORD_ASCII.get(w2, w2)}"


def safe_word_label(word: str, idx: int) -> str:
    """Always return an ASCII-friendly annotation label for plots."""
    if word in WORD_ASCII:
        return WORD_ASCII[word]
    if word.isascii():
        return word
    return f"w{idx + 1}"


def load_models():
    """Load all available models for comparison."""
    models = {}

    # Custom FastText
    if os.path.exists(FASTTEXT_MODEL_PATH):
        models['FastText (custom)'] = FastText.load(FASTTEXT_MODEL_PATH).wv
        print(f"  Loaded FastText: {len(models['FastText (custom)'])} words")
    else:
        print("  WARNING: FastText model not found")

    # Custom Word2Vec
    if os.path.exists(WORD2VEC_MODEL_PATH):
        models['Word2Vec (custom)'] = Word2Vec.load(WORD2VEC_MODEL_PATH).wv
        print(f"  Loaded Word2Vec: {len(models['Word2Vec (custom)'])} words")
    else:
        print("  WARNING: Word2Vec model not found")

    # Pretrained Common Crawl Hindi FastText (.vec text)
    if os.path.exists(PRETRAINED_VEC_PATH):
        print(f"  Loading Common Crawl Hindi pretrained vectors (limit={PRETRAINED_LOAD_LIMIT:,})...")
        models['CC Hindi (pretrained .vec)'] = KeyedVectors.load_word2vec_format(
            PRETRAINED_VEC_PATH, binary=False, limit=PRETRAINED_LOAD_LIMIT
        )
        print(f"  Loaded CC Hindi pretrained (.vec): {len(models['CC Hindi (pretrained .vec)'])} words")
    else:
        print("  WARNING: Pretrained .vec not found - run step4_download_pretrained.py first")

    # Pretrained Common Crawl Hindi FastText (.bin binary with subword vectors)
    if os.path.exists(PRETRAINED_BIN_PATH):
        print("  Loading Common Crawl Hindi pretrained .bin vectors (this can take a while)...")
        try:
            models['CC Hindi (pretrained .bin)'] = load_facebook_vectors(PRETRAINED_BIN_PATH)
            print(f"  Loaded CC Hindi pretrained (.bin): {len(models['CC Hindi (pretrained .bin)'])} words")
        except Exception as e:
            print(f"  WARNING: Failed to load .bin vectors: {e}")
    else:
        print("  WARNING: Pretrained .bin not found")

    return models


def vocab_overlap(models):
    """Compare vocabulary overlap between models."""
    print("\n" + "="*60)
    print("VOCABULARY OVERLAP ANALYSIS")
    print("="*60)

    names = list(models.keys())
    vocabs = {name: set(models[name].key_to_index.keys()) for name in names}

    results = {}
    for name in names:
        results[name] = len(vocabs[name])
        print(f"  {name}: {len(vocabs[name]):,} words")

    # Pairwise overlap
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            overlap = vocabs[names[i]] & vocabs[names[j]]
            print(f"  {names[i]} \u2229 {names[j]}: {len(overlap):,} words")
            results[f"{names[i]} \u2229 {names[j]}"] = len(overlap)

    return results


def nearest_neighbors(models, test_words=None):
    """Compare nearest neighbors across models."""
    if test_words is None:
        test_words = ["\u092d\u093e\u0930\u0924", "\u0938\u0930\u0915\u093e\u0930", "\u092a\u093e\u0928\u0940", "\u0936\u093f\u0915\u094d\u0937\u093e", "\u0915\u094d\u0930\u093f\u0915\u0947\u091f",
                       "\u092e\u0939\u093f\u0932\u093e", "\u0905\u0930\u094d\u0925\u0935\u094d\u092f\u0935\u0938\u094d\u0925\u093e", "\u0924\u0915\u0928\u0940\u0915", "\u0938\u0902\u0917\u0940\u0924", "\u0939\u093f\u0902\u0926\u0940"]

    print("\n" + "="*60)
    print("NEAREST NEIGHBORS COMPARISON (Top 5)")
    print("="*60)

    results = {}
    for word in test_words:
        results[word] = {}
        print(f"\n  Word: '{word}'")
        for name, wv in models.items():
            try:
                neighbors = wv.most_similar(word, topn=5)
                results[word][name] = [(w, round(s, 3)) for w, s in neighbors]
                print(f"    {name}: {[w for w, s in neighbors]}")
            except KeyError:
                results[word][name] = "NOT IN VOCAB"
                print(f"    {name}: NOT IN VOCAB")

    return results


def word_similarity_comparison(models, word_pairs=None):
    """Compare cosine similarity for word pairs across models."""
    if word_pairs is None:
        word_pairs = [
            ("\u0930\u093e\u091c\u093e", "\u0930\u093e\u0928\u0940"),
            ("\u092d\u093e\u0930\u0924", "\u0926\u0947\u0936"),
            ("\u092a\u093e\u0928\u0940", "\u091c\u0932"),
            ("\u092c\u0921\u093c\u093e", "\u091b\u094b\u091f\u093e"),
            ("\u0905\u091a\u094d\u091b\u093e", "\u092c\u0941\u0930\u093e"),
            ("\u0932\u0921\u093c\u0915\u093e", "\u0932\u0921\u093c\u0915\u0940"),
            ("\u0938\u0942\u0930\u091c", "\u091a\u093e\u0902\u0926"),
            ("\u0921\u0949\u0915\u094d\u091f\u0930", "\u0905\u0938\u094d\u092a\u0924\u093e\u0932"),
            ("\u0916\u093e\u0928\u093e", "\u092a\u0940\u0928\u093e"),
            ("\u0938\u094d\u0915\u0942\u0932", "\u0936\u093f\u0915\u094d\u0937\u093e"),
        ]

    print("\n" + "="*60)
    print("WORD SIMILARITY COMPARISON")
    print("="*60)

    results = {}
    for w1, w2 in word_pairs:
        results[f"{w1}-{w2}"] = {}
        sims = []
        for name, wv in models.items():
            try:
                sim = wv.similarity(w1, w2)
                results[f"{w1}-{w2}"][name] = round(float(sim), 4)
                sims.append((name, sim))
            except KeyError:
                results[f"{w1}-{w2}"][name] = "N/A"
                sims.append((name, None))

        sim_str = "  |  ".join(
            f"{n}: {s:.3f}" if s is not None else f"{n}: N/A"
            for n, s in sims
        )
        print(f"  ({w1}, {w2}):  {sim_str}")

    # Plot similarity comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    pair_labels = [pair_label_ascii(w1, w2) for w1, w2 in word_pairs]
    x = np.arange(len(pair_labels))
    n_models = max(len(models), 1)
    width = min(0.25, 0.8 / n_models)

    for i, (name, wv) in enumerate(models.items()):
        sims = []
        for w1, w2 in word_pairs:
            try:
                sims.append(float(wv.similarity(w1, w2)))
            except KeyError:
                sims.append(0)
        display_name = {
            "FastText (custom)": "FastText-custom",
            "Word2Vec (custom)": "Word2Vec-custom",
            "CC Hindi (pretrained .vec)": "CC-pretrained-vec",
            "CC Hindi (pretrained .bin)": "CC-pretrained-bin",
        }.get(name, name)
        ax.bar(x + i * width, sims, width, label=display_name, alpha=0.85)

    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_title('Word Pair Similarity Across Word-Vector Models', fontsize=13)
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(pair_labels, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(WV_OUTPUT_DIR, "word_similarity_comparison.png"), dpi=150)
    plt.close()
    print(f"  Saved word_similarity_comparison.png")

    return results


def analogy_test(models):
    """Test word analogies (e.g., \u0930\u093e\u091c\u093e - \u092a\u0941\u0930\u0941\u0937 + \u092e\u0939\u093f\u0932\u093e \u2248 \u0930\u093e\u0928\u0940)."""
    analogies = [
        ("\u0930\u093e\u091c\u093e", "\u092a\u0941\u0930\u0941\u0937", "\u092e\u0939\u093f\u0932\u093e", "\u0930\u093e\u0928\u0940"),     # king - man + woman = queen
        ("\u092d\u093e\u0930\u0924", "\u0926\u093f\u0932\u094d\u0932\u0940", "\u091c\u093e\u092a\u093e\u0928", "\u091f\u094b\u0915\u094d\u092f\u094b"),   # India - Delhi + Japan = Tokyo
        ("\u0932\u0921\u093c\u0915\u093e", "\u0932\u0921\u093c\u0915\u0940", "\u092a\u093f\u0924\u093e", "\u092e\u093e\u0924\u093e"),       # boy - girl + father = mother
        ("\u092c\u0921\u093c\u093e", "\u092c\u0921\u093c\u0947", "\u091b\u094b\u091f\u093e", "\u091b\u094b\u091f\u0947"),         # big(sg) - big(pl) + small(sg) = small(pl)
        ("\u0916\u093e\u0928\u093e", "\u0916\u093e\u0924\u093e", "\u092a\u0940\u0928\u093e", "\u092a\u0940\u0924\u093e"),        # eat - eats + drink = drinks
    ]

    print("\n" + "="*60)
    print("WORD ANALOGY TEST (a - b + c ~= d)")
    print("="*60)

    results = {}
    for a, b, c, expected in analogies:
        label = f"{a} - {b} + {c} ~= {expected}"
        results[label] = {}
        print(f"\n  {label}")
        for name, wv in models.items():
            try:
                result = wv.most_similar(positive=[a, c], negative=[b], topn=5)
                top_words = [w for w, s in result]
                hit = "Y" if expected in top_words else "N"
                results[label][name] = {"top5": top_words, "hit": expected in top_words}
                print(f"    {name}: {top_words}  {hit}")
            except KeyError:
                results[label][name] = "VOCAB ERROR"
                print(f"    {name}: VOCAB ERROR")

    return results


def tsne_visualization(models, num_words=100):
    """Create t-SNE plots for each model."""
    print("\n" + "="*60)
    print("t-SNE VISUALIZATION")
    print("="*60)

    # Get common words across all models
    common_vocab = None
    for name, wv in models.items():
        words = set(list(wv.key_to_index.keys())[:10000])
        if common_vocab is None:
            common_vocab = words
        else:
            common_vocab = common_vocab & words

    common_words = list(common_vocab)[:num_words]
    print(f"  Using {len(common_words)} common words for t-SNE")

    # Plot side by side
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6))
    if n_models == 1:
        axes = [axes]

    for ax, (name, wv) in zip(axes, models.items()):
        vectors = np.array([wv[w] for w in common_words])
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        coords = tsne.fit_transform(vectors)

        ax.scatter(coords[:, 0], coords[:, 1], s=15, alpha=0.7, c='steelblue')
        # Label a subset
        for i in range(0, min(20, len(common_words))):
            ax.annotate(safe_word_label(common_words[i], i), (coords[i, 0], coords[i, 1]),
                       fontsize=7, alpha=0.8)
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

    plt.suptitle("t-SNE Visualization of Hindi Word Vectors", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(WV_OUTPUT_DIR, "tsne_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved tsne_comparison.png")


def generate_report(vocab_results, similarity_results, neighbor_results, analogy_results):
    """Save a JSON report of all comparisons."""
    report = {
        "vocabulary_overlap": vocab_results,
        "word_similarity": similarity_results,
        "nearest_neighbors": neighbor_results,
        "analogies": analogy_results,
    }

    report_path = os.path.join(WV_OUTPUT_DIR, "comparison_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  Full report saved to {report_path}")


def compare_vectors():
    print("[Step 5] Comparing word vectors...")
    has_devanagari = setup_devanagari_font()

    models = load_models()
    if len(models) < 2:
        print("  Need at least 2 models for comparison. Skipping.")
        return

    vocab_results = vocab_overlap(models)
    neighbor_results = nearest_neighbors(models)
    similarity_results = word_similarity_comparison(models)
    analogy_results = analogy_test(models)
    tsne_visualization(models)
    generate_report(vocab_results, similarity_results, neighbor_results, analogy_results)

    print("\n" + "="*60)
    print("Word vector comparison complete!")
    print(f"Results saved in: {WV_OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    compare_vectors()
