"""
Step 10: Generate additional diagnostic visualizations for both assignment parts.

Outputs are saved separately in:
  outputs/additional_visualizations/
"""
import os
import json
import pickle
import re
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from config import *
from step7_lstm_model import LSTMClassifier, create_data_loaders


ADD_OUT_DIR = os.path.join(OUTPUT_DIR, "additional_visualizations")
ADD_WV_DIR = os.path.join(ADD_OUT_DIR, "word_vectors")
ADD_CLS_DIR = os.path.join(ADD_OUT_DIR, "classification")


def ensure_dirs():
    for d in [ADD_OUT_DIR, ADD_WV_DIR, ADD_CLS_DIR]:
        os.makedirs(d, exist_ok=True)


def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_plot_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial Unicode MS"]


PAIR_ASCII = {
    "राजा-रानी": "raja-rani",
    "भारत-देश": "bharat-desh",
    "पानी-जल": "pani-jal",
    "बड़ा-छोटा": "bada-chhota",
    "अच्छा-बुरा": "accha-bura",
    "लड़का-लड़की": "ladka-ladki",
    "सूरज-चांद": "suraj-chaand",
    "डॉक्टर-अस्पताल": "doctor-aspatal",
    "खाना-पीना": "khana-peena",
    "स्कूल-शिक्षा": "school-shiksha",
}

CLASS_ASCII = {
    "खेल": "Khel",
    "भारत": "Bharat",
    "मनोरंजन": "Manoranjan",
    "विज्ञान": "Vigyan",
    "विदेश": "Videsh",
    "सोशल": "Social",
}


def class_display(name: str) -> str:
    return CLASS_ASCII.get(str(name), str(name))


def short_model_name(name: str) -> str:
    return {
        "FastText (custom)": "FastText-custom",
        "Word2Vec (custom)": "Word2Vec-custom",
        "CC Hindi (pretrained)": "CC-pretrained",
        "CC Hindi (pretrained .vec)": "CC-pretrained-vec",
        "CC Hindi (pretrained .bin)": "CC-pretrained-bin",
    }.get(name, name)


# ---------------------------
# Word-vector visualizations
# ---------------------------
def plot_wv_vocab_and_overlap(wv_report):
    vocab = wv_report["vocabulary_overlap"]

    base_models = [k for k in vocab.keys() if "∩" not in k]
    sizes = [vocab[k] for k in base_models]
    df = pd.DataFrame(
        {"Model": [short_model_name(k) for k in base_models], "VocabSize": sizes}
    ).sort_values("VocabSize", ascending=False)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x="Model", y="VocabSize", hue="Model", palette="Set2", legend=False)
    plt.xticks(rotation=20, ha="right")
    plt.title("Vocabulary Size by Word-Vector Model")
    plt.ylabel("Vocabulary Size")
    save_fig(os.path.join(ADD_WV_DIR, "wv_vocab_sizes.png"))

    # Pairwise overlap heatmap.
    short_models = [short_model_name(k) for k in base_models]
    mat = pd.DataFrame(index=short_models, columns=short_models, dtype=float)
    for m_orig, m_short in zip(base_models, short_models):
        mat.loc[m_short, m_short] = vocab[m_orig]
    for i, m1 in enumerate(base_models):
        for j, m2 in enumerate(base_models):
            if i < j:
                k1 = f"{m1} ∩ {m2}"
                k2 = f"{m2} ∩ {m1}"
                inter = vocab.get(k1, vocab.get(k2, 0))
                mat.loc[short_model_name(m1), short_model_name(m2)] = inter
                mat.loc[short_model_name(m2), short_model_name(m1)] = inter

    plt.figure(figsize=(7, 6))
    sns.heatmap(mat.astype(float), annot=True, fmt=".0f", cmap="Blues")
    plt.title("Pairwise Vocabulary Overlap")
    save_fig(os.path.join(ADD_WV_DIR, "wv_overlap_heatmap.png"))


def plot_wv_similarity_and_delta(wv_report):
    sim = wv_report["word_similarity"]
    sim_df = pd.DataFrame(sim).T
    sim_df.index = [PAIR_ASCII.get(x, x) for x in sim_df.index]
    sim_df.rename(columns=lambda c: short_model_name(c), inplace=True)
    sim_num = sim_df.apply(pd.to_numeric, errors="coerce")

    # Heatmap for raw similarities.
    plt.figure(figsize=(12, 6))
    sns.heatmap(sim_num, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title("Word-Pair Similarity Heatmap")
    plt.xlabel("Model")
    plt.ylabel("Word Pair")
    save_fig(os.path.join(ADD_WV_DIR, "wv_similarity_heatmap.png"))

    # Delta vs CC pretrained.
    cc_col = None
    for candidate in ["CC-pretrained-bin", "CC-pretrained-vec", "CC-pretrained"]:
        if candidate in sim_num.columns:
            cc_col = candidate
            break
    if cc_col is None:
        print("  Skipping delta-vs-CC plot (no CC model column found).")
        return

    non_cc_cols = [c for c in sim_num.columns if c != cc_col]
    delta_df = pd.DataFrame(index=sim_df.index)
    for c in non_cc_cols:
        delta_df[f"{c} - {cc_col}"] = sim_num[c] - sim_num[cc_col]

    plt.figure(figsize=(12, 6))
    delta_df.plot(kind="bar", ax=plt.gca())
    plt.axhline(0, color="black", linewidth=1)
    plt.title(f"Similarity Delta vs {cc_col} (positive means higher pair similarity)")
    plt.ylabel("Delta Cosine Similarity")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Model Delta")
    save_fig(os.path.join(ADD_WV_DIR, "wv_similarity_delta_vs_cc.png"))


def plot_wv_neighbor_overlap(wv_report):
    nn = wv_report["nearest_neighbors"]
    # model names from first entry
    first_word = next(iter(nn.keys()))
    models = list(nn[first_word].keys())

    # Jaccard overlap matrix averaged across query words.
    mat = pd.DataFrame(0.0, index=models, columns=models)
    counts = pd.DataFrame(0.0, index=models, columns=models)

    for query, per_model in nn.items():
        neigh_sets = {}
        for m, lst in per_model.items():
            if isinstance(lst, list):
                neigh_sets[m] = set([x[0] for x in lst])
            else:
                neigh_sets[m] = set()

        for m1 in models:
            for m2 in models:
                s1, s2 = neigh_sets[m1], neigh_sets[m2]
                if len(s1 | s2) == 0:
                    continue
                jac = len(s1 & s2) / len(s1 | s2)
                mat.loc[m1, m2] += jac
                counts.loc[m1, m2] += 1

    avg = mat / counts.replace(0, np.nan)
    avg = avg.fillna(0)
    plt.figure(figsize=(7, 6))
    sns.heatmap(avg, annot=True, fmt=".2f", cmap="Oranges", vmin=0, vmax=1)
    plt.title("Average Neighbor-Set Jaccard Overlap (Top-5)")
    save_fig(os.path.join(ADD_WV_DIR, "wv_neighbor_overlap_heatmap.png"))


def plot_wv_analogy_hits(wv_report):
    analogies = wv_report["analogies"]
    hit_counter = Counter()
    total = len(analogies)

    for _, res in analogies.items():
        for model, val in res.items():
            if isinstance(val, dict) and val.get("hit", False):
                hit_counter[model] += 1

    models = sorted(hit_counter.keys() | set(next(iter(analogies.values())).keys()))
    short_models = [short_model_name(m) for m in models]
    df = pd.DataFrame({
        "Model": short_models,
        "Hits": [hit_counter.get(m, 0) for m in models],
    })
    df["HitRate"] = df["Hits"] / max(total, 1)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="Model", y="HitRate", hue="Model", palette="viridis", legend=False)
    plt.ylim(0, 1)
    plt.title(f"Analogy Hit Rate (Top-5), {total} Questions")
    plt.ylabel("Hit Rate")
    plt.xticks(rotation=20, ha="right")
    save_fig(os.path.join(ADD_WV_DIR, "wv_analogy_hit_rate.png"))


# ----------------------------------
# Classification visualizations
# ----------------------------------
def load_processed_data():
    with open(os.path.join(BBC_PROCESSED_DIR, "processed_data.pkl"), "rb") as f:
        return pickle.load(f)


def plot_cls_split_distribution(data):
    idx2label = data["idx2label"]
    rows = []
    for split in ["train", "val", "test"]:
        labels = [lbl for _, lbl in data[split]]
        c = Counter(labels)
        for idx, count in sorted(c.items()):
            rows.append({"Split": split, "Class": idx2label[idx], "Count": count})
    df = pd.DataFrame(rows)
    df["ClassDisplay"] = df["Class"].map(class_display)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="ClassDisplay", y="Count", hue="Split", palette="Set2")
    plt.title("Class Distribution by Split")
    plt.xticks(rotation=15)
    save_fig(os.path.join(ADD_CLS_DIR, "cls_split_class_distribution.png"))


def clean_hindi_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^\u0900-\u097F\s0-9।,?!.\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list:
    text = re.sub(r"[।,?!.\-]", " ", text)
    return [t for t in text.split() if len(t) > 1]


def build_length_frame():
    df = pd.read_csv(BBC_CSV_PATH)
    df["Category"] = df["Category"].astype(str).str.strip()

    # Mirror step6 behavior (all categories currently enabled).
    valid_categories = sorted(c for c in df["Category"].dropna().unique() if c)
    df = df[df["Category"].isin(valid_categories)].copy()

    label2idx = {label: idx for idx, label in enumerate(sorted(valid_categories))}
    idx2label = {idx: label for label, idx in label2idx.items()}

    df["text"] = df["Headline"].fillna("") + " " + df["Content"].fillna("")
    df["text"] = df["text"].apply(clean_hindi_text)
    df["tokens"] = df["text"].apply(tokenize)
    df["token_count"] = df["tokens"].apply(len)
    df = df[df["token_count"] >= CLS_MIN_TOKEN_COUNT].copy()
    df["label"] = df["Category"].map(label2idx)

    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=RANDOM_SEED, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=RANDOM_SEED, stratify=temp_df["label"]
    )

    frames = []
    for split_name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        part = sdf[["Category", "token_count"]].copy()
        part["split"] = split_name
        part["truncated"] = part["token_count"] > LSTM_MAX_LEN
        frames.append(part)
    out = pd.concat(frames, ignore_index=True)
    out.rename(columns={"Category": "class"}, inplace=True)
    return out, idx2label


def plot_cls_length_and_truncation():
    length_df, _ = build_length_frame()
    length_df["class_display"] = length_df["class"].map(class_display)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=length_df, x="split", y="token_count", palette="pastel")
    plt.yscale("log")
    plt.title("Token Length Distribution by Split (log scale)")
    plt.ylabel("Token Count")
    save_fig(os.path.join(ADD_CLS_DIR, "cls_token_length_by_split.png"))

    trunc = (
        length_df.groupby("class", as_index=False)["truncated"]
        .mean()
        .rename(columns={"truncated": "truncation_rate"})
        .sort_values("truncation_rate", ascending=False)
    )
    trunc["class_display"] = trunc["class"].map(class_display)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=trunc, x="class_display", y="truncation_rate", palette="rocket")
    plt.ylim(0, 1)
    plt.title(f"Truncation Rate by Class (>{LSTM_MAX_LEN} tokens)")
    plt.ylabel("Truncation Rate")
    plt.xticks(rotation=20, ha="right")
    save_fig(os.path.join(ADD_CLS_DIR, "cls_truncation_rate_by_class.png"))


def plot_cls_metrics(eval_report):
    models = list(eval_report.keys())
    metric_cols = ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]

    metric_df = pd.DataFrame(
        [{**{"model": m}, **{k: eval_report[m][k] for k in metric_cols}} for m in models]
    ).set_index("model")

    plt.figure(figsize=(8, 4))
    sns.heatmap(metric_df, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title("Model Metric Heatmap")
    save_fig(os.path.join(ADD_CLS_DIR, "cls_model_metric_heatmap.png"))

    # Per-class F1 bar chart.
    class_rows = []
    for m in models:
        per = eval_report[m]["per_class_report"]
        for cls, vals in per.items():
            if cls in ["accuracy", "macro avg", "weighted avg"]:
                continue
            class_rows.append({"model": m, "class": class_display(cls), "f1": vals["f1-score"]})
    class_df = pd.DataFrame(class_rows)

    plt.figure(figsize=(11, 6))
    sns.barplot(data=class_df, x="class", y="f1", hue="model", palette="Set2")
    plt.ylim(0, 1)
    plt.title("Per-Class F1 by Model")
    plt.xticks(rotation=15)
    save_fig(os.path.join(ADD_CLS_DIR, "cls_per_class_f1.png"))


def get_predictions_with_confidence(model, loader, device):
    model.eval()
    y_true, y_pred, conf = [], [], []
    with torch.no_grad():
        for batch_x, batch_y, lengths in loader:
            batch_x = batch_x.to(device)
            out = model(batch_x, lengths)
            probs = torch.softmax(out, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            c = probs.max(axis=1)
            y_pred.extend(preds.tolist())
            y_true.extend(batch_y.numpy().tolist())
            conf.extend(c.tolist())
    return np.array(y_true), np.array(y_pred), np.array(conf)


def reliability_curve(correct, conf, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(conf, bins) - 1
    xs, ys, ws = [], [], []
    for b in range(n_bins):
        m = bin_ids == b
        if not np.any(m):
            continue
        xs.append(conf[m].mean())
        ys.append(correct[m].mean())
        ws.append(m.mean())
    # ECE
    ece = 0.0
    for x, y, w in zip(xs, ys, ws):
        ece += abs(y - x) * w
    return np.array(xs), np.array(ys), float(ece)


def plot_cls_prediction_diagnostics(data):
    idx2label = data["idx2label"]
    labels_order = sorted(idx2label.keys())
    class_names = [class_display(idx2label[i]) for i in labels_order]

    _, _, test_loader = create_data_loaders(data)
    device = torch.device("cpu")

    model_files = {
        "FastText": os.path.join(MODEL_DIR, "lstm_fasttext.pt"),
        "Word2Vec": os.path.join(MODEL_DIR, "lstm_word2vec.pt"),
        "CC_Pretrained": os.path.join(MODEL_DIR, "lstm_cc_pretrained.pt"),
        "Random": os.path.join(MODEL_DIR, "lstm_random.pt"),
    }

    summary = {}
    for name, mpath in model_files.items():
        if not os.path.exists(mpath):
            continue

        ck = torch.load(mpath, map_location="cpu", weights_only=False)
        model = LSTMClassifier(
            vocab_size=len(data["vocab"]),
            embed_dim=WV_DIM,
            hidden_dim=LSTM_HIDDEN_DIM,
            num_classes=len(data["label2idx"]),
            num_layers=LSTM_NUM_LAYERS,
            bidirectional=LSTM_BIDIRECTIONAL,
            dropout=0,
        ).to(device)
        model.load_state_dict(ck["model_state_dict"])

        y_true, y_pred, conf = get_predictions_with_confidence(model, test_loader, device)
        cm = confusion_matrix(y_true, y_pred, labels=labels_order)

        # Top confusion pairs (off-diagonal).
        off = cm.copy()
        np.fill_diagonal(off, 0)
        pairs = []
        for i in range(off.shape[0]):
            for j in range(off.shape[1]):
                if off[i, j] > 0:
                    pairs.append((off[i, j], class_names[i], class_names[j]))
        pairs = sorted(pairs, reverse=True)[:8]

        if pairs:
            dfp = pd.DataFrame(pairs, columns=["count", "true", "pred"])
            dfp["pair"] = dfp["true"] + " -> " + dfp["pred"]
            plt.figure(figsize=(10, 4))
            sns.barplot(data=dfp, x="pair", y="count", color="#D95F02")
            plt.title(f"Top Confusion Pairs: {name}")
            plt.xticks(rotation=35, ha="right")
            save_fig(os.path.join(ADD_CLS_DIR, f"cls_top_confusions_{name.lower()}.png"))

        # Confidence histogram for correct vs incorrect.
        correct = (y_true == y_pred).astype(float)
        dconf = pd.DataFrame(
            {"confidence": conf, "status": np.where(correct == 1, "correct", "incorrect")}
        )
        plt.figure(figsize=(8, 4))
        sns.histplot(data=dconf, x="confidence", hue="status", bins=20, stat="density", common_norm=False)
        plt.title(f"Prediction Confidence Distribution: {name}")
        save_fig(os.path.join(ADD_CLS_DIR, f"cls_confidence_hist_{name.lower()}.png"))

        # Reliability curve.
        xs, ys, ece = reliability_curve(correct, conf, n_bins=10)
        plt.figure(figsize=(5, 5))
        plt.plot([0, 1], [0, 1], "--", color="gray", label="ideal")
        plt.plot(xs, ys, marker="o", label=f"{name} (ECE={ece:.3f})")
        plt.xlabel("Confidence")
        plt.ylabel("Accuracy")
        plt.title(f"Reliability Plot: {name}")
        plt.legend()
        save_fig(os.path.join(ADD_CLS_DIR, f"cls_reliability_{name.lower()}.png"))

        summary[name] = {
            "ece": ece,
            "top_confusions": [{"true": t, "pred": p, "count": int(c)} for c, t, p in pairs],
        }

    with open(os.path.join(ADD_CLS_DIR, "prediction_diagnostics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {os.path.join(ADD_CLS_DIR, 'prediction_diagnostics_summary.json')}")


def main():
    print("[Step 10] Generating additional visualizations...")
    ensure_dirs()
    set_plot_style()

    # Load existing reports.
    wv_report_path = os.path.join(WV_OUTPUT_DIR, "comparison_report.json")
    cls_report_path = os.path.join(CLS_OUTPUT_DIR, "evaluation_report.json")
    if not os.path.exists(wv_report_path):
        raise FileNotFoundError(f"Missing word-vector report: {wv_report_path}")
    if not os.path.exists(cls_report_path):
        raise FileNotFoundError(f"Missing classification report: {cls_report_path}")

    wv_report = load_json(wv_report_path)
    cls_report = load_json(cls_report_path)
    data = load_processed_data()

    print("\n[Word Vectors] Creating extra plots...")
    plot_wv_vocab_and_overlap(wv_report)
    plot_wv_similarity_and_delta(wv_report)
    plot_wv_neighbor_overlap(wv_report)
    plot_wv_analogy_hits(wv_report)

    print("\n[Classification] Creating extra plots...")
    plot_cls_split_distribution(data)
    plot_cls_length_and_truncation()
    plot_cls_metrics(cls_report)
    plot_cls_prediction_diagnostics(data)

    print("\nDone. Additional visualizations saved under:")
    print(f"  {ADD_OUT_DIR}")


if __name__ == "__main__":
    main()
