"""
Step 9: Evaluate LSTM classifiers and generate comparative report.
Produces confusion matrices, per-class metrics, and a final comparison table.
"""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns

# Robust font fallback: keep Latin labels readable while supporting Hindi if available.
_deva_fonts = [
    "Noto Sans Devanagari", "Kohinoor Devanagari", "Devanagari Sangam MN",
    "Devanagari MT", "Arial Unicode MS",
]
_available = {f.name for f in font_manager.fontManager.ttflist}
_stack = ["DejaVu Sans"] + [f for f in _deva_fonts if f in _available]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = _stack
plt.rcParams["axes.unicode_minus"] = False
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score
)
from step7_lstm_model import LSTMClassifier, create_data_loaders
from gensim.models import Word2Vec, FastText
from step7_lstm_model import build_embedding_matrix
import json

CLASS_ASCII = {
    "खेल": "Khel",
    "भारत": "Bharat",
    "मनोरंजन": "Manoranjan",
    "विज्ञान": "Vigyan",
    "विदेश": "Videsh",
    "सोशल": "Social",
}


def display_class_names(class_names):
    return [CLASS_ASCII.get(c, c) for c in class_names]


def get_predictions(model, loader, device):
    """Get all predictions and true labels."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y, lengths in loader:
            batch_x = batch_x.to(device)
            output = model(batch_x, lengths)
            preds = output.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())

    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm.astype('float') / row_sums
    class_display = display_class_names(class_names)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_display, yticklabels=class_display, ax=axes[0])
    axes[0].set_title(f'{title}\n(Raw Counts)', fontsize=12)
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_display, yticklabels=class_display, ax=axes[1])
    axes[1].set_title(f'{title}\n(Normalized)', fontsize=12)
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def evaluate_model(model, test_loader, idx2label, embedding_name, device):
    """Evaluate a single model and return metrics."""
    print(f"\n{'-'*50}")
    print(f"Evaluating: {embedding_name}")
    print(f"{'-'*50}")

    y_pred, y_true = get_predictions(model, test_loader, device)

    # Class names
    class_names = [idx2label[i] for i in sorted(idx2label.keys())]

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

    print(f"  Accuracy:          {acc:.4f}")
    print(f"  F1 (macro):        {f1_macro:.4f}")
    print(f"  F1 (weighted):     {f1_weighted:.4f}")
    print(f"  Precision (macro): {precision:.4f}")
    print(f"  Recall (macro):    {recall:.4f}")

    # Per-class report
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print(f"\n{report}")

    # Confusion matrix
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        f'Confusion Matrix - {embedding_name}',
        os.path.join(CLS_OUTPUT_DIR, f"confusion_matrix_{embedding_name.lower().replace(' ', '_')}.png")
    )

    metrics = {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision,
        'recall_macro': recall,
        'per_class_report': classification_report(y_true, y_pred,
                                                  target_names=class_names,
                                                  zero_division=0,
                                                  output_dict=True),
    }
    return metrics


def plot_comparison_bar(all_metrics):
    """Create bar chart comparing all models."""
    metric_names = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']
    display_names = ['Accuracy', 'F1 (Macro)', 'F1 (Weighted)', 'Precision', 'Recall']

    model_names = list(all_metrics.keys())
    x = np.arange(len(display_names))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0']

    for i, model_name in enumerate(model_names):
        values = [all_metrics[model_name][m] for m in metric_names]
        bars = ax.bar(x + i * width, values, width, label=model_name,
                      color=colors[i % len(colors)], alpha=0.85)
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('LSTM Classification: Embedding Comparison', fontsize=14)
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(display_names, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(CLS_OUTPUT_DIR, "model_comparison.png"), dpi=150)
    plt.close()
    print(f"  Saved model_comparison.png")


def sample_predictions(model, test_loader, idx2label, device, n=10):
    """Show sample predictions with confidence."""
    model.eval()
    samples = []

    with torch.no_grad():
        for batch_x, batch_y, lengths in test_loader:
            batch_x = batch_x.to(device)
            output = model(batch_x, lengths)
            probs = torch.softmax(output, dim=1)
            preds = probs.argmax(dim=1)
            confidences = probs.max(dim=1).values

            for i in range(min(n - len(samples), batch_x.size(0))):
                samples.append({
                    'true': idx2label[batch_y[i].item()],
                    'pred': idx2label[preds[i].item()],
                    'confidence': confidences[i].item(),
                    'correct': batch_y[i].item() == preds[i].item(),
                })
                if len(samples) >= n:
                    break
            if len(samples) >= n:
                break

    print("\n  Sample Predictions:")
    for i, s in enumerate(samples):
        mark = "Y" if s['correct'] else "N"
        print(f"    [{i+1}] True: {s['true']:8s} | Pred: {s['pred']:8s} | "
              f"Conf: {s['confidence']:.3f} {mark}")


def main():
    print("[Step 9] Evaluating LSTM classifiers...")
    set_seed(RANDOM_SEED)

    # Load processed data
    data_path = os.path.join(BBC_PROCESSED_DIR, "processed_data.pkl")
    with open(data_path, "rb") as f:
        data = pickle.loads(f.read())

    idx2label = data['idx2label']
    vocab = data['vocab']
    num_classes = len(data['label2idx'])
    _, _, test_loader = create_data_loaders(data)

    all_metrics = {}

    # Evaluate each saved model
    embedding_types = {
        'FastText': FASTTEXT_MODEL_PATH,
        'Word2Vec': WORD2VEC_MODEL_PATH,
        'CC_Pretrained': None,
        'Random': None,
    }

    for emb_name, wv_path in embedding_types.items():
        model_path = os.path.join(MODEL_DIR, f"lstm_{emb_name.lower()}.pt")
        if not os.path.exists(model_path):
            print(f"  Skipping {emb_name} - model not found at {model_path}")
            continue

        # Load model
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

        model = LSTMClassifier(
            vocab_size=len(vocab),
            embed_dim=WV_DIM,
            hidden_dim=LSTM_HIDDEN_DIM,
            num_classes=num_classes,
            num_layers=LSTM_NUM_LAYERS,
            bidirectional=LSTM_BIDIRECTIONAL,
            dropout=0,  # no dropout during eval
        ).to(DEVICE)

        model.load_state_dict(checkpoint['model_state_dict'])

        metrics = evaluate_model(model, test_loader, idx2label, emb_name, DEVICE)
        all_metrics[emb_name] = metrics

        sample_predictions(model, test_loader, idx2label, DEVICE)

    # Comparison chart
    if len(all_metrics) > 1:
        plot_comparison_bar(all_metrics)

    # Save full report
    serializable_metrics = {}
    for name, m in all_metrics.items():
        serializable_metrics[name] = {
            k: float(v) if isinstance(v, (float, np.floating)) else v
            for k, v in m.items()
        }

    report_path = os.path.join(CLS_OUTPUT_DIR, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(serializable_metrics, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Full evaluation report saved to {report_path}")

    # Summary table
    print("\n" + "="*70)
    print("FINAL COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Embedding':<20} {'Accuracy':<12} {'F1 (macro)':<12} {'F1 (wt)':<12} {'Precision':<12} {'Recall':<12}")
    print("-"*70)
    for name, m in all_metrics.items():
        print(f"{name:<20} {m['accuracy']:<12.4f} {m['f1_macro']:<12.4f} "
              f"{m['f1_weighted']:<12.4f} {m['precision_macro']:<12.4f} {m['recall_macro']:<12.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
