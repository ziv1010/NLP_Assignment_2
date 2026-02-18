"""
Step 8: Train the LSTM classifier on BBC Hindi news dataset.
Supports training with different embedding types for comparative study.
"""
import sys, os, pickle, time
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gensim.models import Word2Vec, FastText, KeyedVectors
from step7_lstm_model import (
    LSTMClassifier, build_embedding_matrix, create_data_loaders
)

def compute_class_weights(train_data, num_classes, device):
    labels = np.array([label for _, label in train_data], dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.mean()
    print(f"  Train class counts: {counts.astype(int).tolist()}")
    print(f"  Class weights: {[round(float(w), 3) for w in weights.tolist()]}")
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y, lengths in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        output = model(batch_x, lengths)
        loss = criterion(output, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        preds = output.argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_x.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y, lengths in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            output = model(batch_x, lengths)
            loss = criterion(output, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            preds = output.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)

    return total_loss / total, correct / total


def train_model(embedding_name, wv_model, data, device, num_epochs_override=None):
    """Train LSTM with specific embeddings and return history."""
    print(f"\n{'='*60}")
    print(f"Training LSTM with {embedding_name} embeddings")
    print(f"{'='*60}")

    vocab = data['vocab']
    num_classes = len(data['label2idx'])

    # Build embedding matrix
    if wv_model is not None:
        embed_matrix = build_embedding_matrix(vocab, wv_model)
    else:
        embed_matrix = None
        print("  Using random embeddings (no pretrained)")

    # Create model
    model = LSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=WV_DIM,
        hidden_dim=LSTM_HIDDEN_DIM,
        num_classes=num_classes,
        num_layers=LSTM_NUM_LAYERS,
        bidirectional=LSTM_BIDIRECTIONAL,
        dropout=LSTM_DROPOUT,
        pretrained_embeddings=embed_matrix,
        freeze_embeddings=False,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {total_params:,} (trainable: {trainable:,})")

    # Training setup
    class_weights = compute_class_weights(data['train'], num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LSTM_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    train_loader, val_loader, test_loader = create_data_loaders(data)
    num_epochs = num_epochs_override if num_epochs_override is not None else get_lstm_epochs()

    # Training loop
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
    }
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        elapsed = time.time() - start

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch+1}/{num_epochs} ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | LR: {lr:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= LSTM_PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        model = model.to(device)

    # Save model
    model_path = os.path.join(MODEL_DIR, f"lstm_{embedding_name.lower().replace(' ', '_')}.pt")
    torch.save({
        'model_state_dict': best_model_state or model.state_dict(),
        'vocab_size': len(vocab),
        'num_classes': num_classes,
        'embedding_name': embedding_name,
        'history': history,
    }, model_path)
    print(f"  Saved model to {model_path}")

    return model, history


def plot_training_histories(all_histories):
    """Plot training curves for all embedding types."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, hist in all_histories.items():
        epochs = range(1, len(hist['train_loss']) + 1)

        # Loss
        axes[0].plot(epochs, hist['train_loss'], '--', label=f'{name} (train)', alpha=0.7)
        axes[0].plot(epochs, hist['val_loss'], '-', label=f'{name} (val)', linewidth=2)

        # Accuracy
        axes[1].plot(epochs, hist['train_acc'], '--', label=f'{name} (train)', alpha=0.7)
        axes[1].plot(epochs, hist['val_acc'], '-', label=f'{name} (val)', linewidth=2)

    axes[0].set_title('Training & Validation Loss', fontsize=13)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_title('Training & Validation Accuracy', fontsize=13)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.suptitle('LSTM Training: Embedding Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(CLS_OUTPUT_DIR, "training_curves_comparison.png"), dpi=150)
    plt.close()
    print(f"  Saved training_curves_comparison.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Hindi LSTM classifiers with selectable embeddings.")
    parser.add_argument(
        "--embeddings",
        type=str,
        default="fasttext,word2vec,cc_pretrained,random",
        help="Comma-separated list from: fasttext,word2vec,cc_pretrained,random",
    )
    parser.add_argument(
        "--epochs_override",
        type=int,
        default=None,
        help="Override number of training epochs for faster reruns/resume training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("[Step 8] Training LSTM classifier...")
    set_seed(RANDOM_SEED)

    # Load processed data
    data_path = os.path.join(BBC_PROCESSED_DIR, "processed_data.pkl")
    if not os.path.exists(data_path):
        print("  ERROR: Processed data not found. Run step6 first.")
        return

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    print(f"  Train: {len(data['train'])}, Val: {len(data['val'])}, Test: {len(data['test'])}")
    print(f"  Classes: {data['label2idx']}")
    print(f"  Device: {DEVICE}")

    all_histories = {}
    all_models = {}
    selected = {x.strip().lower() for x in args.embeddings.split(",") if x.strip()}
    print(f"  Selected embeddings: {sorted(selected)}")
    if args.epochs_override is not None:
        print(f"  Epoch override: {args.epochs_override}")

    # 1. Train with FastText embeddings (custom trained)
    if "fasttext" in selected and os.path.exists(FASTTEXT_MODEL_PATH):
        ft_model = FastText.load(FASTTEXT_MODEL_PATH)
        model, hist = train_model("FastText", ft_model.wv, data, DEVICE, args.epochs_override)
        all_histories["FastText"] = hist
        all_models["FastText"] = model
    else:
        print("  Skipping FastText")

    # 2. Train with Word2Vec embeddings (custom trained)
    if "word2vec" in selected and os.path.exists(WORD2VEC_MODEL_PATH):
        w2v_model = Word2Vec.load(WORD2VEC_MODEL_PATH)
        model, hist = train_model("Word2Vec", w2v_model.wv, data, DEVICE, args.epochs_override)
        all_histories["Word2Vec"] = hist
        all_models["Word2Vec"] = model
    else:
        print("  Skipping Word2Vec")

    # 3. Train with Common Crawl Hindi pretrained FastText embeddings
    if "cc_pretrained" in selected and os.path.exists(PRETRAINED_VEC_PATH):
        print("\n  Loading Common Crawl Hindi pretrained vectors...")
        cc_wv = KeyedVectors.load_word2vec_format(
            PRETRAINED_VEC_PATH, binary=False, limit=PRETRAINED_LOAD_LIMIT
        )
        model, hist = train_model("CC_Pretrained", cc_wv, data, DEVICE, args.epochs_override)
        all_histories["CC_Pretrained"] = hist
        all_models["CC_Pretrained"] = model
        del cc_wv  # free memory
    else:
        print("  Skipping CC Pretrained")

    # 4. Train with random embeddings (baseline)
    if "random" in selected:
        model, hist = train_model("Random", None, data, DEVICE, args.epochs_override)
        all_histories["Random"] = hist
        all_models["Random"] = model
    else:
        print("  Skipping Random")

    # Plot comparison
    if all_histories:
        plot_training_histories(all_histories)

    print(f"\n  All models trained! Results in {CLS_OUTPUT_DIR}")
    return all_models, all_histories


if __name__ == "__main__":
    main()
