"""
Step 7: LSTM model definition and Dataset class.
Bidirectional LSTM with pretrained embeddings for Hindi news classification.
"""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
from gensim.models import Word2Vec, FastText


# ----------- Dataset -----------
class HindiNewsDataset(Dataset):
    """Dataset for Hindi news classification."""

    def __init__(self, data_list):
        """
        data_list: list of (token_indices, label) tuples
        """
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        indices, label = self.data[idx]
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def collate_fn(batch):
    """Pad sequences in a batch to the same length."""
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences])
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return padded, labels, lengths


# ----------- Model -----------
class LSTMClassifier(nn.Module):
    """Bidirectional LSTM text classifier with pretrained embeddings."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 num_layers=2, bidirectional=True, dropout=0.3,
                 pretrained_embeddings=None, freeze_embeddings=False):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Attention mechanism for better classification
        self.attention = nn.Linear(lstm_output_dim, 1)

        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, lengths):
        # x: (batch, seq_len)
        embedded = self.dropout(self.embedding(x))  # (batch, seq_len, embed_dim)

        # Pack for efficiency
        packed = pack_padded_sequence(embedded, lengths.cpu().clamp(min=1),
                                      batch_first=True, enforce_sorted=False)
        lstm_out, (hidden, cell) = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        # lstm_out: (batch, seq_len, hidden_dim * 2)

        # Attention-weighted sum
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, seq_len, 1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden_dim * 2)

        output = self.fc(context)
        return output


def build_embedding_matrix(vocab, wv_model, embed_dim=WV_DIM):
    """Build embedding matrix from trained word vectors."""
    vocab_size = len(vocab)
    matrix = np.random.uniform(-0.1, 0.1, (vocab_size, embed_dim)).astype(np.float32)
    matrix[0] = 0  # PAD token

    found = 0
    for word, idx in vocab.items():
        if word in wv_model:
            matrix[idx] = wv_model[word]
            found += 1

    print(f"  Embedding matrix: {found}/{vocab_size} words covered "
          f"({100*found/vocab_size:.1f}%)")
    return torch.tensor(matrix)


def create_data_loaders(data, batch_size=LSTM_BATCH_SIZE):
    """Create DataLoaders for train/val/test."""
    train_dataset = HindiNewsDataset(data['train'])
    val_dataset = HindiNewsDataset(data['val'])
    test_dataset = HindiNewsDataset(data['test'])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    return train_loader, val_loader, test_loader
