"""
Utility functions for extracting and saving three representation families used in
this thesis benchmark: character TF-IDF n-grams, fastText static embeddings, and
transformer-derived embeddings (BERTje) with mean pooling.

Designed to be run separately (extraction + caching) and then used by the
benchmarking script (09_pilot_modeling.py) which loads cached files.

Usage examples:
  python src_representation_utils.py --mode bert --processed_csv data/processed/emotion_norms_processed.csv --out_npz data/processed/bertje_emb.npz
  python src_representation_utils.py --mode fasttext --fasttext_path data/models/cc.nl.300.vec --out_npz data/processed/fasttext_emb.npz

"""

import os
import argparse
import sys
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

# fastText via gensim 
try:
    from gensim.models import KeyedVectors
    GENSIM_AVAILABLE = True
except Exception:
    GENSIM_AVAILABLE = False

# Transformers 
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# Defaults
DEFAULT_NGRAM_RANGE = (2, 4)
DEFAULT_MIN_DF = 3
DEFAULT_MAX_FEATURES = 50000
BERTJE_DEFAULT = "GroNLP/bert-base-dutch-cased"

# Character n-gram TF-IDF

def build_char_ngram_tfidf(words: List[str], ngram_range: Tuple[int,int]=DEFAULT_NGRAM_RANGE,
                           min_df: int=DEFAULT_MIN_DF, max_features: int=DEFAULT_MAX_FEATURES,
                           lowercase: bool=True):
    """Fit a char-level TF-IDF (char_wb). Returns fitted vectorizer and sparse matrix X."""
    vec = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        lowercase=lowercase,
        strip_accents=None,
    )
    vec.fit(words)
    X = vec.transform(words)
    return vec, X

# fastText static loader + builder

def load_fasttext_kv(path: str):
    """Load fastText KeyedVectors (.bin or .vec) using gensim. Raises if gensim missing."""
    if not GENSIM_AVAILABLE:
        raise ImportError("gensim is required to load fastText vectors. Install gensim.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"fastText file not found: {path}")
    print(f"Loading fastText vectors from {path} ...")
    if path.endswith('.bin'):
        kv = KeyedVectors.load_word2vec_format(path, binary=True)
    else:
        kv = KeyedVectors.load_word2vec_format(path, binary=False)
    print("fastText loaded.")
    return kv


def build_fasttext_matrix(words: List[str], kv: KeyedVectors, unk_strategy: str='zero') -> Tuple[np.ndarray, float]:
    """Return (emb_matrix, oov_rate) where emb_matrix.shape = (len(words), dim)."""
    dim = kv.vector_size
    emb = np.zeros((len(words), dim), dtype=np.float32)
    n_miss = 0
    rng = np.random.default_rng(42)
    for i, w in enumerate(words):
        try:
            emb[i] = kv[w]
        except Exception:
            n_miss += 1
            if unk_strategy == 'zero':
                emb[i] = np.zeros(dim, dtype=np.float32)
            else:
                emb[i] = rng.normal(0, 0.01, size=(dim,)).astype(np.float32)
    return emb, n_miss / len(words)


# Transformer extraction (BERTje) with mean pooling

def extract_transformer_meanpool(words: List[str], model_id: str = BERTJE_DEFAULT,
                                 batch_size: int = 256, device: Optional[str] = None,
                                 cache_path: Optional[str] = None) -> np.ndarray:
    """Extract mean-pooled embeddings for a list of words and cache (npz).

    If cache_path exists, it will be loaded and returned. If not, embeddings are
    computed and saved to cache_path if provided.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading transformer embeddings from cache: {cache_path}")
        with np.load(cache_path, allow_pickle=True) as d:
            return d['emb']

    if not TORCH_AVAILABLE:
        raise ImportError("torch + transformers are required for transformer extraction")

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device={device} for transformer extraction")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()

    out_embs = []
    with torch.no_grad():
        for i in range(0, len(words), batch_size):
            batch = words[i:i+batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
            enc = {k:v.to(device) for k,v in enc.items()}
            outputs = model(**enc)
            last = outputs.last_hidden_state  # (B, L, H)
            mask = enc['attention_mask'].unsqueeze(-1)  # (B,L,1)
            summed = (last * mask).sum(dim=1)  # (B,H)
            counts = mask.sum(dim=1).clamp(min=1)
            mean_pooled = (summed / counts).cpu().numpy()
            out_embs.append(mean_pooled)
    emb = np.vstack(out_embs)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, emb=emb)
        print(f"Saved transformer embeddings to cache: {cache_path}")
    return emb


# Save/load helpers (NPZ/CSV)

def save_embeddings_npz(words: List[str], emb: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, words=np.array(words, dtype=object), emb=emb)
    print(f"Saved embeddings NPZ: {path}")


def load_embeddings_npz(path: str) -> Tuple[List[str], np.ndarray]:
    with np.load(path, allow_pickle=True) as d:
        return d['words'].tolist(), d['emb']


def save_embeddings_csv(words: List[str], emb: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(emb, index=words)
    df.index.name = 'word'
    df.to_csv(path)
    print(f"Saved embeddings CSV: {path}")

# CLI wrapper when running this module directly
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['ngram','fasttext','bert'], required=True)
    parser.add_argument('--processed_csv', default='data/processed/emotion_norms_processed.csv')
    parser.add_argument('--fasttext_path', default='data/models/cc.nl.300.vec')
    parser.add_argument('--fasttext_out', default='data/processed/fasttext_emb.npz')
    parser.add_argument('--bert_out', default='data/processed/bertje_emb.npz')
    parser.add_argument('--out_csv', default=None)
    args = parser.parse_args()

    if not os.path.exists(args.processed_csv):
        print('Processed dataset not found:', args.processed_csv)
        sys.exit(1)
    df = pd.read_csv(args.processed_csv)
    words = df['Word'].astype(str).tolist()

    if args.mode == 'ngram':
        vec, X = build_char_ngram_tfidf(words)
        print('TF-IDF built. shape:', X.shape)
    elif args.mode == 'fasttext':
        if not GENSIM_AVAILABLE:
            print('gensim not available. pip install gensim')
            sys.exit(1)
        kv = load_fasttext_kv(args.fasttext_path)
        emb, oov = build_fasttext_matrix(words, kv)
        save_embeddings_npz(words, emb, args.fasttext_out)
        print('fastText OOV rate:', oov)
        if args.out_csv:
            save_embeddings_csv(words, emb, args.out_csv)
    elif args.mode == 'bert':
        if not TORCH_AVAILABLE:
            print('torch/transformers not installed. pip install torch transformers')
            sys.exit(1)
        emb = extract_transformer_meanpool(words, cache_path=args.bert_out)
        if args.out_csv:
            save_embeddings_csv(words, emb, args.out_csv)
        print('BERT extraction done; shape:', emb.shape)
