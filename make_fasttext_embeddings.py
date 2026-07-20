"""make_fasttext_embeddings.py

One-off helper to build and save fastText embeddings (NPZ + optional CSV).
Uses src_representation_utils for loading and building.

Usage examples:
  python make_fasttext_embeddings.py --fasttext_path data/models/cc.nl.300.vec
  python make_fasttext_embeddings.py --fasttext_path data/models/cc.nl.300.vec --out_csv data/processed/fasttext_emb.csv
"""

import argparse
import os
import pandas as pd
from src_representation_utils import load_fasttext_kv, build_fasttext_matrix, save_embeddings_npz, save_embeddings_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_csv', default=os.path.join('data','processed','emotion_norms_processed.csv'))
    parser.add_argument('--fasttext_path', required=True, help='Path to cc.nl.300.vec or .bin')
    parser.add_argument('--out_npz', default=os.path.join('data','processed','fasttext_emb.npz'))
    parser.add_argument('--out_csv', default=None, help='Optional CSV output path')
    parser.add_argument('--unk_strategy', choices=['zero','random'], default='zero')
    args = parser.parse_args()

    if not os.path.exists(args.processed_csv):
        raise FileNotFoundError(f'Processed CSV not found: {args.processed_csv}')
    df = pd.read_csv(args.processed_csv)
    words = df['Word'].astype(str).tolist()

    kv = load_fasttext_kv(args.fasttext_path)
    emb, oov = build_fasttext_matrix(words, kv, unk_strategy=args.unk_strategy)

    save_embeddings_npz(words, emb, args.out_npz)
    print(f'fastText embeddings saved: {args.out_npz} (OOV rate={oov:.4f})')
    if args.out_csv:
        save_embeddings_csv(words, emb, args.out_csv)

if __name__ == '__main__':
    main()
