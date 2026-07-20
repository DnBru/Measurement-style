"""make_bertje_embeddings.py

Helper script to extract BERTje embeddings for the processed word list and save
as NPZ (cached) and optional CSV. Uses mean pooling over token embeddings.

Usage examples:
  python make_bertje_embeddings.py --model_id GroNLP/bert-base-dutch-cased
  python make_bertje_embeddings.py --model_id GroNLP/bert-base-dutch-cased --out_csv data/processed/bertje_emb.csv
"""

import argparse
import os
import pandas as pd
from src_representation_utils import (
    extract_transformer_meanpool,
    save_embeddings_npz,
    save_embeddings_csv,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed_csv",
        default=os.path.join("data", "processed", "emotion_norms_processed.csv")
    )
    parser.add_argument("--word_col", default="Word")
    parser.add_argument("--model_id", default="GroNLP/bert-base-dutch-cased")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--out_npz",
        default=os.path.join("data", "processed", "bertje_emb.npz")
    )
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()

    if not os.path.exists(args.processed_csv):
        raise FileNotFoundError(f"Processed CSV not found: {args.processed_csv}")

    df = pd.read_csv(args.processed_csv)

    if args.word_col not in df.columns:
        raise KeyError(f"Word column '{args.word_col}' not found in {args.processed_csv}")

    words = df[args.word_col].astype(str).tolist()

    emb = extract_transformer_meanpool(
        words,
        model_id=args.model_id,
        batch_size=args.batch_size,
        device=args.device,
        cache_path=None,   # IMPORTANT: do not let the utility save an emb-only cache
    )

    # Save NPZ with BOTH words and embeddings
    save_embeddings_npz(words, emb, args.out_npz)

    print(f"BERTje embeddings shape: {emb.shape}")

    # optional CSV
    if args.out_csv:
        save_embeddings_csv(words, emb, args.out_csv)


if __name__ == "__main__":
    main()