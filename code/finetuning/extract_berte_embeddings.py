#!/usr/bin/env python3
"""Write BERTE embeddings for a prepared test split."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow import keras
import keras_bert

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model_utils import configure_tensorflow  # noqa: E402
from sequence_utils import get_token_dict, seq2tokens  # noqa: E402


class TestFragmentGenerator(Sequence):
    def __init__(self, fragments, seq_len, max_seq_len, k, stride, batch_size):
        self.fragments = fragments
        self.seq_len = seq_len
        self.max_seq_len = max_seq_len
        self.k = k
        self.stride = stride
        self.batch_size = batch_size
        self.token_dict = get_token_dict(k=k)

    def __len__(self):
        return int(np.ceil(len(self.fragments) / float(self.batch_size)))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        batch = [
            seq2tokens(
                seq,
                self.token_dict,
                seq_length=self.seq_len,
                max_length=self.max_seq_len,
                k=self.k,
                stride=self.stride,
                window=False,
                seq_len_like=None,
            )
            for seq in self.fragments[start:end]
        ]
        return [
            np.asarray([item[0] for item in batch]),
            np.asarray([item[1] for item in batch]),
        ]


def load_model(path: Path):
    custom_objects = {
        "GlorotNormal": keras.initializers.glorot_normal,
        "GlorotUniform": keras.initializers.glorot_uniform,
    }
    custom_objects.update(keras_bert.get_custom_objects())
    return keras.models.load_model(path, compile=False, custom_objects=custom_objects)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Fine-tuned BERTE model .h5")
    parser.add_argument("fragments_dir", type=Path, help="Directory containing <prefix>_fragments.json")
    parser.add_argument("--prefix", default="test")
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--stride", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=502)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    configure_tensorflow()
    fragments_path = args.fragments_dir / f"{args.prefix}_fragments.json"
    with fragments_path.open("r", encoding="utf-8") as handle:
        fragments = json.load(handle)
    if not fragments:
        raise ValueError(f"No fragments found in {fragments_path}")

    model = load_model(args.model)
    max_seq_len = int(model.input_shape[0][1])
    seq_len = min(args.seq_len, max_seq_len)
    if args.seq_len > max_seq_len:
        print(f"Requested seq_len={args.seq_len}; model max_seq_len={max_seq_len}; using seq_len={seq_len}")

    generator = TestFragmentGenerator(
        fragments=fragments,
        seq_len=seq_len,
        max_seq_len=max_seq_len,
        k=args.k,
        stride=args.stride,
        batch_size=args.batch_size,
    )
    nsp_layer = model.get_layer(name="NSP-Dense").output
    nsp_model = keras.Model(inputs=model.inputs[:2], outputs=nsp_layer)
    output = nsp_model.predict(generator, verbose=1)
    output = np.asarray(output, dtype=np.float32)

    out_path = args.output or (args.fragments_dir / f"{args.prefix}_{args.k}mer_embeddings.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(output, handle)
    print(f"NSP-Dense output shape: {output.shape}")
    print(f"Saved NSP-Dense output to: {out_path}")


if __name__ == "__main__":
    main()
