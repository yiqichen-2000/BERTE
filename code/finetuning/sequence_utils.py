"""Minimal tokenization helpers for direct BERTE fine-tuning."""

from itertools import product
from random import randint

import numpy as np

ALPHABET = "ACGT"
TOKEN_PAD = ""
TOKEN_UNK = "[UNK]"
TOKEN_CLS = "[CLS]"
TOKEN_SEP = "[SEP]"
TOKEN_MASK = "[MASK]"


def get_base_dict():
    return {
        TOKEN_PAD: 0,
        TOKEN_UNK: 1,
        TOKEN_CLS: 2,
        TOKEN_SEP: 3,
        TOKEN_MASK: 4,
    }


def get_token_dict(alph=ALPHABET, k=3):
    token_dict = get_base_dict()
    for word in ("".join(parts) for parts in product(alph, repeat=k)):
        token_dict[word] = len(token_dict)
    return token_dict


def seq2kmers(seq, k=3, stride=3, pad=True, to_upper=True):
    if k == 1 and stride == 1:
        return seq.upper() if to_upper else seq
    kmers = []
    for i in range(0, len(seq) - k + 1, stride):
        kmer = seq[i:i + k]
        kmers.append(kmer.upper() if to_upper else kmer)
    remaining_start = (len(seq) - k) // stride * stride + stride
    if pad and len(seq) > remaining_start:
        remaining_seq = seq[remaining_start:]
        if remaining_seq:
            kmers.append(remaining_seq.ljust(k, "N"))
    return kmers


def seq2tokens(seq, token_dict, seq_length=250, max_length=None,
               k=3, stride=3, window=True, seq_len_like=None):
    if max_length is None:
        max_length = seq_length
    if seq_len_like is not None:
        seq_length = min(max_length, np.random.choice(seq_len_like))
    words = seq2kmers(seq, k=k, stride=stride, pad=True)
    if window:
        start = randint(0, max(len(words) - seq_length - 1, 0))
        end = start + seq_length - 1
    else:
        start = 0
        end = seq_length
    indices = [token_dict[TOKEN_CLS]] + [
        token_dict[word] if word in token_dict else token_dict[TOKEN_UNK]
        for word in words[start:end]
    ]
    if len(indices) < max_length:
        indices += [token_dict[TOKEN_PAD]] * (max_length - len(indices))
    else:
        indices = indices[:max_length]
    segments = [0 for _ in range(max_length)]
    return [np.array(indices), np.array(segments)]


def get_class_vectors(classes):
    return {c: [1 if c_i == i else 0 for i in range(len(classes))]
            for c_i, c in enumerate(sorted(classes))}
