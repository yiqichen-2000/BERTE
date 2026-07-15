"""Minimal sequence helpers for BERTE pretraining."""

ALPHABET = "ACGT"


def seq_to_kmers_500(seq, k=3, stride=None, pad=True, to_upper=True, target_len=500):
    """Split a DNA sequence into k-mers and right-pad to target_len with the PAD token."""
    if stride is None:
        stride = k
    if k == 1 and stride == 1:
        kmers = list(seq.upper() if to_upper else seq)
    else:
        kmers = []
        for i in range(0, len(seq) - k + 1, stride):
            kmer = seq[i:i + k]
            kmers.append(kmer.upper() if to_upper else kmer)

        remaining_start = (len(seq) - k) // stride * stride + stride
        if pad and len(seq) > remaining_start:
            remaining_seq = seq[remaining_start:]
            if remaining_seq:
                kmers.append(remaining_seq.ljust(k, "N"))

    if pad and len(kmers) < target_len:
        kmers.extend([""] * (target_len - len(kmers)))
    return kmers[:target_len]
