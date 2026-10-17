"""Minimal fragment loading helpers for BERTE pretraining."""

import json
import os
import random


def _read_json(path):
    with open(path, "r") as fh:
        return json.load(fh)


def _read_lines(path):
    with open(path, "r") as fh:
        return [line.strip() for line in fh if line.strip()]


def load_fragments_split(fragments_dir, classes, split, shuffle_=True, balance=False, nr_seqs=None, seed=42):
    """Load class-specific fragment JSON and superfamily text files for one split."""
    fragments_by_class = []
    superfamilies_by_class = []

    for class_name in classes:
        fragments_path = os.path.join(fragments_dir, f"{class_name}_{split}_fragments.json")
        superfamily_path = os.path.join(fragments_dir, f"{class_name}_{split}_fragment_superfamilies.txt")
        class_fragments = _read_json(fragments_path)
        class_superfamilies = _read_lines(superfamily_path)
        if len(class_fragments) != len(class_superfamilies):
            raise ValueError(
                f"{class_name} {split}: fragments ({len(class_fragments)}) and "
                f"superfamilies ({len(class_superfamilies)}) have different lengths"
            )
        fragments_by_class.append((class_name, class_fragments))
        superfamilies_by_class.append(class_superfamilies)

    nr_seqs_max = min(len(class_fragments) for _, class_fragments in fragments_by_class)
    if nr_seqs is None or nr_seqs > nr_seqs_max:
        nr_seqs = nr_seqs_max

    if balance and nr_seqs == 0:
        raise ValueError(
            'Balanced pretraining requires a non-empty input for every class.'
        )

    rng = random.Random(seed)
    
    x = []
    y = []
    y_superfamilies = []

    for index, (class_name, class_fragments) in enumerate(fragments_by_class):
        class_superfamilies = superfamilies_by_class[index]
        if balance:
            picked = rng.sample(list(zip(class_fragments, class_superfamilies)), nr_seqs)
            picked_fragments, picked_superfamilies = zip(*picked)
            x.extend(picked_fragments)
            y_superfamilies.extend(picked_superfamilies)
            y.extend([class_name] * nr_seqs)
        else:
            x.extend(class_fragments)
            y.extend([class_name] * len(class_fragments))
            y_superfamilies.extend(class_superfamilies)

    if len(x) != len(y):
        raise AssertionError("fragment and label counts differ")

    if shuffle_:
        combined = list(zip(x, y, y_superfamilies))
        rng.shuffle(combined)
        x, y, y_superfamilies = zip(*combined)

    print(f"{len(x)} fragments loaded in total; balanced={balance}, shuffle_={shuffle_}, nr_seqs={nr_seqs}")
    return list(x), list(y), list(y_superfamilies)
