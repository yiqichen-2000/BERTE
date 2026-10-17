#!/usr/bin/env python3
"""Train and evaluate BERTE hierarchical RF models from BERT and CDD features.

Required files in fragments_dir:
  train/test_{3,4,5,6}mer_embeddings.pkl
  train/test_cdd_features.txt
  train/test_fragment_superfamilies.txt
  train/test_fragment_headers.txt

Generate these intermediate files by following the Model Training Workflow in README.md.
If this module is called directly, generate BERTE NSP embeddings and CDD protein-domain
features for train/test first. The embedding files are fragment-level outputs;
this script averages them back to sequence-level features, concatenates protein
features, trains a top-down RandomForest hierarchy on train, and evaluates on
test.
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

CLASS_I = {
    "Bel-Pao", "Copia", "Gypsy", "ERV",
    "DIRS", "PLE", "CR1", "I", "Jockey", "L1", "R2", "RTE", "Rex1",
    "SINE1/7SL", "SINE2/tRNA", "SINE3/5S", "ID",
}
CLASS_II = {"CACTA", "MULE", "PIF", "TcMar", "hAT", "Helitron"}
ORDER_BY_SUPERFAMILY = {
    "Bel-Pao": "LTR", "Copia": "LTR", "Gypsy": "LTR", "ERV": "LTR",
    "DIRS": "DIRS", "PLE": "PLE",
    "CR1": "LINE", "I": "LINE", "Jockey": "LINE", "L1": "LINE",
    "R2": "LINE", "RTE": "LINE", "Rex1": "LINE",
    "SINE1/7SL": "SINE", "SINE2/tRNA": "SINE", "SINE3/5S": "SINE", "ID": "SINE",
    "CACTA": "TIR", "MULE": "TIR", "PIF": "TIR", "TcMar": "TIR", "hAT": "TIR",
    "Helitron": "Helitron",
}
PROTEIN_EVALUE_THRESHOLD = 0.1


def normalize_superfamily(label: str) -> str:
    return str(label)


def get_taxonomy_path(superfamily: str) -> Tuple[str, str, str]:
    sf = normalize_superfamily(superfamily)
    class_name = "class1" if sf in CLASS_I else "class2" if sf in CLASS_II else "unknown"
    order = ORDER_BY_SUPERFAMILY.get(sf, "unknown")
    return class_name, order, sf


def read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def validate_required_inputs(file_pdb: Path, fragments_dir: Path, splits: Iterable[str]) -> None:
    required = [file_pdb]
    for split in splits:
        required.extend([
            fragments_dir / f"{split}_cdd_features.txt",
            fragments_dir / f"{split}_fragment_superfamilies.txt",
            fragments_dir / f"{split}_fragment_headers.txt",
        ])
        required.extend(fragments_dir / f"{split}_{k}mer_embeddings.pkl" for k in (3, 4, 5, 6))

    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(
            "Missing RF training inputs:\n  "
            + "\n  ".join(missing)
            + "\nGenerate these files by following the Model Training Workflow in README.md. "
            "When calling train_hierarchical_rf.py directly, first generate "
            "the *_embeddings.pkl and *_cdd_features.txt files."
        )


def load_protein_features(path: Path, n_features: int) -> np.ndarray:
    rows: List[List[int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("@")[-n_features:]
            row = []
            for value in parts:
                values = [float(item) for item in value.split(",") if item]
                row.append(1 if values and min(values) <= PROTEIN_EVALUE_THRESHOLD else 0)
            rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def sequence_base(fragment_header: str) -> str:
    return re.split(r"_\d+", fragment_header)[0]


def average_embeddings_by_sequence(x: np.ndarray, fragment_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, fragment_name in enumerate(fragment_names):
        groups[sequence_base(fragment_name)].append(idx)

    seq_names: List[str] = []
    averaged: List[np.ndarray] = []
    for seq_name, indices in groups.items():
        seq_names.append(seq_name)
        averaged.append(x[indices].mean(axis=0))
    return np.stack(averaged, axis=0), seq_names


def load_split_features(file_pdb: Path, fragments_dir: Path, split: str) -> Tuple[np.ndarray, List[str], List[str]]:
    n_proteins = len(read_lines(file_pdb))
    protein_features = load_protein_features(fragments_dir / f"{split}_cdd_features.txt", n_proteins)

    labels = read_lines(fragments_dir / f"{split}_fragment_superfamilies.txt")
    fragment_names = read_lines(fragments_dir / f"{split}_fragment_headers.txt")

    embedding_outputs = []
    for k in (3, 4, 5, 6):
        path = fragments_dir / f"{split}_{k}mer_embeddings.pkl"
        with path.open("rb") as handle:
            data = pickle.load(handle)
        print(f"Loaded {k}-mer {split} embeddings with shape: {data.shape}")
        embedding_outputs.append(np.asarray(data))

    n_samples = embedding_outputs[0].shape[0]
    if any(arr.shape[0] != n_samples for arr in embedding_outputs):
        raise ValueError(f"Inconsistent embedding row counts for split {split}")

    fragment_features = np.concatenate(embedding_outputs, axis=1)
    if fragment_features.shape[0] != len(labels) or fragment_features.shape[0] != len(fragment_names):
        raise ValueError(
            f"{split} row mismatch: embeddings={fragment_features.shape[0]}, "
            f"labels={len(labels)}, fragments={len(fragment_names)}"
        )

    averaged_embeddings, seq_names = average_embeddings_by_sequence(fragment_features, fragment_names)
    if protein_features.shape[0] != averaged_embeddings.shape[0]:
        raise ValueError(
            f"{split} protein/embedding sequence mismatch: protein={protein_features.shape[0]}, "
            f"averaged_embeddings={averaged_embeddings.shape[0]}"
        )

    seq_to_labels: Dict[str, List[str]] = defaultdict(list)
    for fragment_name, label in zip(fragment_names, labels):
        seq_to_labels[sequence_base(fragment_name)].append(label)
    seq_labels = [normalize_superfamily(seq_to_labels[name][0]) for name in seq_names]

    features = np.concatenate((protein_features, averaged_embeddings), axis=1).astype(np.float32)
    print(f"{split}: fragment_features={fragment_features.shape}, sequence_features={features.shape}")
    return features, seq_labels, seq_names


def build_node_list(paths: Iterable[str]) -> List[str]:
    nodes = set()
    for path in paths:
        parts = path.split("|")
        for depth in range(1, len(parts) + 1):
            nodes.add("|".join(parts[:depth]))
    return sorted(nodes, key=lambda item: (item.count("|"), item))


def get_children(nodes: Iterable[str]) -> Dict[str | None, List[str]]:
    children: Dict[str | None, List[str]] = defaultdict(list)
    for node in nodes:
        parent = node.rsplit("|", 1)[0] if "|" in node else None
        children[parent].append(node)
    for key in list(children):
        children[key] = sorted(children[key])
    return children


def train_hierarchy(x_train: np.ndarray, train_paths: List[str], nodes: List[str], threads: int) -> Dict[str, RandomForestClassifier]:
    models: Dict[str, RandomForestClassifier] = {}
    n = x_train.shape[0]
    for node in nodes:
        parent = node.rsplit("|", 1)[0] if "|" in node else None
        if parent is None:
            indices = np.arange(n)
        else:
            prefix = parent + "|"
            indices = np.asarray([i for i, path in enumerate(train_paths) if path.startswith(prefix)], dtype=int)
        if indices.size == 0:
            continue

        y_node = np.asarray([
            1 if (train_paths[i] == node or train_paths[i].startswith(node + "|")) else 0
            for i in indices
        ], dtype=int)
        if y_node.sum() == 0 or y_node.sum() == len(y_node):
            continue

        clf = RandomForestClassifier(n_jobs=threads, random_state=42)
        clf.fit(x_train[indices], y_node)
        models[node] = clf
        print(f"[Train] node='{node}' pos={int(y_node.sum())} neg={int(len(y_node) - y_node.sum())}")
    return models


def predict_hierarchy(models: Dict[str, RandomForestClassifier], x_test: np.ndarray, nodes: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    children = get_children(nodes)
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    pred_matrix = np.zeros((x_test.shape[0], len(nodes)), dtype=int)
    prob_matrix = np.zeros((x_test.shape[0], len(nodes)), dtype=float)
    pred_paths: List[str] = []

    for i, row in enumerate(x_test):
        sample = row.reshape(1, -1)
        parent = None
        chosen: List[str] = []
        while True:
            candidates = children.get(parent, [])
            if not candidates:
                break
            if len(candidates) == 1:
                best_node, best_score = candidates[0], 1.0
            else:
                best_node, best_score = None, -1.0
                for node in candidates:
                    model = models.get(node)
                    if model is None:
                        continue
                    score = model.predict_proba(sample)[0, 1]
                    prob_matrix[i, node_to_idx[node]] = score
                    if score > best_score:
                        best_node, best_score = node, score
            if best_node is None:
                break
            pred_matrix[i, node_to_idx[best_node]] = 1
            prob_matrix[i, node_to_idx[best_node]] = best_score
            chosen.append(best_node)
            parent = best_node
        pred_paths.append(chosen[-1] if chosen else "unknown")
    return pred_matrix, prob_matrix, pred_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the BERTE hierarchical RF classifier")
    parser.add_argument("fragments_dir", type=Path)
    parser.add_argument("--filePDB", required=True, type=Path)
    parser.add_argument("--save_name", default="RF")
    parser.add_argument("--threads", type=int, default=1, help="CPU cores used by each Random Forest node model.")
    parser.add_argument("--store_predictions", action="store_true", help="Accepted for compatibility; predictions are always stored.")
    args = parser.parse_args()
    if args.threads == 0:
        raise ValueError("--threads must be a positive integer or -1 for all available CPU cores.")

    validate_required_inputs(args.filePDB, args.fragments_dir, ("train", "test"))

    train_x, train_labels, train_names = load_split_features(args.filePDB, args.fragments_dir, "train")
    test_x, test_labels, test_names = load_split_features(args.filePDB, args.fragments_dir, "test")

    expected_superfamilies = CLASS_I | CLASS_II
    observed_superfamilies = set(train_labels)

    missing = sorted(expected_superfamilies - observed_superfamilies)
    unsupported = sorted(observed_superfamilies - expected_superfamilies)

    if missing or unsupported:
        raise ValueError(
            "RF training requires all 23 supported superfamilies. "
            f"Missing: {missing}. Unsupported: {unsupported}."
        )
    
    print(f"Loaded sequences: train={len(train_labels)}, test={len(test_labels)}")

    train_paths = ["|".join(get_taxonomy_path(label)) for label in train_labels]
    test_paths = ["|".join(get_taxonomy_path(label)) for label in test_labels]
    print("Train label counts:", dict(sorted(Counter(train_paths).items())))

    nodes = build_node_list(train_paths)
    models = train_hierarchy(train_x, train_paths, nodes, args.threads)
    model_path = f"{args.save_name}.pkl"
    joblib.dump(models, model_path)
    print("Saved models to", model_path)

    pred_matrix, prob_matrix, pred_paths = predict_hierarchy(models, test_x, nodes)
    print("=== Full-path classification report ===")
    print(classification_report(test_paths, pred_paths, zero_division=0, digits=4))

    results = {
        "nodes": nodes,
        "test_names": test_names,
        "y_true_paths": test_paths,
        "y_pred_paths": pred_paths,
        "pred_matrix": pred_matrix,
        "prob_matrix": prob_matrix,
        "train_label_counts": dict(Counter(train_paths)),
    }
    prediction_path = f"{args.save_name}_predictions.pkl"
    joblib.dump(results, prediction_path)
    print("Saved predictions to", prediction_path)


if __name__ == "__main__":
    main()
