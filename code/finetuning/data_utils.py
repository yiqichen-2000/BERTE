"""Minimal data loading and TE taxonomy helpers for fine-tuning."""

import json
import os
import random
import re
from collections import defaultdict

import numpy as np

CLASS1_DIRS = ['DIRS']
CLASS1_LINE = ['CR1', 'I', 'Jockey', 'L1', 'R2', 'RTE', 'Rex1']
CLASS1_LTR = ['Bel-Pao', 'Copia', 'Gypsy', 'ERV']
CLASS1_PLE = ['PLE']
CLASS1_SINE = ['SINE1/7SL', 'SINE2/tRNA', 'SINE3/5S', 'ID']
CLASS2_TIR = ['CACTA', 'MULE', 'PIF', 'TcMar', 'hAT']
CLASS2_HELITRON = ['Helitron']


def te_rank_one(taxid):
    taxid = str(taxid)
    if taxid in CLASS1_DIRS:
        return {'class': 'class1', 'order': 'DIRS', 'superfamily': taxid}
    if taxid in CLASS1_LINE:
        return {'class': 'class1', 'order': 'LINE', 'superfamily': taxid}
    if taxid in CLASS1_LTR:
        return {'class': 'class1', 'order': 'LTR', 'superfamily': taxid}
    if taxid in CLASS1_PLE:
        return {'class': 'class1', 'order': 'PLE', 'superfamily': taxid}
    if taxid in CLASS1_SINE:
        return {'class': 'class1', 'order': 'SINE', 'superfamily': taxid}
    if taxid in CLASS2_TIR:
        return {'class': 'class2', 'order': 'TIR', 'superfamily': taxid}
    if taxid in CLASS2_HELITRON:
        return {'class': 'class2', 'order': 'Helitron', 'superfamily': taxid}
    return {'class': 'unknown', 'order': 'unknown', 'superfamily': 'unknown'}


def get_te_rank(species_list, all=True):
    tax_ranks = ['class', 'order', 'superfamily']
    tax_ranks_dict = {rank: {} for rank in tax_ranks}
    teid_y = []
    last_rank = None
    for taxid in species_list:
        last_rank = te_rank_one(taxid)
        teid_y.append([last_rank[rank] for rank in tax_ranks])
        for rank in tax_ranks:
            tax_ranks_dict[rank][last_rank[rank]] = 1
    if all:
        return tax_ranks_dict
    return last_rank, teid_y


def get_classes_and_weights_multi_tax1(
    species_list,
    tax_ranks=('class', 'order', 'superfamily'),
    unknown_thr=500,
    norm_weights=True,
):
    num_entries = len(species_list)
    unq, unq_idx, unq_cnt = np.unique(
        species_list, return_inverse=True, return_counts=True
    )
    tax_ranks_dict = {rank: {} for rank in tax_ranks}
    teid_y = []
    for taxid, count in zip(unq, unq_cnt):
        rank_map = te_rank_one(taxid)
        teid_y.append([rank_map[rank] for rank in tax_ranks])
        for rank in tax_ranks:
            current = tax_ranks_dict[rank].get(rank_map[rank], 0)
            tax_ranks_dict[rank][rank_map[rank]] = current + count

    classes = {}
    weight_classes = {}
    for rank in tax_ranks:
        rank_counts = tax_ranks_dict[rank]
        classes_rank = dict(rank_counts)
        weights_rank = {}
        unknown = 0
        for key, value in list(rank_counts.items()):
            if value < unknown_thr:
                unknown += value
                classes_rank.pop(key, None)
            else:
                weights_rank[key] = num_entries / value
        unknown += classes_rank.get('unknown', 0)
        classes_rank['unknown'] = unknown
        weights_rank['unknown'] = num_entries / unknown if unknown else 1
        if norm_weights:
            n_classes = len(weights_rank)
            weights_rank = {key: value / n_classes for key, value in weights_rank.items()}
        classes[rank] = classes_rank
        weight_classes[rank] = weights_rank

    teid_y = np.array(teid_y)
    mapped = []
    for row in teid_y:
        mapped.append([
            row[i] if row[i] in classes[tax_ranks[i]] else 'unknown'
            for i in range(len(tax_ranks))
        ])
    mapped = np.array(mapped)[unq_idx]
    return classes, weight_classes, mapped


def read_json(path):
    with open(path, 'r') as fh:
        return json.load(fh)


def read_lines(path):
    with open(path, 'r') as fh:
        return [line.strip() for line in fh if line.strip()]


def reorder_data(x, y, y_superfamilies, fragment_names, original_fragment_names):
    name_to_index = {name: idx for idx, name in enumerate(fragment_names)}
    indices = [name_to_index[name] for name in original_fragment_names if name in name_to_index]
    return (
        np.array(x, dtype=object)[indices],
        np.array(y)[indices],
        np.array(y_superfamilies)[indices],
        np.array(fragment_names)[indices],
    )


def load_fragments_finetune(
    fragments_dir,
    file_suffix,
    classes=('class1', 'class2'),
    shuffle_=True,
    balance=False,
    nr_seqs=None,
):
    class_data = []

    for class_name in classes:
        prefix = f'{class_name}_{file_suffix}'

        fragments = read_json(os.path.join(
            fragments_dir, f'{prefix}_fragments.json'))
        superfamilies = read_lines(os.path.join(
            fragments_dir, f'{prefix}_fragment_superfamilies.txt'))
        names = read_lines(os.path.join(
            fragments_dir, f'{prefix}_fragment_headers.txt'))

        if not (len(fragments) == len(superfamilies) == len(names)):
            raise ValueError(
                f'{prefix}: fragments={len(fragments)}, '
                f'labels={len(superfamilies)}, headers={len(names)}'
            )

        class_data.append(
            (class_name, fragments, superfamilies, names)
        )

    if not class_data:
        raise ValueError('No classes specified.')

    per_class = None

    if balance:
        per_class = min(len(item[1]) for item in class_data)

        if nr_seqs is not None:
            if nr_seqs < 1:
                raise ValueError(
                    'nr_seqs must be positive when balancing.'
                )
            per_class = min(per_class, nr_seqs)

        if per_class == 0:
            raise ValueError(
                'Balanced sampling requires non-empty classes.'
            )

    rng = random.Random(42)
    x, y, y_superfamilies, fragment_names = [], [], [], []

    for class_name, fragments, superfamilies, names in class_data:
        indices = range(len(fragments))

        if balance:
            indices = rng.sample(indices, per_class)

        for index in indices:
            x.append(fragments[index])
            y.append(class_name)
            y_superfamilies.append(superfamilies[index])
            fragment_names.append(names[index])

    if shuffle_:
        original_names = read_lines(os.path.join(
            fragments_dir, f'{file_suffix}_fragment_headers.txt'))

        missing = set(fragment_names) - set(original_names)
        if missing:
            raise ValueError(
                f'{file_suffix}: fragment headers missing '
                'from the original order file.'
            )

        x, y, y_superfamilies, fragment_names = reorder_data(
            x, y, y_superfamilies, fragment_names, original_names
        )

    print(
        f'{len(x)} fragments loaded; balanced={balance}, '
        f'per_class={per_class}, original_order={shuffle_}'
    )

    return x, y, y_superfamilies, fragment_names


def prepare_multitax_sets(
    train_x,
    train_y,
    train_y_superfamilies,
    val_x,
    val_y,
    val_y_superfamilies,
    test_x,
    test_y,
    test_y_superfamilies,
    tax_ranks,
    unknown_thr=0,
    norm_weights=True,
):
    classes, weight_classes, species_list_y = get_classes_and_weights_multi_tax1(
        train_y_superfamilies,
        tax_ranks=tax_ranks,
        unknown_thr=unknown_thr,
        norm_weights=norm_weights,
    )
    print('len(train_y):', len(train_y))
    print('len(val_y):', len(val_y))
    print('len(test_y):', len(test_y))
    return (
        train_x, train_y, train_y_superfamilies,
        val_x, val_y, val_y_superfamilies,
        classes, weight_classes, species_list_y,
        test_x, test_y, test_y_superfamilies,
    )


def vote_by_sequence(y_true, y_pred, fragment_names, tax_ranks):
    sequence_scores = {rank: defaultdict(list) for rank in tax_ranks}
    sequence_labels = {rank: defaultdict(list) for rank in tax_ranks}
    for i, fragment in enumerate(fragment_names):
        base_name = re.split(r'_\d+', str(fragment))[0]
        for rank_idx, rank in enumerate(tax_ranks):
            sequence_scores[rank][base_name].append(y_pred[rank_idx][i])
            sequence_labels[rank][base_name].append(np.argmax(y_true[rank_idx][i]))

    aggregated_y_true = {rank: [] for rank in tax_ranks}
    aggregated_y_pred = {rank: [] for rank in tax_ranks}
    for rank in tax_ranks:
        for seq_name, scores in sequence_scores[rank].items():
            label_counts = defaultdict(float)
            true_counts = defaultdict(float)
            for score, true_label in zip(scores, sequence_labels[rank][seq_name]):
                label_counts[np.argmax(score)] += np.max(score)
                true_counts[true_label] += 1
            aggregated_y_pred[rank].append(max(label_counts, key=label_counts.get))
            aggregated_y_true[rank].append(max(true_counts, key=true_counts.get))
    return (
        [np.array(aggregated_y_true[rank]) for rank in tax_ranks],
        [np.array(aggregated_y_pred[rank]) for rank in tax_ranks],
    )
