#!/usr/bin/env python
"""BERTE pretraining entrypoint for class1/class2 fragments."""

import argparse
import itertools
import os

os.environ.setdefault("TF_KERAS", "1")

import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.utils.all_utils import Sequence
from keras_bert import compile_model, gen_batch_inputs, get_model

from data_utils import load_fragments_split
from sequence_utils import seq_to_kmers_500



def configure_tensorflow():
    tf.config.optimizer.set_jit(True)
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("No GPU detected by TensorFlow.")
        return
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled.")
    except RuntimeError as exc:
        print(exc)


def generate_token_dict_and_list(k):
    special_tokens = ["", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    kmers = ["".join(parts) for parts in itertools.product("ACGT", repeat=k)]
    token_list = special_tokens + kmers
    token_dict = {token: index for index, token in enumerate(token_list)}
    return token_dict, token_list


class FragmentGenerator(Sequence):
    def __init__(
        self,
        fragments,
        seq_len,
        kmer_size,
        batch_size,
        token_dict,
        token_list,
    ):
        self.fragments = fragments
        self.seq_len = seq_len
        self.kmer_size = kmer_size
        self.batch_size = batch_size
        self.token_dict = token_dict
        self.token_list = token_list

    def __len__(self):
        return int(np.ceil(len(self.fragments) / float(self.batch_size)))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        batch_fragments = self.fragments[start:end]
        batch_kmers = [
            seq_to_kmers_500(
                seq,
                k=self.kmer_size,
                stride=self.kmer_size,
                pad=True,
                to_upper=True,
            )
            for seq in batch_fragments
        ]
        sentence_pairs = [
            [seq[: len(seq) // 2], seq[len(seq) // 2 :]]
            for seq in batch_kmers
        ]
        return gen_batch_inputs(
            sentence_pairs,
            self.token_dict,
            self.token_list,
            seq_len=self.seq_len,
        )

def parse_bool(value):
    value = value.strip().lower()
    if value == 'true':
        return True
    if value == 'false':
        return False
    raise argparse.ArgumentTypeError('Use true or false.')

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='pre-train BERT on pre-generated fragments',
    )
    parser.add_argument('fragments_dir')
    parser.add_argument('--nr_seqs', type=int, default=300_000, help='Maximum fragments per class when --no_balance is false; otherwise ignored.')
    parser.add_argument('--seq_len', help=' ', type=int, default=502)
    parser.add_argument('--kmer', help=' ', type=int, default=3)
    parser.add_argument('--batch_size', help=' ', type=int, default=32)
    parser.add_argument('--head_num', help=' ', type=int, default=5)
    parser.add_argument('--transformer_num', help=' ', type=int, default=6)
    parser.add_argument('--embed_dim', help=' ', type=int, default=250)
    parser.add_argument('--feed_forward_dim', help=' ', type=int, default=1024)
    parser.add_argument('--dropout_rate', help=' ', type=float, default=0.05)
    parser.add_argument('--epochs', help=' ', type=int, default=10)
    parser.add_argument(
        '--no_balance',
        type=parse_bool,
        nargs='?',
        const=True,
        default=True,
        help='Disable balancing by default. Set to false to enable balanced sampling.',
    )
    parser.add_argument('--name', help=' ', type=str, default='berte_pretrained')
    args = parser.parse_args()
    if args.nr_seqs < 1:
        parser.error('--nr_seqs must be a positive integer.')
    return args


def build_model(args, token_dict):
    model = get_model(
        token_num=len(token_dict),
        head_num=args.head_num,
        transformer_num=args.transformer_num,
        embed_dim=args.embed_dim,
        feed_forward_dim=args.feed_forward_dim,
        seq_len=args.seq_len,
        pos_num=args.seq_len,
        dropout_rate=args.dropout_rate,
    )
    compile_model(model)
    return model


def main():
    args = parse_args()
    configure_tensorflow()

    token_dict, token_list = generate_token_dict_and_list(args.kmer)
    print("token_list:\n", token_list)

    model = build_model(args, token_dict)
    model.summary()

    balance = not args.no_balance
    f_train, c_train, _ = load_fragments_split(
        args.fragments_dir,
        classes=["class1", "class2"],
        split="train",
        shuffle_=True,
        balance=balance,
        nr_seqs=args.nr_seqs,
    )
    print("len(c_train):", len(c_train))

    train_gen = FragmentGenerator(
        f_train,
        args.seq_len,
        args.kmer,
        args.batch_size,
        token_dict,
        token_list,
    )
    callbacks = [
        ModelCheckpoint(args.name + "_ep{epoch:02d}.h5"),
    ]

    model.fit(
        train_gen,
        epochs=args.epochs,
        callbacks=callbacks,
    )
    model.save(args.name + "_trained.h5")


if __name__ == "__main__":
    main()

