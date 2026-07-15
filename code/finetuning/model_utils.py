"""Minimal model, generator, prediction, and callback helpers."""

import os
import pickle
from logging import warning

import keras_bert
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import Sequence

from data_utils import get_te_rank
from sequence_utils import get_class_vectors, get_token_dict, seq2tokens


def configure_tensorflow():
    tf.config.optimizer.set_jit(True)
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print('No GPU detected by TensorFlow.')
        return
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print('GPU memory growth enabled.')
    except RuntimeError as exc:
        print(exc)


def load_bert(pretrained_path, compile_=False):
    custom_objects = {
        'GlorotNormal': keras.initializers.glorot_normal,
        'GlorotUniform': keras.initializers.glorot_uniform,
    }
    custom_objects.update(keras_bert.get_custom_objects())
    return keras.models.load_model(
        pretrained_path,
        compile=compile_,
        custom_objects=custom_objects,
    )


def generate_bert_with_pretrained_multi_tax(
    pretrained_path,
    nr_classes,
    tax_ranks,
):
    model = load_bert(pretrained_path, compile_=False)
    inputs = model.inputs[:2]
    nsp_dense_layer = model.get_layer(name='NSP-Dense').output
    print('nsp_dense_layer:', nsp_dense_layer)

    tax_i_in = nsp_dense_layer
    out_layers = []
    for index, nr_classes_tax_i in enumerate(nr_classes):
        tax_i_out = Dense(
            nr_classes_tax_i,
            name=f'{tax_ranks[index]}_out',
            activation='softmax',
        )(tax_i_in)
        out_layers.append(tax_i_out)
        tax_i_in = tf.keras.layers.concatenate(out_layers + [nsp_dense_layer])

    model_fine = tf.keras.Model(inputs=inputs, outputs=out_layers)
    print('inputs:', inputs)
    print('outputs:', out_layers)
    return model_fine


def get_fine_model_multi_tax(pretrained_model_file, num_classes, tax_ranks, learning_rate):
    model_fine = generate_bert_with_pretrained_multi_tax(
        pretrained_model_file,
        nr_classes=num_classes,
        tax_ranks=tax_ranks,
    )
    model_fine.compile(
        keras.optimizers.Adam(learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    max_length = model_fine.input_shape[0][1]
    print('max_length:', max_length)
    return model_fine, max_length


class FragmentGeneratorMultiTax(Sequence):
    def __init__(
        self,
        x,
        y,
        y_superfamilies,
        classes,
        seq_len,
        tax_ranks,
        max_seq_len=None,
        k=3,
        stride=3,
        batch_size=16,
        seq_len_like=None,
        window=False,
    ):
        self.x = x
        self.y = y
        self.y_superfamilies = y_superfamilies
        self.classes = classes
        self.seq_len = seq_len
        self.tax_ranks = tax_ranks
        self.max_seq_len = max_seq_len or seq_len
        self.k = k
        self.stride = stride
        self.batch_size = batch_size
        self.seq_len_like = seq_len_like
        self.window = window
        self.class_vectors = {
            rank: get_class_vectors(self.classes[rank]) for rank in self.classes
        }
        self.token_dict = get_token_dict(k=k)
        print(self.token_dict)

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def get_class_vectors_multi_tax(self, taxid):
        vector = []
        taxid_list = [str(taxid)]
        ranks, _ = get_te_rank(taxid_list, all=False)
        for index, _rank_name in enumerate(ranks):
            tax_rank = self.tax_ranks[index]
            vector.append(self.class_vectors[tax_rank].get(
                ranks[_rank_name],
                self.class_vectors[tax_rank]['unknown'],
            ))
        return vector

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        batch_fragments = self.x[start:end]
        batch_x = [
            seq2tokens(
                seq,
                self.token_dict,
                seq_length=self.seq_len,
                max_length=self.max_seq_len,
                k=self.k,
                stride=self.stride,
                window=self.window,
                seq_len_like=self.seq_len_like,
            )
            for seq in batch_fragments
        ]
        x = [np.array([item[0] for item in batch_x]),
             np.array([item[1] for item in batch_x])]
        if self.y is None or len(self.y) == 0:
            return x
        batch_classes = self.y_superfamilies[start:end]
        batch_y = [self.get_class_vectors_multi_tax(taxid) for taxid in batch_classes]
        y = []
        for index in range(len(self.tax_ranks)):
            y.append(np.array([item[index] for item in batch_y]))
        return x, y


class PredictGenerator(Sequence):
    def __init__(self, generator, store_x=False):
        self.g = generator
        self.store_x = store_x
        self.targets = []
        self.x = []

    def __len__(self):
        return len(self.g)

    def __getitem__(self, idx):
        batch = self.g[idx]
        if not isinstance(batch[0], np.ndarray):
            x = batch[0]
            targets = batch[1] if isinstance(batch[1], np.ndarray) else batch[1][:]
            self.targets.append((idx, targets))
        else:
            x = batch
        if self.store_x:
            self.x.append((idx, x))
        return x

    def _get_stored(self, stored):
        stored = sorted(stored, key=lambda x: x[0])
        stored = [[s for s in stored if s[0] == i][-1]
                  for i in range(stored[-1][0] + 1)]
        if not all(t[0] == i for i, t in enumerate(stored)):
            warning('something probably went wrong storing prediction values')
        return [stored[i][1] for i in range(stored[-1][0] + 1)]

    def get_targets(self):
        if len(self.targets) == 0:
            return []
        try:
            return np.concatenate(self._get_stored(self.targets))
        except Exception:
            stored_targets = self._get_stored(self.targets)
            return [
                np.concatenate([
                    np.array(stored_targets[i][j])
                    for i in range(len(stored_targets))
                ])
                for j in range(len(stored_targets[0]))
            ]

    def get_x(self):
        if not self.store_x:
            warning('option to store x values was not set')
            return None
        return self._get_stored(self.x)


def predict(model, test_generator, return_data=False, store_x=False, nonverbose=False):
    predict_g = PredictGenerator(test_generator, store_x=store_x)
    print('predict_g:', predict_g)
    preds = model.predict(predict_g, verbose=0 if nonverbose else 1)
    targets = predict_g.get_targets()
    if len(targets) > 0 and len(targets[0].shape) >= 2:
        y = [np.array(target[:len(preds[0])]) for target in targets]
    elif len(targets) > 0:
        y = targets[:len(preds)]
    else:
        y = []
    return {
        'metrics': [],
        'metrics_names': [],
        'data': (y, preds) if return_data else None,
        'x': predict_g.get_x() if return_data and store_x else None,
    }


class SaveEmbeddingOutput(Callback):
    def __init__(self, save_prefix, model, test_data, data_name, output_dir=None, k=None):
        super().__init__()
        self.save_prefix = save_prefix
        self.data_name = data_name
        self.output_dir = output_dir
        self.k = k
        self.intermediate_model = tf.keras.Model(
            inputs=model.input,
            outputs=model.get_layer('NSP-Dense').output,
        )
        self.test_data = test_data

    def on_train_end(self, logs=None):
        embedding_output = self.intermediate_model.predict(self.test_data)
        if self.output_dir is not None and self.k is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            filename = os.path.join(self.output_dir, f'{self.data_name}_{self.k}mer_embeddings.pkl')
        else:
            filename = f'{self.save_prefix}_{self.data_name}_embeddings.pkl'
        with open(filename, 'wb') as fh:
            pickle.dump(embedding_output, fh)
        print(f'NSP-Dense embeddings saved to {filename}')


SaveNSPDenseOutput = SaveEmbeddingOutput
