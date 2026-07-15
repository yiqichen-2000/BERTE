#!/usr/bin/env python
"""BERTE multi-tax fine-tuning entrypoint."""

import argparse
import os
import pickle
from logging import warning

from sklearn.metrics import balanced_accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

TAX_RANKS = ['class', 'order', 'superfamily']

from data_utils import (
    load_fragments_finetune,
    prepare_multitax_sets,
    vote_by_sequence,
)
from model_utils import (
    FragmentGeneratorMultiTax,
    SaveEmbeddingOutput,
    configure_tensorflow,
    get_fine_model_multi_tax,
    load_bert,
    predict,
)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='direct class/order/superfamily fine-tuning on pre-generated fragments',
    )
    parser.add_argument('pretrained_bert')
    parser.add_argument('fragments_dir')
    parser.add_argument('--seq_len', help=' ', type=int, default=502)
    parser.add_argument('--k', help=' ', default=3, type=int)
    parser.add_argument('--stride', help=' ', default=3, type=int)
    parser.add_argument('--batch_size', help=' ', type=int, default=32)
    parser.add_argument('--epochs', help=' ', type=int, default=4)
    parser.add_argument('--nr_seqs', help=' ', type=int, default=100_000_000)
    parser.add_argument('--learning_rate', help=' ', type=float, default=5e-5)
    parser.add_argument('--save_name', help='custom name for saved finetuned model', default=None)
    parser.add_argument('--store_predictions', help=' ', action='store_true')
    parser.add_argument('--only_test_model', help="don't train a model but evaluate the performance", action='store_true')
    parser.add_argument('--export_embeddings_only', help="don't train or evaluate; load the model and export train/val/test embeddings", action='store_true')
    return parser.parse_args()


def make_generators(args, max_length, classes, train_data, val_data, test_data):
    common = {
        'max_seq_len': max_length,
        'k': args.k,
        'stride': args.stride,
        'batch_size': args.batch_size,
        'window': False,
    }
    train_g = FragmentGeneratorMultiTax(
        train_data[0], train_data[1], train_data[2],
        classes=classes, seq_len=args.seq_len, tax_ranks=args.tax_ranks, **common)
    val_g = FragmentGeneratorMultiTax(
        val_data[0], val_data[1], val_data[2],
        classes=classes, seq_len=args.seq_len, tax_ranks=args.tax_ranks, **common)
    test_g = FragmentGeneratorMultiTax(
        test_data[0], test_data[1], test_data[2],
        classes=classes, seq_len=args.seq_len, tax_ranks=args.tax_ranks, **common)
    return train_g, val_g, test_g


def summarize_prediction_accuracy(predicted, test_g, test_fragment_names):
    y_true, y_pred = predicted['data']
    y_true_voted, y_pred_voted = vote_by_sequence(
        y_true, y_pred, test_fragment_names, test_g.tax_ranks)

    for i, rank in enumerate(test_g.tax_ranks):
        y_true_level = y_true_voted[i]
        y_pred_level = y_pred_voted[i]
        acc = balanced_accuracy_score(y_true_level, y_pred_level)
        print(f'{rank} Balanced Accuracy:', acc)


def export_embeddings(model, args, train_g, val_g, test_g, save_prefix):
    for data_name, generator in [('train', train_g), ('val', val_g), ('test', test_g)]:
        SaveEmbeddingOutput(
            save_prefix,
            model,
            generator,
            data_name=data_name,
            output_dir=args.fragments_dir,
            k=args.k,
        ).on_train_end()


def main():
    args = parse_args()
    args.tax_ranks = TAX_RANKS
    configure_tensorflow()

    if args.only_test_model and args.export_embeddings_only:
        raise ValueError('--only_test_model and --export_embeddings_only are mutually exclusive.')

    train_x, train_y, train_sf, train_names = load_fragments_finetune(
        args.fragments_dir, 'train', shuffle_=True, balance=False, nr_seqs=args.nr_seqs)
    val_x, val_y, val_sf, val_names = load_fragments_finetune(
        args.fragments_dir, 'val', shuffle_=True, balance=False, nr_seqs=args.nr_seqs)
    test_x, test_y, test_sf, test_names = load_fragments_finetune(
        args.fragments_dir, 'test', shuffle_=True, balance=False, nr_seqs=args.nr_seqs)

    (
        f_train_x, f_train_y, f_train_sf,
        f_val_x, f_val_y, f_val_sf,
        classes, weight_classes, species_list_y,
        f_test_x, f_test_y, f_test_sf,
    ) = prepare_multitax_sets(
        train_x, train_y, train_sf,
        val_x, val_y, val_sf,
        test_x, test_y, test_sf,
        tax_ranks=args.tax_ranks,
        unknown_thr=0,
        norm_weights=True,
    )
    print('classes:\n', classes)

    if args.only_test_model or args.export_embeddings_only:
        model = load_bert(args.pretrained_bert, compile_=args.only_test_model)
        max_length = model.input_shape[0][1]
    else:
        num_classes = [len(classes[tax].keys()) for tax in args.tax_ranks]
        model, max_length = get_fine_model_multi_tax(
            args.pretrained_bert,
            num_classes=num_classes,
            tax_ranks=args.tax_ranks,
            learning_rate=args.learning_rate,
        )

    if args.seq_len > max_length:
        warning(f'desired seq len ({args.seq_len}) is higher than possible ({max_length}); setting seq len to {max_length}')
        args.seq_len = max_length

    train_g, val_g, test_g = make_generators(
        args,
        max_length,
        classes,
        (f_train_x, f_train_y, f_train_sf),
        (f_val_x, f_val_y, f_val_sf),
        (f_test_x, f_test_y, f_test_sf),
    )
    model.summary()

    save_prefix = args.save_name if args.save_name is not None else os.path.splitext(args.pretrained_bert)[0]
    if args.export_embeddings_only:
        export_embeddings(model, args, train_g, val_g, test_g, save_prefix)
        return

    if not args.only_test_model:
        callbacks = [
            ModelCheckpoint(save_prefix + '_model.best.acc.h5', monitor='val_superfamily_out_accuracy', verbose=1, save_best_only=True, mode='max'),
            ModelCheckpoint(save_prefix + '_finetune_ep{epoch:02d}.h5', verbose=1),
            ModelCheckpoint(save_prefix + '_model.best.loss.h5', monitor='loss', verbose=1, save_best_only=True, mode='min'),
            EarlyStopping('superfamily_out_loss', min_delta=0.01, patience=2, restore_best_weights=True),
            SaveEmbeddingOutput(save_prefix, model, train_g, data_name='train', output_dir=args.fragments_dir, k=args.k),
            SaveEmbeddingOutput(save_prefix, model, val_g, data_name='val', output_dir=args.fragments_dir, k=args.k),
            SaveEmbeddingOutput(save_prefix, model, test_g, data_name='test', output_dir=args.fragments_dir, k=args.k),
        ]
        try:
            model.fit(train_g, callbacks=callbacks, epochs=args.epochs, validation_data=val_g, verbose=1)
        except KeyboardInterrupt:
            print('training interrupted, saving aborted model and testing current status')
            model.save(save_prefix + '_aborted.h5')
        save_path = save_prefix + '.h5' if args.save_name is not None else save_prefix + '_finetuned.h5'
        model.save(save_path)
        print('testing...')
    else:
        save_path = args.pretrained_bert

    if args.store_predictions:
        predicted = predict(
            model,
            test_g,
            return_data=True,
            store_x=False,
        )
        summarize_prediction_accuracy(predicted, test_g, test_names)
        out_pkl = os.path.splitext(save_path)[0] + '_vote_predictions.pkl'
        with open(out_pkl, 'wb') as fh:
            pickle.dump(predicted, fh)
        print(f'Predictions saved to {out_pkl}')
        result = predicted['metrics']
        metrics_names = predicted['metrics_names']
    else:
        result = model.evaluate(test_g)
        metrics_names = model.metrics_names
    print('test results:', *zip(metrics_names, result))


if __name__ == '__main__':
    main()


