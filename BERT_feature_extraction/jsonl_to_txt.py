import numpy as np
import pickle
import jsonlines
import argparse

def process_features(json_file, layer_mode):
    """
    Extracts feature embeddings from a JSONL file generated by BERT.

    Args:
        json_file (str): Path to the JSONL file containing output from BERT.
        layer_mode (str): Mode of operation, dictates how layer outputs are handled.
                          Options: 'last', 'sum_all', 'concat_all', 'save_separate'
    """
    layer_cls = []
    layer_cls1, layer_cls2, layer_cls3, layer_cls4 = [], [], [], []

    with jsonlines.open(json_file) as reader:
        for obj in reader:
            if layer_mode == 'last':
                cls = obj['features'][0]['layers'][0]['values']
                layer_cls.append(cls)
            elif layer_mode == 'sum_all':
                # Sum of all layer outputs
                cls_all = np.sum([obj['features'][0]['layers'][i]['values'] for i in range(4)], axis=0).tolist()
                layer_cls.append(cls_all)
            elif layer_mode == 'concat_all':
                # Concatenation of all layer outputs
                cls_all = np.concatenate([obj['features'][0]['layers'][i]['values'] for i in range(4)], axis=0).tolist()
                layer_cls.append(cls_all)
            elif layer_mode == 'save_separate':
                # Save each layer's output separately
                cls1 = obj['features'][0]['layers'][0]['values']
                cls2 = obj['features'][0]['layers'][1]['values']
                cls3 = obj['features'][0]['layers'][2]['values']
                cls4 = obj['features'][0]['layers'][3]['values']
                layer_cls1.append(cls1)
                layer_cls2.append(cls2)
                layer_cls3.append(cls3)
                layer_cls4.append(cls4)

    if layer_mode != 'save_separate':
        # Save combined or processed embeddings
        with open(json_file + '_embedding_features.pkl', 'wb') as file:
            pickle.dump(layer_cls, file)
        np.savetxt(json_file + '_embedding_features.txt', layer_cls, fmt='%s')
        print(f'Output saved: {json_file}_embedding_features.pkl and {json_file}_embedding_features.txt')
    else:
        # Save separate layer embeddings
        layer_names = ["cls1", "cls2", "cls3", "cls4"]
        for idx, layer_data in enumerate([layer_cls1, layer_cls2, layer_cls3, layer_cls4]):
            with open(f'{json_file}_{layer_names[idx]}_embedding_features.pkl', 'wb') as file:
                pickle.dump(layer_data, file)
            np.savetxt(f'{json_file}_{layer_names[idx]}_embedding_features.txt', layer_data, fmt='%s')
            print(f'Output saved: {json_file}_{layer_names[idx]}_embedding.pkl and {json_file}_{layer_names[idx]}_embedding_features.txt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process JSONL file to extract BERT embeddings.')
    parser.add_argument('json_file', help='Path to the JSONL file containing BERT outputs')
    parser.add_argument('layer_mode', help='Mode of layer output processing (last, sum_all, concat_all, save_separate)',
                        choices=['last', 'sum_all', 'concat_all', 'save_separate'])
    args = parser.parse_args()

    print(f"Starting processing of {args.json_file} with layer mode {args.layer_mode}...")
    process_features(args.json_file, args.layer_mode)
    print("Processing complete.")
