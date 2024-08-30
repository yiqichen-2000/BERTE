# BERTE
This repository includes the implementations of BERTE from:

**BERTE：High-precision hierarchical classification of transposable elements by a transfer learning method with BERT pre-trained model and convolutional neural network**

**bioRxiv 2024**

[![Paper](https://img.shields.io/badge/Paper-PDF-blue)](<https://www.biorxiv.org/content/10.1101/2024.01.28.577612v1>)

**The code is being uploaded, it is expected that the upload will be completed by 2024.8.30 12:00 BST.**

![workflow](https://github.com/yiqichen-2000/BERTE/assets/76149916/bb7ce8a9-b3d0-4239-b9f5-c9bdce724614)

## Requirement
BERTE runs with conda

h5py == 3.1.0

tensorflow == 2.6.0
tensorflow-gpu == 2.6.0
python == 3.8.1

cudnn == 8.9.2.26, cudatoolkit == 11.8.0


## Example Feature Extraction
First extract the zip to \<your directory\>
```
cd \<your directory\>
unzip BERTE-main.zip
cd BERTE-main/
```

The usage is exemplified by the demo_SINE.fasta data \(which has been filtered for similarity\)

This step is divided into **Feature extraction module based on BERT** and **Full-length k-mer extraction**

### Feature extraction module based on BERT
```
cd ./Kmer_pre-processing  # Enter the directory to generate k-mer
```
`seq_bothend_truncate.py`: Truncate sequences in a FASTA file by keeping only the both ends of the sequences

Usage: python seq_bothend_truncate.py \<input_fasta_file\> \<max_length\>

Output: The fasta file after truncation, with filename adding \<max_length\>. E.g. demo_SINE_506bp.fasta, demo_SINE_508bp.fasta, demo_SINE_510bp.fasta

Arguments:

  - \<input_fasta_file\>: Path to the input FASTA file.
  - \<max_length\>: Maximum length of the truncated sequences.

```
python seq_bothend_truncate.py ../demo_data/demo_SINE.fasta 506  # Truncate 506bp to generate 4-mer sequence both end fragments with stride 1
python seq_bothend_truncate.py ../demo_data/demo_SINE.fasta 508  # Truncate 508bp to generate 5-mer sequence both end fragments with stride 1
python seq_bothend_truncate.py ../demo_data/demo_SINE.fasta 510  # Truncate 510bp to generate 6-mer sequence both end fragments with stride 1

```

`seq_to_kmercount_horner.py`: Generate k-mer count using Horner's rule (in this step generates both end k-mer count for BERT)

Usage: seq_to_kmercount_horner.py [-h] [--is_full_length] fasta_file k stride

Output: superfamily pickle file, k-mer fragment json and txt files, k-mer count pickle file, full header file.

positional arguments:
  - \<fasta_file\>: Input FASTA file containing sequences
  - \<k\>: Length of k-mers to generate
  - \<stride\>: Stride for k-mer generation

options:
  - \-h, --help: Show this help message and exit
  - \-\-is_full_length: Generate full-length k-mers

```
python seq_to_kmercount_horner.py ../demo_data/demo_SINE_506bp.fasta 4 1 # Generate 4-mer count for both end of the sequences using Horner's rule, with stride 1
python seq_to_kmercount_horner.py ../demo_data/demo_SINE_508bp.fasta 5 1 # Generate 5-mer count for both end of the sequences using Horner's rule, with stride 1
python seq_to_kmercount_horner.py ../demo_data/demo_SINE_510bp.fasta 6 1 # Generate 6-mer count for both end of the sequences using Horner's rule, with stride 1

```

```
cd ../BERT_feature_extraction # Enter the directory to generate BERT \[CLS\] token embedding
```

`extract_features_tf2_cls.py`: Google's official code for generating token embedding from pre-trained BERT models, adapted for tensorflow 2.x.

Output: \[CLS\] token embedding jsonl file

```
CUDA_VISIBLE_DEVICES=0 python extract_features_tf2_cls.py \
    --input_file=../demo_data/demo_SINE_506bp_bothend_kmer_fragments.txt \
    --output_file=./demo_SINE_506bp_bothend_kmer_fragments_cls.jsonl \
    --vocab_file=./BERTMini/kmer_vocab.txt \
    --bert_config_file=./BERTMini/bert_config.json \
    --init_checkpoint=./BERTMini/bert_model.ckpt.index \
    --do_lower_case=False \
    --layers=-1 \
    --max_seq_length=503 \
    --batch_size=16

CUDA_VISIBLE_DEVICES=0 python extract_features_tf2_cls.py \
    --input_file=../demo_data/demo_SINE_508bp_bothend_kmer_fragments.txt \
    --output_file=./demo_SINE_508bp_bothend_kmer_fragments_cls.jsonl \
    --vocab_file=./BERTMini/kmer_vocab.txt \
    --bert_config_file=./BERTMini/bert_config.json \
    --init_checkpoint=./BERTMini/bert_model.ckpt.index \
    --do_lower_case=False \
    --layers=-1 \
    --max_seq_length=503 \
    --batch_size=16

CUDA_VISIBLE_DEVICES=0 python extract_features_tf2_cls.py \
    --input_file=../demo_data/demo_SINE_510bp_bothend_kmer_fragments.txt \
    --output_file=./demo_SINE_510bp_bothend_kmer_fragments_cls.jsonl \
    --vocab_file=./BERTMini/kmer_vocab.txt \
    --bert_config_file=./BERTMini/bert_config.json \
    --init_checkpoint=./BERTMini/bert_model.ckpt.index \
    --do_lower_case=False \
    --layers=-1 \
    --max_seq_length=503 \
    --batch_size=16
```

`jsonl_to_txt.py`: Process JSONL file to extract BERT embeddings in pickle.

Usage: seq_to_kmercount_horner.py [-h] [--is_full_length] fasta_file k stride

Output: BERT embeddings pickle file

Positional arguments:
  - \<json_file\>: Path to the JSONL file containing BERT outputs
  - \{last,sum_all,concat_all,save_separate\}: Mode of layer output processing \(last, sum_all, concat_all, save_separate\)

```
python jsonl_to_txt.py demo_SINE_506bp_bothend_kmer_fragments_cls.jsonl last
python jsonl_to_txt.py demo_SINE_508bp_bothend_kmer_fragments_cls.jsonl last
python jsonl_to_txt.py demo_SINE_510bp_bothend_kmer_fragments_cls.jsonl last
```

```
mv *_cls_embedding_features.txt ../working_files/
# Move demo_SINE's 4-mer, 5-mer, and 6-mer transformed BERT features, to be used in training
```

### Full-length k-mer extraction
```
cd ../Kmer_pre-processing  # Enter the directory to generate k-mer (generating full-length k-mer in this step)
```

```
python seq_to_kmercount_horner.py ../demo_data/demo_SINE_506bp.fasta 4 1 --is_full_length # Generate full-length 4-mer sequence with stride 1
python seq_to_kmercount_horner.py ../demo_data/demo_SINE_508bp.fasta 5 1 --is_full_length # Generate full-length 5-mer sequence with stride 1
python seq_to_kmercount_horner.py ../demo_data/demo_SINE_510bp.fasta 6 1 --is_full_length # Generate full-length 6-mer sequence with stride 1
```

```
mv *_full_length_kmer_counts_list.pkl ../working_files/
# Move demo_SINE's full length 4-mer, 5-mer, and 6-mer features, to be used in training

mv *_superfamliy_.pkl ../working_files/
# Move demo_SINE's superfamily ids, to be used as labels in training
```

## Example Training
```
cd ../Train  # Enter the directory to train models
```

`BERTE_train.py`: Train a CNN model to classify DNA sequences using BERT embeddings and k-mer counts.

Usage: BERTE_train.py [-h] rank epoch batchsize

Output: superfamily pickle file, k-mer fragment json and txt files, k-mer count pickle file, full header file.

positional arguments:
  - \<path\>: Path to the directory containing .pkl files.
  - \<rank\>: The selected rank name (i.e. parent node).
  - \<epoch\>: Number of epochs for training the model.
  - \<batchsize\>: Batch size for training the model.

```
CUDA_VISIBLE_DEVICES=1 python BERTE_train.py ../working_files/ SINE 50 64
```
