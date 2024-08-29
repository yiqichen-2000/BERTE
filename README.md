# BERTE
This repository includes the implementations of BERTE from:

**BERTE：High-precision hierarchical classification of transposable elements by a transfer learning method with BERT pre-trained model and convolutional neural network**

**bioRxiv 2024**

[![Paper](https://img.shields.io/badge/Paper-PDF-blue)](<https://www.biorxiv.org/content/10.1101/2024.01.28.577612v1>)

**The code is being uploaded, it is expected that the upload will be completed by 2024.8.30 12:00 BST.**

![workflow](https://github.com/yiqichen-2000/BERTE/assets/76149916/bb7ce8a9-b3d0-4239-b9f5-c9bdce724614)

## Requirement

First extract the zip to \<your directory\>
```
cd \<your directory\>
unzip BERTE-main.zip
cd BERTE-main/
```

The usage is exemplified by the demo_SINE.fasta data \(which has been filtered for similarity\)

## Feature Extraction
This step is divided into **Feature extraction module based on BERT** and **Full-length kmer extraction**

### Feature extraction module based on BERT
```
cd ./Kmer_pre-processing  # Enter the directory to generate kmer
```
`seq_bothend_truncate.py`: Truncate sequences in a FASTA file by keeping only the both ends of the sequences

Usage: python seq_bothend_truncate.py \<input_fasta_file\> \<max_length\>

Arguments:

  - \<input_fasta_file\>: Path to the input FASTA file.
  - \<max_length\>: Maximum length of the truncated sequences.

```
python seq_bothend_truncate.py ../demo_data/demo_SINE.fasta 506  # Truncate 506bp to generate 4-mer sequence both end fragments with stride 1
python seq_bothend_truncate.py ../demo_data/demo_SINE.fasta 508  # Truncate 508bp to generate 5-mer sequence both end fragments with stride 1
python seq_bothend_truncate.py ../demo_data/demo_SINE.fasta 510  # Truncate 510bp to generate 6-mer sequence both end fragments with stride 1
# Output: demo_SINE_506bp.fasta, demo_SINE_508bp.fasta, demo_SINE_510bp.fasta
```

`seq_to_kmercount_horner.py`: Generate k-mer count using Horner's rule (in this step generates both end kmer count for BERT)

Usage: seq_to_kmercount_horner.py [-h] [--is_full_length] fasta_file k stride

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
# Output: superfamily pickle file, kmer fragment json and txt files, kmer count pickle file, full header file.
```

```
cd ../BERT_feature_extraction # 
```










