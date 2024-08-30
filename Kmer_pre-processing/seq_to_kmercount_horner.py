
import numpy as np
import os
import re
import json
import argparse
from Bio.SeqIO import parse
from collections import Counter
from itertools import product
import pickle

ALPHABET = 'ACGT'



def poly_horner(coeffi_list, x):
    """ Calculate polynomial value using Horner's method for given coefficients and base """
    degree = len(coeffi_list) - 1  # highest degree
    result = coeffi_list[0]
    for i in range(1, degree+1):
        result = result * x + coeffi_list[i]

    return result

def seq2kmers_honer(seqnumber, k=3, stride=3):
    """ Transform the numbers of sequence both end into k-mers using Horner's rule """
    if (k == 1 and stride == 1):
        # for performance reasons
        return seqnumber
    kmers = []

    half_length = int((len(seqnumber) - k + 1)/2)
    for i in range(0, half_length-1, stride):
        kmer = seqnumber[i:i+k]
        kmer = list("".join(kmer))
        kmer = list(map(int,kmer))
        count = poly_horner(kmer, 4)
        kmers.append(count)

    for j in range(half_length+1, len(seqnumber) - k + 1, stride):
        kmer = seqnumber[j:j+k]
        kmer = list("".join(kmer))
        kmer = list(map(int,kmer))
        count = poly_horner(kmer, 4)
        kmers.append(count)

    return kmers

def seq2kmers_all_honer(seqnumber, k=3, stride=1):
    """ Transform full length sequence numbers into k-mers using Horner's rule """
    if (k == 1 and stride == 1):
        # for performance reasons
        return seqnumber
    kmers = []

    for i in range(0, len(seqnumber), stride):
        kmer = seqnumber[i:i+k]
        if len(kmer) > k-1:
            # print(kmer)
            kmer = list("".join(kmer))
            kmer = list(map(int,kmer))
            count = poly_horner(kmer, 4)
            kmers.append(count)
        else:
            break

    return kmers

def number_to_sequence(number,k):
    """ Convert number to nucleotide sequence based on given k-mer length """
    bases = ['A', 'C', 'G', 'T']
    sequence = ''
    while number > 0:
        sequence = bases[number % 4] + sequence
        number //= 4
    while len(sequence) < k:
        sequence = 'A' + sequence
    return sequence

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process sequences to generate k-mer frequencies using Horner's rule.")
    parser.add_argument('fasta_file', help='Input FASTA file containing sequences')
    parser.add_argument('k', type=int, help='Length of k-mers to generate')
    parser.add_argument('stride', type=int, help='Stride for k-mer generation')
    parser.add_argument('--is_full_length', action='store_true', help='Generate full-length k-mers')


    args = parser.parse_args()
    knum = int(args.k)
    stride = int(args.stride)
    is_full_length = args.is_full_length
    records = list(parse(args.fasta_file, 'fasta'))
    print(f"Processing {len(records)} records from {args.fasta_file}...")
    output_prefix = os.path.splitext(args.fasta_file)[0]
    fragments = [str(r.seq) for r in records]
    headers_list = [r.description for r in records]
    superfamliy_list = []
    for r in records:
        superfamliy = re.search('.+?\|(.+?)\|.+?$', r.description).group(1)
        superfamliy_list.append(superfamliy)

    all_kmer = [''.join(_) for _ in product(ALPHABET, repeat=knum)]
    all_kmer = list(all_kmer)

    if is_full_length:
        if os.path.exists(f'{output_prefix}_full_length_kmer_fragments.json'):
            os.remove(f'{output_prefix}_full_length_kmer_fragments.json')
        seq_each = []
        all_kmers_stat_by_ACGT_list = []

        with open(f'{output_prefix}_full_length_kmer_fragments.txt', "w+") as w:
            print("Generating full length k-mer sequences...")
            for i in fragments: 
                # print("i:\n",i)    
                count=0
                converted_string = ''.join(['0' if nucleotide == 'A' else
                                '1' if nucleotide == 'C' else
                                '2' if nucleotide == 'G' else
                                '3' if nucleotide == 'T' else
                                'X'
                                for nucleotide in i])

                converted_string = converted_string.replace("X", "")
                decimal_numbers = seq2kmers_all_honer(converted_string, k=knum, stride=stride)
                kmer_sequences = [number_to_sequence(number,k=knum) for number in decimal_numbers]
                for j in kmer_sequences:
                    w.write(j + ' ')
                w.write('\n')
                
                json.dump(kmer_sequences, open(f'{output_prefix}_full_length_kmer_fragments.json', 'a+'))
                counter = Counter(decimal_numbers)
                keys = range(4**knum)
                sorted_dict = {key: counter[key] if counter[key] else 0 for key in sorted(keys)}
                kmers_stat_by_ACGT_list = list(sorted_dict.values())
                all_kmers_stat_by_ACGT_list.append(kmers_stat_by_ACGT_list)

        with open(f'{output_prefix}_full_length_kmer_counts_list.pkl', 'wb') as file:
            pickle.dump(all_kmers_stat_by_ACGT_list, file)
        print("Saving superfamilies and full headers...")
        with open(f'{output_prefix}_superfamliy.pkl', 'wb') as f:
            pickle.dump(superfamliy_list, f)
        with open(f'{output_prefix}_header_full.txt', 'w') as f:
            f.write('\n'.join(headers_list))

    else:
        if os.path.exists(f'{output_prefix}_bothend_kmer_fragments.json'):
            os.remove(f'{output_prefix}_bothend_kmer_fragments.json')
        seq_each = []
        all_kmers_stat_by_ACGT_list = []

        with open(f'{output_prefix}_bothend_kmer_fragments.txt', "w+") as w:
            print("Generating bothend k-mer sequences...")
            for i in fragments: 
                # print("i:\n",i)    
                count=0
                converted_string = ''.join(['0' if nucleotide == 'A' else
                                '1' if nucleotide == 'C' else
                                '2' if nucleotide == 'G' else
                                '3' if nucleotide == 'T' else
                                'X'
                                for nucleotide in i])

                converted_string = converted_string.replace("X", "")
                decimal_numbers = seq2kmers_honer(converted_string, k=knum, stride=stride)
                kmer_sequences = [number_to_sequence(number,k=knum) for number in decimal_numbers]
                for j in kmer_sequences:
                    w.write(j + ' ')
                    count += 1
                    # Inserting '|||' at the 250th tokens to indicate sentence separation for BERT model training/inference
                    if count == 250:
                        w.write('|||' + ' ')
                w.write('\n')
                
                json.dump(kmer_sequences, open(f'{output_prefix}_bothend_kmer_fragments.json', 'a+'))
                # a: ['AGG', 'GGG', 'GGG', 'GGN', 'GNX']
                counter = Counter(decimal_numbers)
                keys = range(4**knum)
                sorted_dict = {key: counter[key] if counter[key] else 0 for key in sorted(keys)}
                kmers_stat_by_ACGT_list = list(sorted_dict.values())
                all_kmers_stat_by_ACGT_list.append(kmers_stat_by_ACGT_list)

        with open(f'{output_prefix}_bothend_kmer_counts_list.pkl', 'wb') as file:
            pickle.dump(all_kmers_stat_by_ACGT_list, file)
        print("Saving superfamilies and full headers...")
        with open(f'{output_prefix}_superfamliy.pkl', 'wb') as f:
            pickle.dump(superfamliy_list, f)
        with open(f'{output_prefix}_header_full.txt', 'w') as f:
            f.write('\n'.join(headers_list))
        





