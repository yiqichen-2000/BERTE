import sys
from Bio import SeqIO

def Truncate_seq(input_seq_file, max_length):
    """Truncate sequences in the fasta file to keep only the edges."""
    seq_list = []
    for seq_record in SeqIO.parse(input_seq_file, 'fasta'):
        seq_record.seq = seq_record.seq.upper()
        half_length = int(max_length / 2)
        
        if len(seq_record.seq) > max_length:
            left_part = seq_record.seq[:half_length]
            right_part = seq_record.seq[-half_length:]
            seq_record.seq = left_part + right_part
        
        seq_list.append(seq_record)

    return seq_list

def print_help():
    print("Usage: python seq_bothend_truncate.py <input_fasta_file> <max_length>")
    print("Arguments:")
    print("  <input_fasta_file>: Path to the input FASTA file.")
    print("  <max_length>: Maximum length of the truncated sequences.")
    sys.exit(1)

def main():
    if len(sys.argv) < 3 or sys.argv[1] in ['-h', '--help']:
        print_help()

    input_seq_file = sys.argv[1]
    max_length = int(sys.argv[2])
    
    
    processed_sequences = Truncate_seq(input_seq_file, max_length)
    output_file_name = f"{input_seq_file.rsplit('.', 1)[0]}_{max_length}bp.fasta"
    
    SeqIO.write(processed_sequences, output_file_name, "fasta")
    print(f"Truncated sequences written to {output_file_name}")

if __name__ == "__main__":
    main()
