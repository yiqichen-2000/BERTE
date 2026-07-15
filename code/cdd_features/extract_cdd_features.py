#!/usr/bin/env python3
"""Run rpstblastn and write one protein-feature row per input FASTA record.

This script intentionally differs from the historical parser in one important
way: output2 is initialized from the query FASTA, not from BLAST comment blocks.
Therefore sequences with no rpstblastn hit still get an empty feature row, which
keeps protein features aligned with BERTE sequence-level NSP features.
"""
from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path


def read_query_headers(fasta_file: Path) -> list[str]:
    headers: list[str] = []
    with fasta_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                headers.append(line[1:].strip())
    if not headers:
        raise ValueError(f"No FASTA records found in {fasta_file}")
    return headers


def query_aliases(header: str) -> set[str]:
    aliases = {header, header.split()[0]}
    # BLAST qseqid can sometimes expose only the first pipe-delimited token.
    aliases.add(header.split("|", 1)[0])
    return aliases


def load_protein_names(db_list: Path) -> list[str]:
    with db_list.open("r", encoding="utf-8") as handle:
        return [line.split(".")[0].lower() for line in handle if line.strip()]


def output_name(fasta_file: Path, suffix: str, add_timestamp: bool) -> str:
    suffix_path = Path(suffix)
    if suffix_path.parent != Path("."):
        suffix_path.parent.mkdir(parents=True, exist_ok=True)
        return str(suffix_path)
    if add_timestamp:
        return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix}"
    prefix = fasta_file.stem if fasta_file.suffix else fasta_file.name
    return f"{prefix}_{suffix}"


def run_rpstblastn(fasta_file: Path, db_list: Path, output_file: Path, threads: str) -> None:
    db_prefix = str(db_list)[:-3]
    command = [
        "rpstblastn",
        "-query", str(fasta_file),
        "-db", db_prefix,
        "-outfmt", "6 qseqid sacc stitle qframe evalue bitscore qstart qend qlen sstart send",
        "-evalue", "0.1",
        "-out", str(output_file),
        "-num_threads", str(threads),
    ]
    print(" ".join(command))
    subprocess.run(command, check=True)


def write_protein_features(blast_output: Path, feature_output: Path, query_headers: list[str], proteins: list[str]) -> None:
    protein_to_index = {protein: idx for idx, protein in enumerate(proteins)}
    header_by_alias = {}
    for header in query_headers:
        for alias in query_aliases(header):
            header_by_alias[alias] = header

    features = {header: [""] * len(proteins) for header in query_headers}
    unmapped_queries = set()
    with blast_output.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            qseqid = parts[0]
            subject_title = parts[2].lower()
            e_value = parts[4]
            header = header_by_alias.get(qseqid)
            if header is None:
                unmapped_queries.add(qseqid)
                continue
            protein_key = subject_title.split(",")[0]
            protein_idx = protein_to_index.get(protein_key)
            if protein_idx is not None:
                features[header][protein_idx] += "," + e_value

    if unmapped_queries:
        preview = ", ".join(sorted(unmapped_queries)[:10])
        print(f"Warning: {len(unmapped_queries)} BLAST qseqid values did not match FASTA headers; first: {preview}")

    with feature_output.open("w", encoding="utf-8", newline="\n") as out_handle:
        for header in query_headers:
            out_handle.write(header.replace("@", "") + "@" + "@".join(features[header]) + "\n")
    print(f"Protein feature rows written: {len(query_headers)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RPSTBLASTN and extract CDD protein-domain features.")
    parser.add_argument("--fastaFile", "-i", required=True, type=Path, help="Input FASTA query file")
    parser.add_argument("--db", "-d", default="CDD/RPSTBLASTN_LIB/db_large.pn", type=Path, help="Protein DB .pn list")
    parser.add_argument("--threads", "-t", default="1", help="rpstblastn thread count")
    parser.add_argument("--out1", default="out1", help="RPSTBLASTN raw output suffix or output path")
    parser.add_argument("--out2", default="out2", help="CDD feature output suffix or output path")
    parser.add_argument("--timestamp", action="store_true", help="Prefix outputs with timestamp for compatibility")
    args = parser.parse_args()

    query_headers = read_query_headers(args.fastaFile)
    proteins = load_protein_names(args.db)
    out1 = Path(output_name(args.fastaFile, args.out1, args.timestamp))
    out2 = Path(output_name(args.fastaFile, args.out2, args.timestamp))
    run_rpstblastn(args.fastaFile, args.db, out1, str(args.threads))
    write_protein_features(out1, out2, query_headers, proteins)
    print(f"Done. Outputs: {out1}, {out2}")


if __name__ == "__main__":
    main()
