#!/usr/bin/env python3
"""Prepare FASTA for BERTE prediction; use --evaluate to extract true labels.

This test-only preprocessor writes the unified files consumed by embedding and
RF prediction steps, without class1/class2 training split files.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    seq_parts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_parts)))
                header = line[1:].strip()
                seq_parts = []
            else:
                seq_parts.append(line)
    if header is not None:
        records.append((header, "".join(seq_parts)))
    return records

def validate_sequence_ids(records: list[tuple[str, str]]) -> None:
    seen = set()

    for header, _ in records:
        sequence_id = header.split("|", 1)[0]

        if (
            not sequence_id
            or any(char.isspace() for char in sequence_id)
            or re.search(r"_\d+", sequence_id)
        ):
            raise ValueError(
                f"Unsupported sequence ID: {sequence_id!r}. "
                "Use a non-empty ID without whitespace or "
                "underscores followed by digits."
            )

        if sequence_id in seen:
            raise ValueError(f"Duplicate sequence ID: {sequence_id!r}")

        seen.add(sequence_id)

def write_fasta(records: list[tuple[str, str]], path: Path, line_width: int = 80) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for header, sequence in records:
            handle.write(f">{header}\n")
            for start in range(0, len(sequence), line_width):
                handle.write(sequence[start:start + line_width] + "\n")


def get_superfamily(header: str) -> str:
    parts = header.split("|")
    if len(parts) >= 3:
        return parts[1]
    raise ValueError(
        f"Header format error: >{header}\n"
        "Expected format like: >1112675|Bel-Pao|Dfam"
    )


def split_records(records: list[tuple[str, str]], fragment_size: int) -> list[tuple[str, str]]:
    fragmented: list[tuple[str, str]] = []
    for header, sequence in records:
        if len(sequence) <= fragment_size:
            fragmented.append((header, sequence))
            continue
        match = re.match(r"^(\d+)(\|.*)$", header)
        if match:
            base_name = match.group(1)
            rest = match.group(2)
        else:
            base_name = header
            rest = ""
        for index, start in enumerate(range(0, len(sequence), fragment_size), start=1):
            fragment = sequence[start:start + fragment_size]
            fragmented.append((f"{base_name}_{index}{rest}", fragment))
    return fragmented


def write_test_inputs(records: list[tuple[str, str]], output_dir: Path, prefix: str, evaluate: bool = False) -> Counter[str]:
    fragments = [sequence for _, sequence in records]
    headers = [header for header, _ in records]
    superfamilies = []
    counts: Counter[str] = Counter()
    if evaluate:
        for header, _ in records:
            superfamily = get_superfamily(header)
            superfamilies.append(superfamily)
            counts[superfamily] += 1

    with (output_dir / f"{prefix}_fragments.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(fragments, handle, ensure_ascii=False)
    with (output_dir / f"{prefix}_fragment_headers.txt").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(headers) + ("\n" if headers else ""))
    if evaluate:
        with (output_dir / f"{prefix}_fragment_superfamilies.txt").open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(superfamilies) + ("\n" if superfamilies else ""))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fasta", type=Path, help="Input FASTA; --evaluate requires headers like >id|superfamily|database")
    parser.add_argument("--fragment-size", "-l", type=int, default=1500)
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("."))
    parser.add_argument("--prefix", "-p", default="test")
    parser.add_argument("--evaluate", action="store_true", help="Extract true superfamily labels for evaluation.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = read_fasta(args.fasta)
    if not records:
        raise ValueError(f"No FASTA records found in {args.fasta}")
    validate_sequence_ids(records)

    fragmented = split_records(records, args.fragment_size)
    fragment_fasta = args.output_dir / f"{args.prefix}_fragments.fasta"
    write_fasta(fragmented, fragment_fasta)
    sequence_fasta = args.output_dir / f"{args.prefix}_sequences.fasta"
    write_fasta(records, sequence_fasta)

    with (args.output_dir / f"{args.prefix}_sequence_headers.txt").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(header for header, _ in records) + "\n")

    counts = write_test_inputs(fragmented, args.output_dir, args.prefix, evaluate=args.evaluate)
    summary = args.output_dir / f"{args.prefix}_preprocess_summary.tsv"
    with summary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("item\tcount\n")
        handle.write(f"input_records\t{len(records)}\n")
        handle.write(f"fragment_records\t{len(fragmented)}\n")
        for name, count in counts.most_common():
            handle.write(f"superfamily:{name}\t{count}\n")

    print(f"Input records: {len(records)}")
    print(f"Fragment records: {len(fragmented)}")
    print(f"Wrote test inputs to: {args.output_dir}")
    print(f"Fragment FASTA: {fragment_fasta}")
    print(f"Original sequence FASTA: {sequence_fasta}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
