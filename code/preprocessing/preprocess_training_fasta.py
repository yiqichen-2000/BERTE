#!/usr/bin/env python3
"""
Merge the original BERTE FASTA preprocessing pipeline into one script.

Pipeline:
1. Shuffle FASTA records.
2. Split long sequences into fixed-size fragments.
3. Extract ClassI and ClassII fragment sets for training.
4. Write model input files:
   - <prefix>_fragments.json
   - <prefix>_fragment_headers.txt
   - <prefix>_fragment_superfamilies.txt
   - class1_<prefix>_...
   - class2_<prefix>_...
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path


CLASS_I = {
    "Bel-Pao",
    "Copia",
    "Gypsy",
    "ERV",
    "DIRS",
    "PLE",
    "CR1",
    "I",
    "Jockey",
    "L1",
    "R2",
    "RTE",
    "Rex1",
    "SINE1/7SL",
    "SINE2/tRNA",
    "SINE3/5S",
    "ID",
}

CLASS_II = {
    "CACTA",
    "MULE",
    "PIF",
    "TcMar",
    "hAT",
    "Helitron",
}


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    seq_parts: list[str] = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
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
            for i in range(0, len(sequence), line_width):
                handle.write(sequence[i : i + line_width] + "\n")


def get_superfamily(header: str) -> str:
    parts = header.split("|")
    if len(parts) >= 3:
        return parts[1]
    raise ValueError(
        f"Header format error: >{header}\n"
        "Expected format like: >1112675|Bel-Pao|Dfam"
    )


def shuffle_records(
    records: list[tuple[str, str]], seed: int | None
) -> list[tuple[str, str]]:
    shuffled = records[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    return shuffled


def split_records(
    records: list[tuple[str, str]], max_length: int
) -> list[tuple[str, str]]:
    fragmented: list[tuple[str, str]] = []

    for header, sequence in records:
        if len(sequence) <= max_length:
            fragmented.append((header, sequence))
            continue

        match = re.match(r"^(\d+)(\|.*)$", header)
        if match:
            base_name = match.group(1)
            rest_of_header = match.group(2)
        else:
            base_name = header
            rest_of_header = ""

        for index, start in enumerate(range(0, len(sequence), max_length), start=1):
            fragment = sequence[start : start + max_length]
            fragment_header = f"{base_name}_{index}{rest_of_header}"
            fragmented.append((fragment_header, fragment))

    return fragmented


def split_class_records(
    records: list[tuple[str, str]]
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], Counter[str]]:
    class_i_records: list[tuple[str, str]] = []
    class_ii_records: list[tuple[str, str]] = []
    counts: Counter[str] = Counter()

    for header, sequence in records:
        superfamily = get_superfamily(header)

        counts[superfamily] += 1
        if superfamily in CLASS_I:
            class_i_records.append((header, sequence))
        elif superfamily in CLASS_II:
            class_ii_records.append((header, sequence))

    return class_i_records, class_ii_records, counts


def write_fragment_inputs(
    records: list[tuple[str, str]],
    out_prefix: Path,
) -> None:
    fragments = [sequence for _, sequence in records]
    headers = [header for header, _ in records]
    superfamilies = [get_superfamily(header) for header, _ in records]

    with (out_prefix.parent / f"{out_prefix.name}_fragments.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(fragments, handle, ensure_ascii=False)


    with (out_prefix.parent / f"{out_prefix.name}_fragment_headers.txt").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write("\n".join(headers))

    with (out_prefix.parent / f"{out_prefix.name}_fragment_superfamilies.txt").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write("\n".join(superfamilies))


def process_one_fasta(args: argparse.Namespace, fasta_path: Path, prefix: str) -> None:
    fasta_path = fasta_path.expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else fasta_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing: {fasta_path}")
    print(f"Output prefix: {prefix}")

    records = read_fasta(fasta_path)
    if not records:
        raise ValueError(f"No FASTA records found in {fasta_path}")
    validate_sequence_ids(records)

    if args.no_shuffle:
        ordered_records = records
    else:
        ordered_records = shuffle_records(records, args.seed)

    fragmented_records = split_records(
        ordered_records, args.fragment_size
    )

    class_i_records, class_ii_records, counts = split_class_records(
        fragmented_records
    )

    unsupported = sorted(
        name for name in counts
        if name not in CLASS_I and name not in CLASS_II
    )
    if unsupported:
        raise ValueError(
            "Unsupported superfamily labels for class1/class2 training split: "
            + ", ".join(unsupported)
        )

    write_fragment_inputs(
        fragmented_records,
        output_dir / prefix,
    )
    write_fragment_inputs(
        class_i_records,
        output_dir / f"class1_{prefix}",
    )
    write_fragment_inputs(
        class_ii_records,
        output_dir / f"class2_{prefix}",
    )

    print(f"Input records: {len(records)}")
    print(f"Fragment records: {len(fragmented_records)}")
    print(f"ClassI fragments: {len(class_i_records)}")
    print(f"ClassII fragments: {len(class_ii_records)}")
    print("Superfamily counts after fragmentation:")
    for name, count in counts.most_common():
        print(f"  {name}: {count}")
    print(f"Done: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert arbitrary FASTA files into BERTE model input files."
    )
    parser.add_argument(
        "fasta",
        nargs="+",
        help="Input FASTA path(s). Header format should be >id|superfamily|database.",
    )
    parser.add_argument(
        "-l",
        "--fragment-size",
        type=int,
        default=1500,
        help="Maximum fragment length. Default: 1500.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Directory for output files. Default: same directory as each input FASTA.",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        default=None,
        help="Output prefix for files such as <prefix>_fragments.json. Only allowed when processing one FASTA file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling. Default: 42.",
    )
    parser.add_argument(
        "--shuffle",
        dest="no_shuffle",
        action="store_false",
        help="Shuffle records before fragmentation. Default is no shuffle for reproducibility.",
    )
    parser.add_argument(
        "--no-shuffle",
        dest="no_shuffle",
        action="store_true",
        help="Do not shuffle records before fragmentation. This is the default.",
    )
    parser.set_defaults(no_shuffle=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fasta_paths = [Path(path).expanduser().resolve() for path in args.fasta]

    if args.prefix and len(fasta_paths) != 1:
        raise ValueError("--prefix can only be used with one input FASTA file.")

    for fasta_path in fasta_paths:
        if not fasta_path.exists():
            raise FileNotFoundError(fasta_path)
        prefix = args.prefix or fasta_path.stem
        process_one_fasta(args, fasta_path, prefix)


if __name__ == "__main__":
    main()
