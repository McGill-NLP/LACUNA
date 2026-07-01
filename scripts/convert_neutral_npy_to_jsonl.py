"""
Convert neutral OLMo pre-tokenized NPY data to JSONL for the train_ready stage.

Downloads (if needed) and converts the .npy file from the OLMo data repository into
chunked JSONL files at $HOME/data/olmo_data/. Each JSONL record contains an
"input_ids" list of length `seq_len` (default 4096).

Usage:
    python scripts/convert_neutral_npy_to_jsonl.py [--input PATH] [--output DIR] [--seq-len N] [--sequences-per-file N]

If --input is omitted the script downloads the file from the OLMo data repository.
"""

import argparse
import json
import os
import sys
import urllib.request

import numpy as np

NPY_URL = (
    "http://olmo-data.org/preprocessed/dclm/"
    "text_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/"
    "allenai/dolma2-tokenizer/part-000-00000.npy"
)


def download_npy(dest_path: str) -> str:
    """Download the NPY file if it doesn't already exist. Returns the file path."""
    if os.path.isfile(dest_path):
        print(f"NPY file already exists at {dest_path}, skipping download.")
        return dest_path
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    print(f"Downloading {NPY_URL} ...")
    urllib.request.urlretrieve(NPY_URL, dest_path)
    print(f"Saved to {dest_path}")
    return dest_path


def convert(input_path: str, output_dir: str, seq_len: int, sequences_per_file: int):
    os.makedirs(output_dir, exist_ok=True)

    # Memory-map the file so we don't load the whole array into RAM
    tokens = np.load(input_path, mmap_mode="r")
    total_tokens = tokens.shape[0]
    total_sequences = total_tokens // seq_len
    print(f"Total tokens: {total_tokens:,}  →  {total_sequences:,} sequences of length {seq_len}")

    file_idx = 0
    seq_in_file = 0
    out_path = os.path.join(output_dir, f"data_{file_idx:03d}.jsonl")
    fh = open(out_path, "w")

    for i in range(total_sequences):
        start = i * seq_len
        record = {"input_ids": tokens[start : start + seq_len].tolist()}
        fh.write(json.dumps(record) + "\n")
        seq_in_file += 1

        if seq_in_file >= sequences_per_file:
            fh.close()
            file_idx += 1
            seq_in_file = 0
            out_path = os.path.join(output_dir, f"data_{file_idx:03d}.jsonl")
            fh = open(out_path, "w")

        if (i + 1) % 10000 == 0:
            print(f"  {i + 1}/{total_sequences} sequences written …")

    fh.close()
    # Remove last file if empty
    if seq_in_file == 0 and file_idx > 0:
        os.remove(out_path)
        file_idx -= 1

    print(f"Done. Wrote {total_sequences} sequences across {file_idx + 1} file(s) in {output_dir}/")


def main():
    default_output = os.path.join(os.environ["HOME"], "data", "olmo_data")

    parser = argparse.ArgumentParser(description="Convert neutral OLMo NPY data to JSONL.")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to the .npy file. If omitted, downloads from the OLMo data repo.")
    parser.add_argument("--output", type=str, default=default_output,
                        help=f"Output directory for JSONL files (default: {default_output})")
    parser.add_argument("--seq-len", type=int, default=4096,
                        help="Sequence length for each JSONL record (default: 4096)")
    parser.add_argument("--sequences-per-file", type=int, default=50000,
                        help="Max sequences per JSONL file (default: 50000)")
    args = parser.parse_args()

    if args.input is None:
        npy_dir = os.path.join(os.environ["HOME"], "data")
        os.makedirs(npy_dir, exist_ok=True)
        args.input = download_npy(os.path.join(npy_dir, "part-000-00000.npy"))

    if not os.path.isfile(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    convert(args.input, args.output, args.seq_len, args.sequences_per_file)


if __name__ == "__main__":
    main()
