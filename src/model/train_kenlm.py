import os
import argparse
import subprocess
import math

def train_kenlm(corpus_file, order, arpa_file, binary_file):
    """
    Trains a KenLM language model using the provided corpus file.
    
    1. Calls lmplz to generate an ARPA file with the specified n-gram order.
    2. Calls build_binary to create a binary KenLM model.
    
    Args:
        corpus_file: Path to the text corpus file.
        order: n-gram order (e.g., 3 for a 3-gram LM).
        arpa_file: Output ARPA file path.
        binary_file: Output binary model file path.
    
    Returns:
        True if both steps succeed, otherwise False.
    """
    # Construct the lmplz command
    lmplz_cmd = f"lmplz -o {order} < {corpus_file} > {arpa_file}"
    print("Running command:", lmplz_cmd)
    result = subprocess.run(lmplz_cmd, shell=True)
    if result.returncode != 0:
        print("Error running lmplz command.")
        return False

    # Construct the build_binary command
    build_binary_cmd = f"build_binary {arpa_file} {binary_file}"
    print("Running command:", build_binary_cmd)
    result = subprocess.run(build_binary_cmd, shell=True)
    if result.returncode != 0:
        print("Error running build_binary command.")
        return False

    print(f"KenLM model built successfully: {binary_file}")
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Train KenLM language model using an existing GRID corpus text file."
    )
    parser.add_argument("--corpus_file", type=str, default="grid_corpus.txt",
                        help="Path to the existing corpus file (one utterance per line).")
    parser.add_argument("--order", type=int, default=3, help="n-gram order (e.g., 3 for 3-gram).")
    parser.add_argument("--arpa_file", type=str, default="3gram.arpa",
                        help="Output ARPA file path.")
    parser.add_argument("--binary_file", type=str, default="3gram.bin",
                        help="Output binary KenLM model file path.")
    args = parser.parse_args()

    if not os.path.exists(args.corpus_file):
        print(f"Corpus file {args.corpus_file} does not exist. Exiting.")
        return

    success = train_kenlm(args.corpus_file, args.order, args.arpa_file, args.binary_file)
    if not success:
        print("KenLM training failed.")
    else:
        print("KenLM training completed successfully.")

if __name__ == "__main__":
    main()
