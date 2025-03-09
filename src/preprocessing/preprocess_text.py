import os
import glob

def process_align_file(align_path):
    """
    Reads a single .align file, extracts the tokens (3rd column) 
    while skipping 'sil' (or 'sp' if needed), and returns the utterance
    as a single space-separated string.
    """
    words = []
    with open(align_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                token = parts[2]
                if token.lower() not in ["sil", "sp"]:
                    words.append(token)
    # Return the utterance as a single line
    return " ".join(words)

def gather_alignments(data_dir, output_file):
    """
    Iterates over all speaker directories (s1, s2, …) in data_dir,
    reads all .align files from the 'align' subfolder, and writes each
    processed utterance to a new line in output_file.
    """
    # Find all alignment directories under data/raw that match s*
    align_dirs = glob.glob(os.path.join(data_dir, "s*", "alignments"))
    all_utterances = []

    for align_dir in align_dirs:
        # For each .align file in the directory
        print(f"Processing alignments in {align_dir}")
        align_files = glob.glob(os.path.join(align_dir, "*.align"))
        for af in align_files:
            utterance = process_align_file(af)
            if utterance.strip():  # Only add if there's some content
                all_utterances.append(utterance)

    # Write all utterances to the output file (one line per utterance)
    with open(output_file, "w", encoding="utf-8") as out_f:
        for utt in all_utterances:
            out_f.write(utt + "\n")
    print(f"Saved {len(all_utterances)} utterances to {output_file}")

if __name__ == "__main__":
    # Base directory where raw GRID data is stored
    data_dir = "data/raw"
    # Output file to store the corpus
    output_file = "data/processed/grid_corpus.txt"
    
    gather_alignments(data_dir, output_file)
