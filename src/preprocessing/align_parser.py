def parse_align_file(align_path):
    """Parse the alignment file to extract phoneme/word labels."""
    labels = []
    with open(align_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3:  # Format: start_time end_time phoneme/word
                labels.append(parts[2])
    return labels
