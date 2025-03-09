def parse_align_file(align_path):
    """Parse .align file to build the reference text. 
       By default, skip 'sil' or special tokens if you like.
    """
    words = []
    with open(align_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                token = parts[2]
                if token not in ["sil", "sp"]:  # ignoring silence tokens
                    words.append(token)
    # Return joined words as reference text
    return " ".join(words)

def levenshtein_distance(seq1, seq2):
    """
    Compute the Levenshtein distance between two token lists.
    seq1, seq2: lists of tokens (words or chars)
    """
    m, n = len(seq1), len(seq2)
    # dp[i][j] = edit distance between seq1[:i] and seq2[:j]
    dp = [[0]*(n+1) for _ in range(m+1)]

    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # deletion
                    dp[i][j-1],    # insertion
                    dp[i-1][j-1]   # substitution
                )
    return dp[m][n]

def compute_wer(ref_text, hyp_text):
    """
    WER = (Levenshtein distance on word level) / (# words in reference)
    ref_text, hyp_text: space-separated strings
    """
    ref_words = ref_text.split()
    hyp_words = hyp_text.split()
    distance = levenshtein_distance(ref_words, hyp_words)
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    return distance / len(ref_words)

def compute_cer(ref_text, hyp_text):
    """
    CER = (Levenshtein distance on character level) / (# chars in reference)
    ref_text, hyp_text: strings
    - remove spaces if you want literal character
    """
    ref_chars = ref_text.replace(" ", "")
    hyp_chars = hyp_text.replace(" ", "")
    distance = levenshtein_distance(list(ref_chars), list(hyp_chars))
    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0
    return distance / len(ref_chars)
