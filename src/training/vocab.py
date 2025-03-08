# We'll store your original vocab in a temporary dict
VOCAB_BASE = {
    "a": 0,
    "again": 1,
    "at": 2,
    "b": 3,
    "bin": 4,
    "blue": 5,
    "by": 6,
    "c": 7,
    "d": 8,
    "e": 9,
    "eight": 10,
    "f": 11,
    "five": 12,
    "four": 13,
    "g": 14,
    "green": 15,
    "h": 16,
    "i": 17,
    "in": 18,
    "j": 19,
    "k": 20,
    "l": 21,
    "lay": 22,
    "m": 23,
    "n": 24,
    "nine": 25,
    "now": 26,
    "o": 27,
    "one": 28,
    "p": 29,
    "place": 30,
    "please": 31,
    "q": 32,
    "r": 33,
    "red": 34,
    "s": 35,
    "set": 36,
    "seven": 37,
    "sil": 38,
    "six": 39,
    "soon": 40,
    "sp": 41,
    "t": 42,
    "three": 43,
    "two": 44,
    "u": 45,
    "v": 46,
    "white": 47,
    "with": 48,
    "x": 49,
    "y": 50,
    "z": 51,
    "zero": 52
}

# Create a new VOCAB with a dedicated <blank> token at index 0
VOCAB = {"<blank>": 0}
for token, idx in VOCAB_BASE.items():
    # Shift original indices by +1
    VOCAB[token] = idx + 1

def text_to_labels(words):
    """
    Convert a list of word strings to a list of integer labels.
    If a word is not found, we skip it (or you could handle as <unk>).
    """
    labels = []
    for w in words:
        if w in VOCAB:
            labels.append(VOCAB[w])
        else:
            # Unknown words can be skipped or mapped to <unk> if you add an <unk> token
            pass
    return labels

def vocab_size():
    return len(VOCAB)  # Includes the <blank> token
