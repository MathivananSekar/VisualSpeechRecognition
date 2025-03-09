import torch
from transformers import MBartForConditionalGeneration, MBartTokenizer, MBart50TokenizerFast

def translate_en_to_fr(text, model, tokenizer, device):
    """
    Translates English text to French using facebook/mbart-large-50-many-to-many-mmt.

    Args:
        text (str): English text to translate.
        model: Pre-loaded mBART model.
        tokenizer: Pre-loaded mBART tokenizer.
        device: Device to run inference on (CPU/GPU).

    Returns:
        str: Translated text in French.
    """
    try:
        # Set source language (English)
        tokenizer.src_lang = "en_XX"
        if "en_XX" not in tokenizer.lang_code_to_id:
            raise ValueError("Source language 'en_XX' not supported by tokenizer")

        # Tokenize input text
        encoded_input = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)

        # Generate translation with target language (French)
        if "fr_XX" not in tokenizer.lang_code_to_id:
            raise ValueError("Target language 'fr_XX' not supported by tokenizer")
        generated_tokens = model.generate(
            **encoded_input,
            forced_bos_token_id=tokenizer.lang_code_to_id["fr_XX"],
            max_length=100,
            num_beams=5,
            early_stopping=True
        )

        # Decode tokens
        translated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        return translated_text

    except Exception as e:
        print(f"Translation error: {str(e)}")
        return None

def translate_en_to_es(text, model, tokenizer, device):
    """
    Translates English text to Spanish using facebook/mbart-large-50-many-to-many-mmt.
    """
    try:
        tokenizer.src_lang = "en_XX"
        encoded_input = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        generated_tokens = model.generate(
            **encoded_input,
            forced_bos_token_id=tokenizer.lang_code_to_id["es_XX"],
            max_length=100,
            num_beams=5,
            early_stopping=True
        )
        translated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        return translated_text
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return None

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model & tokenizer
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    try:
        # Try fast tokenizer first
        tokenizer = MBart50TokenizerFast.from_pretrained(model_name, src_lang="en_XX", tgt_lang="fr_XX")
    except Exception as e:
        print(f"Fast tokenizer failed: {str(e)}. Falling back to slow tokenizer.")
        # Fallback to slow tokenizer
        tokenizer = MBartTokenizer.from_pretrained(model_name, src_lang="en_XX", tgt_lang="fr_XX")

    try:
        model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        exit(1)

    # Example English text from GRID corpus
    english_text = "bin blue at f two now"

    # Translate to French and Spanish
    french_translation = translate_en_to_fr(english_text, model, tokenizer, device)
    spanish_translation = translate_en_to_es(english_text, model, tokenizer, device)

    print("English Text:", english_text)
    if french_translation:
        print("French Translation:", french_translation)
    if spanish_translation:
        print("Spanish Translation:", spanish_translation)