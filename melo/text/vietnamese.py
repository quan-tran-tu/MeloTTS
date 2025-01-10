from viphoneme import vi2IPA
import re
from transformers import AutoTokenizer
from typing import List, Tuple

from .vi_symbols import symbols
from . import punctuation

# Initialize tokenizer
model_id = 'vinai/phobert-base'
tokenizer = AutoTokenizer.from_pretrained(model_id)

def text_normalize(text: str) -> str:
    """Normalize Vietnamese text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Convert basic punctuation
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace('…', '...')
    
    # Replace multiple periods with ellipsis
    text = re.sub(r'\.{3,}', '...', text)
    
    # Normalize whitespace around punctuation
    text = re.sub(r'\s*([,\.!?;:])\s*', r'\1 ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def distribute_phone(n_phone: int, n_word: int) -> List[int]:
    """Distribute phones across words."""
    phones_per_word = [0] * n_word
    for task in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word

def process_phonemes(text: str) -> Tuple[List[str], List[int]]:
    """Convert text to phonemes and tones using viphoneme."""
    try:
        # Convert to IPA and split into words
        ipa_text = vi2IPA(text)
        phoneme_words = ipa_text.strip().split()
        
        phones = []
        tones = []
        
        for word in phoneme_words:
            # Extract tone number if present at the end of the phoneme
            if word[-1].isdigit():
                tone = int(word[-1])
                phoneme = word[:-1]
            else:
                tone = 1  # Default tone (ngang)
                phoneme = word
            
            # Handle multi-character phonemes
            current_pos = 0
            while current_pos < len(phoneme):
                found_phoneme = False
                # Try longer phonemes first
                for length in [2, 1]:
                    if current_pos + length <= len(phoneme):
                        candidate = phoneme[current_pos:current_pos + length]
                        if candidate in symbols:
                            phones.append(candidate)
                            tones.append(tone)
                            current_pos += length
                            found_phoneme = True
                            break
                
                if not found_phoneme:
                    # Skip unknown character
                    current_pos += 1
        
        return phones, tones
    except Exception as e:
        print(f"Error processing phonemes for text '{text}': {str(e)}")
        return ["UNK"], [1]

def g2p(norm_text: str) -> Tuple[List[str], List[int], List[int]]:
    """Convert normalized text to phonemes, tones, and word-to-phoneme mapping."""
    tokenized = tokenizer.tokenize(norm_text)
    
    # Group subtokens
    ph_groups = []
    for t in tokenized:
        if not t.startswith('##'):
            ph_groups.append([t])
        else:
            ph_groups[-1].append(t[2:])  # Remove '##' prefix
    
    phones = []
    tones = []
    word2ph = []
    
    for group in ph_groups:
        text = ''.join(group)
        
        if text in punctuation:
            # Handle punctuation
            phones.append(text)
            tones.append(0)
            word2ph.append(1)
            continue
        
        # Convert to phonemes
        ph_list, tone_list = process_phonemes(text)
        
        # Skip empty results
        if not ph_list:
            continue
            
        phones.extend(ph_list)
        tones.extend(tone_list)
        
        # Distribute phones across the word tokens
        word2ph.extend(distribute_phone(len(ph_list), len(group)))
    
    # Add start and end tokens
    phones = ["_"] + phones + ["_"]
    tones = [0] + tones + [0]
    word2ph = [1] + word2ph + [1]
    
    return phones, tones, word2ph

def get_bert_feature(text: str, word2ph: List[int], device: str = None):
    """Get BERT features for the text."""
    from . import vietnamese_bert
    return vietnamese_bert.get_bert_feature(text, word2ph, device=device)

if __name__ == "__main__":
    # Test the functions
    test_text = "Xin chào thế giới!"
    print("Original text:", test_text)
    
    normalized_text = text_normalize(test_text)
    print("Normalized text:", normalized_text)
    
    phones, tones, word2ph = g2p(normalized_text)
    print("Phones:", phones)
    print("Tones:", tones)
    print("Word2ph:", word2ph)