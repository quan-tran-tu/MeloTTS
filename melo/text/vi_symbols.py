# Vietnamese phoneme symbols based on IPA (International Phonetic Alphabet)
from typing import List, Dict

# Punctuation marks commonly used in Vietnamese
punctuation = ["!", "?", "…", ",", ".", "'", "-", '"', '"', ":", ";"]
pu_symbols = punctuation + ["SP", "UNK"]
pad = "_"

# Vietnamese vowels (monophthongs)
vi_monophthongs = [
    'a',    # as in 'ba'
    'ă',    # as in 'băn'
    'â',    # as in 'bận'
    'e',    # as in 'be'
    'ê',    # as in 'bê'
    'i',    # as in 'bi'
    'o',    # as in 'bo'
    'ô',    # as in 'bô'
    'ơ',    # as in 'bơ'
    'u',    # as in 'bu'
    'ư',    # as in 'bư'
    'ɯ',    # IPA representation of 'ư'
    'ɛ',    # IPA representation of open 'e'
    'ɔ',    # IPA representation of open 'o'
    'ə',    # IPA schwa sound
]

# Vietnamese vowels (diphthongs)
vi_diphthongs = [
    'ie', 'iê',     # as in 'biển'
    'uo', 'uô',     # as in 'buôn'
    'ưo', 'ươ',     # as in 'mương'
    'ua', 'uă',     # as in 'lua'
]

# Vietnamese consonants
vi_consonants = [
    'b',    # as in 'ba'
    'c',    # as in 'ca'
    'd',    # as in 'da'
    'đ',    # as in 'đa'
    'g',    # as in 'ga'
    'h',    # as in 'ha'
    'k',    # as in 'ka'
    'kʰ',   # aspirated k
    'l',    # as in 'la'
    'm',    # as in 'ma'
    'n',    # as in 'na'
    'ŋ',    # as in 'nga'
    'ɲ',    # as in 'nha'
    'p',    # as in 'pa'
    'r',    # as in 'ra'
    's',    # as in 'sa'
    't',    # as in 'ta'
    'tʰ',   # aspirated t
    'ʈ',    # retroflex t
    'v',    # as in 'va'
    'x',    # as in 'xa'
    'ʂ',    # (for 's' in some positions)
]

# Combine all normal symbols (maintain order like in symbols.py)
normal_symbols = sorted(set(
    vi_monophthongs +
    vi_diphthongs +
    vi_consonants
))

# All symbols including padding and punctuation
symbols: List[str] = [pad] + normal_symbols + pu_symbols

# Get indices of silence phonemes
sil_phonemes_ids = [symbols.index(i) for i in pu_symbols]

# Vietnamese has 6 tones:
# 1: ngang (level) - default tone
# 2: huyền (low falling)
# 3: hỏi (dipping)
# 4: ngã (breaking)
# 5: sắc (rising)
# 6: nặng (heavy)
num_tones = 6

# Language mappings (single Vietnamese language)
language_id_map: Dict[str, int] = {
    "VI": 0,  # Standard Vietnamese
}
num_languages = len(language_id_map)

# Mapping language to tone start position
language_tone_start_map: Dict[str, int] = {
    "VI": 0,  # Vietnamese tones start at position 0
}

if __name__ == "__main__":
    print(f"Number of symbols: {len(symbols)}")
    print(f"Symbols: {symbols}")
    print(f"Number of tones: {num_tones}")
    print(f"Silence phoneme IDs: {sil_phonemes_ids}")