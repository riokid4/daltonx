"""daltonx.grounding.process
=============================
Module containing various string processing functions used for grounding.
Ported 1-to-1 from gilda-master/gilda/process.py.
"""
from typing import List, Tuple
import regex as re
import unidecode

from .greek_alphabet import greek_alphabet, greek_to_latin

__all__ = [
    "replace_dashes",
    "remove_dashes",
    "replace_whitespace",
    "normalize",
    "split_preserve_tokens",
    "replace_greek_uni",
    "replace_greek_latin",
    "replace_greek_spelled_out",
    "replace_unicode",
    "get_capitalization_pattern",
    "depluralize",
    "replace_roman_arabic",
]

#: A list of all kinds of dashes
dashes = [chr(0x2212), chr(0x002d)] + [chr(c) for c in range(0x2010, 0x2016)]


def replace_dashes(s, rep='-'):
    """Replace all types of dashes in a given string with a given replacement."""
    for d in dashes:
        s = s.replace(d, rep)
    return s


def remove_dashes(s):
    """Remove all types of dashes in the given string."""
    return replace_dashes(s, '')


def replace_whitespace(s, rep=' '):
    """Replace any length white spaces in the given string with a replacement."""
    s = re.sub(r'\s+', rep, s)
    return s


def normalize(s):
    """Normalize white spaces, dashes and case of a given string."""
    s = replace_whitespace(s)
    s = remove_dashes(s)
    s = replace_unicode(s)
    s = s.lower()
    return s


def split_preserve_tokens(s):
    """Return split words of a string including the non-word tokens."""
    return re.split(r'(\W)', s)


def replace_greek_uni(s):
    """Replace Greek spelled out letters with their unicode character."""
    for greek_uni, greek_spelled_out in greek_alphabet.items():
        s = s.replace(greek_spelled_out, greek_uni)
    return s


def replace_greek_latin(s):
    """Replace Greek spelled out letters with their latin character."""
    for greek_spelled_out, latin in greek_to_latin.items():
        s = s.replace(greek_spelled_out, latin)
    return s


def replace_greek_spelled_out(s):
    """Replace Greek unicode character with latin spelled out."""
    for greek_uni, greek_spelled_out in greek_alphabet.items():
        s = s.replace(greek_uni, greek_spelled_out)
    return s


def replace_unicode(s):
    """Replace unicode with ASCII equivalent, except Greek letters."""
    if unidecode.unidecode(s) == s:
        return s
    return ''.join(unidecode.unidecode(c) if c not in greek_alphabet else c
                   for c in s)


def get_capitalization_pattern(word, beginning_of_sentence=False):
    """Return the type of capitalization for the string.

    Returns one of: sentence_initial_cap, single_cap_letter, all_caps,
    all_lower, initial_cap, mixed.
    """
    if beginning_of_sentence and re.match(r'^\p{Lu}\p{Ll}*$', word):
        return 'sentence_initial_cap'
    elif re.match(r'^\p{Lu}$', word):
        return 'single_cap_letter'
    elif re.match(r'^\p{Lu}+$', word):
        return 'all_caps'
    elif re.match(r'^\p{Ll}+$', word):
        return 'all_lower'
    elif re.match(r'^\p{Lu}\p{Ll}+$', word):
        return 'initial_cap'
    else:
        return 'mixed'


def depluralize(word: str) -> List[Tuple[str, str]]:
    """Return the depluralized version of the word, along with a status flag."""
    if not word.endswith('s'):
        return [(word, 'non_plural')]
    elif word.endswith('sis'):
        return [(word, 'non_plural')]
    elif word.endswith('oes'):
        return [(word[:-2], 'plural_oes'),
                (word[:-1], 'plural_s')]
    elif word.endswith('ies'):
        return [(word[:-3] + 'y', 'plural_ies'),
                (word[:-1], 'plural_s')]
    elif word.endswith(('xes', 'ses', 'ches', 'shes')):
        return [(word[:-2], 'plural_es'), (word[:-1], 'plural_s')]
    elif re.match(r'^\p{Lu}+$', word[:-1]):
        return [(word[:-1], 'plural_caps_s')]
    else:
        return [(word[:-1], 'plural_s')]


def _make_roman_arabic_patterns():
    roman_arabic = {
        'I': '1',
        'II': '2',
        'III': '3',
        'IV': '4',
        'V': '5',
        'VI': '6',
        'VII': '7',
        'VIII': '8',
        'IX': '9',
        'X': '10'
    }

    patterns = {}
    for r, a in roman_arabic.items():
        for x, y in [(r, a), (a, r)]:
            patterns[x] = (re.compile(r'^(.*[- ])(%s)$' % x, re.IGNORECASE),
                           r'\g<1>%s' % y)
    return patterns


roman_arabic_patterns = _make_roman_arabic_patterns()
roman_arabic_prefilter = re.compile(r'^.*[- ](\d+|[IXV]+)$', re.IGNORECASE)


def replace_roman_arabic(s):
    match = roman_arabic_prefilter.match(s)
    if not match:
        return s
    else:
        pattern = roman_arabic_patterns.get(match.groups()[0].upper())
        return pattern[0].sub(pattern[1], s) if pattern else s
