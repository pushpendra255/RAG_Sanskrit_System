"""Utility helpers for Sanskrit document processing and query handling."""

from __future__ import annotations

import os
import re
import unicodedata
from typing import List

DEVANAGARI_PATTERN = re.compile(r"[\u0900-\u097F]")
LATIN_PATTERN = re.compile(r"[A-Za-z]")
TOKEN_PATTERN = re.compile(r"[\u0900-\u097F]+|[A-Za-z0-9]+", re.UNICODE)
LATIN_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
FALLBACK_ANSWER = "उत्तर उपलब्ध नहीं है।"
QUERY_STOPWORDS = {
    "किम्",
    "किं",
    "कथम्",
    "कथं",
    "कः",
    "का",
    "के",
    "कुत्र",
    "कदा",
    "कथ",
    "what",
    "who",
    "when",
    "where",
    "why",
    "how",
}
QUERY_HINT_MAP = {
    "सहाय्य": ["दत्तवान्", "आनीतवान्", "सुभाषित", "उपाय"],
    "सहाय्यं": ["दत्तवान्", "आनीतवान्", "सुभाषित", "उपाय"],
    "help": ["दत्तवान्", "आनीतवान्", "सुभाषित", "उपाय"],
    "नष्ट": ["हत", "नाश", "मया"],
    "नष्टः": ["हत", "नाश", "मया"],
    "destroy": ["हत", "नाश", "मया"],
}


def normalize_sanskrit_text(text: str) -> str:
    """Normalize mixed Sanskrit text while preserving paragraph structure."""
    normalized = unicodedata.normalize("NFC", text or "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("“", '"').replace("”", '"').replace("’", "'")
    normalized = re.sub(r"[^\u0900-\u097FA-Za-z0-9\s.,;:!?()\[\]\"'/_\-–—|।॥]", " ", normalized)

    cleaned_lines = []
    for raw_line in normalized.split("\n"):
        line = re.sub(r"\s+", " ", raw_line).strip()
        cleaned_lines.append(line)

    normalized = "\n".join(cleaned_lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def clean_text(text: str) -> str:
    """Remove obvious noise while keeping document-level structure intact."""
    cleaned = unicodedata.normalize("NFC", text or "")
    cleaned = re.sub(r"http\S+|www\S+|https\S+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\S+@\S+", "", cleaned)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def contains_devanagari(text: str) -> bool:
    """Return True when the text contains Devanagari characters."""
    return bool(DEVANAGARI_PATTERN.search(text or ""))


def looks_like_transliterated_sanskrit(text: str) -> bool:
    """Heuristic check for Latin-script Sanskrit queries."""
    if not text:
        return False

    latin_chars = LATIN_PATTERN.findall(text)
    if not latin_chars:
        return False

    devanagari_chars = DEVANAGARI_PATTERN.findall(text)
    return len(latin_chars) >= max(4, len(devanagari_chars))


def transliterate_query(text: str) -> List[str]:
    """Generate Devanagari variants from a transliterated Sanskrit query."""
    variants: List[str] = []
    if not looks_like_transliterated_sanskrit(text):
        return variants

    try:
        from indic_transliteration import sanscript
        from indic_transliteration.detect import detect
    except ImportError:
        return variants

    normalized = unicodedata.normalize("NFC", text or "").strip()
    schemes_to_try = []
    detected_scheme = detect(normalized)
    if detected_scheme:
        schemes_to_try.append(detected_scheme)

    for scheme in (sanscript.HK, sanscript.ITRANS, sanscript.IAST, sanscript.VELTHUIS):
        if scheme not in schemes_to_try:
            schemes_to_try.append(scheme)

    for scheme in schemes_to_try:
        try:
            transliterated = sanscript.transliterate(normalized, scheme, sanscript.DEVANAGARI)
        except Exception:
            continue
        transliterated = normalize_query_text(transliterated)
        if transliterated and transliterated not in variants:
            variants.append(transliterated)

    return variants


def normalize_query_text(text: str) -> str:
    """Normalize Sanskrit or transliterated queries for retrieval and matching."""
    original = unicodedata.normalize("NFC", text or "")
    normalized = original.lower()
    normalized = normalized.replace("_", " ")
    normalized = re.sub(r"[^\u0900-\u097FA-Za-z0-9\s]", " ", normalized, flags=re.UNICODE)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    keyword_map = {
        "कहानी": "कथा",
        "सच्चाई": "तत्त्व",
        "संदेश": "सन्देश",
        "क्या": "किम्",
        "message": "सन्देश",
        "story": "कथा",
        "what": "किम्",
    }
    for source_token, target_token in keyword_map.items():
        normalized = re.sub(rf"\b{re.escape(source_token)}\b", target_token, normalized)

    return normalized if normalized else original.strip()


def prepare_query_variants(text: str) -> List[str]:
    """Build a small set of query variants for retrieval and re-ranking."""
    normalized = normalize_query_text(text)
    variants = [normalized] if normalized else []

    if not contains_devanagari(normalized):
        for variant in transliterate_query(text):
            if variant not in variants:
                variants.append(variant)

    return variants


def extract_paragraphs(text: str) -> List[str]:
    """Split cleaned document text into paragraphs."""
    if not text:
        return []

    blocks = re.split(r"\n\s*\n|\n", text)
    paragraphs = []
    for block in blocks:
        paragraph = re.sub(r"\s+", " ", block).strip()
        if paragraph:
            paragraphs.append(paragraph)
    return paragraphs


def is_probably_heading(text: str) -> bool:
    """Identify short Sanskrit title-like lines."""
    if not text or not contains_devanagari(text):
        return False

    stripped = text.strip(" \"'।॥")
    word_count = len(stripped.split())
    if word_count == 0 or word_count > 6:
        return False

    narrative_openers = {"तदा", "ततः", "अथ", "एकः", "सा", "सः", "भो", "यदा", "किंचित"}
    first_token = stripped.split()[0]
    if first_token in narrative_openers or "इति" in stripped:
        return False

    if any(punct in stripped for punct in ("?", "!", ",")):
        return False

    return len(stripped) <= 60


def is_probably_noise(text: str) -> bool:
    """Filter out metadata and English-heavy helper text from the corpus."""
    if not text:
        return True

    latin_count = len(LATIN_PATTERN.findall(text))
    devanagari_count = len(DEVANAGARI_PATTERN.findall(text))
    if devanagari_count == 0 and latin_count > 0:
        return True

    return "@" in text and devanagari_count < latin_count


def split_into_sentences(text: str) -> List[str]:
    """Split Sanskrit text into sentence-like segments."""
    if not text:
        return []

    sentences = re.split(r"[।॥\n]+", text)
    result = []
    for sentence in sentences:
        parts = re.split(r"[.!?]+", sentence)
        result.extend([part.strip() for part in parts if part.strip()])
    return result


def normalize_matching_text(text: str) -> str:
    """Normalize text for token overlap and answer grounding."""
    return normalize_query_text(text).strip()


def get_normalized_tokens(text: str) -> List[str]:
    """Return normalized Sanskrit or Latin tokens for matching."""
    normalized = normalize_matching_text(text)
    return [_normalize_search_token(token) for token in TOKEN_PATTERN.findall(normalized) if token]


def get_focus_tokens(text: str) -> List[str]:
    """Extract the informative query tokens used for anchor-style matching."""
    tokens = [token for token in get_normalized_tokens(text) if token]
    focus_tokens = [token for token in tokens if token not in QUERY_STOPWORDS and len(token) >= 3]

    if focus_tokens:
        return focus_tokens
    return tokens


def get_query_hint_tokens(text: str) -> List[str]:
    """Expand a query with a few lightweight semantic hint tokens."""
    hint_tokens: List[str] = []
    for token in get_focus_tokens(text):
        for source_key, mapped_tokens in QUERY_HINT_MAP.items():
            if _rough_token_match(token, _normalize_search_token(source_key)):
                for mapped_token in mapped_tokens:
                    normalized = _normalize_search_token(mapped_token)
                    if normalized and normalized not in hint_tokens:
                        hint_tokens.append(normalized)
    return hint_tokens


def text_overlap_score(query_text: str, candidate_text: str) -> float:
    """Compute a lightweight lexical overlap score in the range [0, 1]."""
    query_tokens = set(get_normalized_tokens(query_text))
    if not query_tokens:
        return 0.0

    candidate_tokens = set(get_normalized_tokens(candidate_text))
    if not candidate_tokens:
        return 0.0

    overlap = 0
    for query_token in query_tokens:
        if any(_rough_token_match(query_token, candidate) for candidate in candidate_tokens):
            overlap += 1
    return overlap / max(1, len(query_tokens))


def token_anchor_score(query_tokens: List[str], candidate_text: str) -> float:
    """Measure how well informative query tokens are covered by candidate text."""
    if not query_tokens:
        return 0.0

    candidate_tokens = set(get_normalized_tokens(candidate_text))
    if not candidate_tokens:
        return 0.0

    overlap = 0
    for query_token in query_tokens:
        if any(_rough_token_match(query_token, candidate) for candidate in candidate_tokens):
            overlap += 1

    return overlap / max(1, len(query_tokens))


def normalize_latin_sanskrit(text: str) -> str:
    """Roughly normalize romanized Sanskrit text for fuzzy lexical matching."""
    normalized = normalize_query_text(text).lower()
    replacements = {
        "aa": "a",
        "ii": "i",
        "uu": "u",
        "ee": "e",
        "oo": "o",
        "sh": "s",
        "ṣ": "s",
        "ś": "s",
        "ṅ": "n",
        "ñ": "n",
        "ṇ": "n",
        "ṃ": "m",
        "ṁ": "m",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)

    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    tokens = []
    for token in normalized.split():
        if token.endswith("h") and len(token) > 3:
            token = token[:-1]
        if token.endswith("m") and len(token) > 3:
            token = token[:-1]
        if token:
            tokens.append(token)
    return " ".join(tokens)


def romanized_overlap_score(query_text: str, candidate_text: str) -> float:
    """Fuzzy overlap between transliterated query text and romanized chunk text."""
    query_tokens = set(LATIN_TOKEN_PATTERN.findall(normalize_latin_sanskrit(query_text)))
    if not query_tokens:
        return 0.0

    candidate_tokens = set(LATIN_TOKEN_PATTERN.findall(normalize_latin_sanskrit(candidate_text)))
    if not candidate_tokens:
        return 0.0

    overlap = 0
    for query_token in query_tokens:
        if any(_rough_latin_token_match(query_token, candidate) for candidate in candidate_tokens):
            overlap += 1

    return overlap / max(1, len(query_tokens))


def romanized_anchor_score(query_tokens: List[str], candidate_text: str) -> float:
    """Measure anchor coverage for Latin-script Sanskrit matching."""
    if not query_tokens:
        return 0.0

    candidate_tokens = set(LATIN_TOKEN_PATTERN.findall(normalize_latin_sanskrit(candidate_text)))
    if not candidate_tokens:
        return 0.0

    latin_query_tokens = [normalize_latin_sanskrit(token) for token in query_tokens]
    latin_query_tokens = [token for token in latin_query_tokens if token]
    if not latin_query_tokens:
        return 0.0

    overlap = 0
    for query_token in latin_query_tokens:
        if any(_rough_latin_token_match(query_token, candidate) for candidate in candidate_tokens):
            overlap += 1

    return overlap / max(1, len(latin_query_tokens))


def _rough_latin_token_match(query_token: str, candidate_token: str) -> bool:
    """Match romanized Sanskrit tokens conservatively."""
    if query_token == candidate_token:
        return True

    min_length = min(len(query_token), len(candidate_token))
    if min_length < 5:
        return False

    shared_prefix = 0
    for left, right in zip(query_token, candidate_token):
        if left != right:
            break
        shared_prefix += 1

    return shared_prefix >= max(4, int(0.7 * min_length))


def _normalize_search_token(token: str) -> str:
    """Reduce small orthographic variation before overlap scoring."""
    normalized = token.strip().lower()
    normalized = re.sub(r"[^\u0900-\u097Fa-z0-9]", "", normalized)

    devanagari_suffixes = ["ः", "म्", "ं", "ा", "ी", "ि", "ु", "ू", "े", "ै", "ो", "ौ"]
    for suffix in devanagari_suffixes:
        if normalized.endswith(suffix) and len(normalized) > len(suffix) + 1:
            normalized = normalized[: -len(suffix)]
            break

    if normalized.endswith("h") and len(normalized) > 3:
        normalized = normalized[:-1]

    return normalized


def _rough_token_match(query_token: str, candidate_token: str) -> bool:
    """Match normalized Sanskrit tokens conservatively."""
    if query_token == candidate_token:
        return True

    min_length = min(len(query_token), len(candidate_token))
    if min_length < 3:
        return False

    if query_token in candidate_token or candidate_token in query_token:
        return min_length >= 4

    shared_prefix = 0
    for left, right in zip(query_token, candidate_token):
        if left != right:
            break
        shared_prefix += 1

    return shared_prefix >= max(3, int(0.7 * min_length))


def get_file_extension(file_path: str) -> str:
    """Get a file extension in lowercase."""
    return os.path.splitext(file_path)[1].lower()


def ensure_dir_exists(dir_path: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(dir_path, exist_ok=True)


def save_embeddings_metadata(metadata: dict, save_path: str) -> None:
    """Save metadata as UTF-8 JSON."""
    import json

    with open(save_path, "w", encoding="utf-8") as file_handle:
        json.dump(metadata, file_handle, ensure_ascii=False, indent=2)


def load_embeddings_metadata(metadata_path: str) -> dict:
    """Load metadata from UTF-8 JSON."""
    import json

    with open(metadata_path, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def print_progress(current: int, total: int, prefix: str = "Processing") -> None:
    """Render a small ASCII progress bar."""
    if total <= 0:
        return

    percent = 100 * (current / float(total))
    bar_length = 40
    filled = int(bar_length * current // total)
    bar = "#" * filled + "-" * (bar_length - filled)
    print(f"\r{prefix} |{bar}| {percent:.1f}% ({current}/{total})", end="", flush=True)
    if current == total:
        print()
