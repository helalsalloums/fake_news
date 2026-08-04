from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass

ARABIC_DIACRITICS = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
TATWEEL = "\u0640"
WHITESPACE = re.compile(r"\s+")
URL_PATTERN = re.compile(r"https?://[^\s<>\]\[\"']+", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"(?<!\w)[+-]?(?:\d+(?:[.,]\d+)?|[٠-٩]+(?:[٫٬][٠-٩]+)?)(?!\w)")
DATE_PATTERN = re.compile(
    r"(?:(?:\d{1,2}|[٠-٩]{1,2})[/-](?:\d{1,2}|[٠-٩]{1,2})[/-](?:\d{2,4}|[٠-٩]{2,4}))"
)

ARABIC_TO_ASCII = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
LETTER_TRANSLATION = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ؤ": "و",
        "ئ": "ي",
    }
)
PUNCT_TRANSLATION = str.maketrans({"،": ",", "؛": ";", "؟": "?", "٪": "%", "٫": ".", "٬": ","})


@dataclass(frozen=True)
class NormalizedArabic:
    original: str
    normalized: str
    urls: tuple[str, ...]
    numbers: tuple[str, ...]
    dates: tuple[str, ...]
    content_hash: str


def normalize_arabic(text: str, *, remove_diacritics: bool = True) -> str:
    value = unicodedata.normalize("NFKC", text).replace(TATWEEL, "")
    if remove_diacritics:
        value = ARABIC_DIACRITICS.sub("", value)
    value = value.translate(ARABIC_TO_ASCII)
    value = value.translate(LETTER_TRANSLATION)
    value = value.translate(PUNCT_TRANSLATION)
    value = WHITESPACE.sub(" ", value).strip()
    return value


def analyze_arabic(text: str, *, remove_diacritics: bool = True) -> NormalizedArabic:
    normalized = normalize_arabic(text, remove_diacritics=remove_diacritics)
    dates = tuple(DATE_PATTERN.findall(normalized))
    numbers = tuple(
        match.group(0).replace(",", "") for match in NUMBER_PATTERN.finditer(normalized)
    )
    urls = tuple(URL_PATTERN.findall(text))
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return NormalizedArabic(text, normalized, urls, numbers, dates, digest)


def normalized_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[\w%]+", normalize_arabic(text).lower(), flags=re.UNICODE)
        if len(token) > 1
    }


def similarity_ratio(left: str, right: str) -> float:
    left_tokens = normalized_tokens(left)
    right_tokens = normalized_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def is_duplicate(left: str, right: str, threshold: float = 0.92) -> bool:
    left_normalized = normalize_arabic(left)
    right_normalized = normalize_arabic(right)
    return left_normalized == right_normalized or similarity_ratio(left, right) >= threshold
