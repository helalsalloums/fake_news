from __future__ import annotations

import re

from app.schemas.domain import ClaimType, ExtractedClaim
from app.services.arabic import analyze_arabic, is_duplicate, normalize_arabic

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!؟?؛;])\s+|\n+")
CLAUSE_BOUNDARY = re.compile(r"\s+(?:،\s*)?(?:وأضاف(?:ت)?|كما|بينما|ولكن|إلا أن)\s+")
REPORTING_PREFIX = re.compile(
    r"^(?:قال(?:ت)?|أعلن(?:ت)?|أوضح(?:ت)?|أكد(?:ت)?|ذكر(?:ت)?|صرح(?:ت)?)\s+[^،,.]{1,100}?\s+(?:إن|أن)\s+"
)
FACTUAL_MARKERS = re.compile(
    r"\d|بلغ|ارتفع|انخفض|وصل|أعلن|قال|أكد|وقع|حدث|افتتح|أغلق|فاز|توفي|أصيب|دخل|خرج|أصدر|قرر"
)
ORG_PATTERN = re.compile(
    r"(?:وزارة|منظمة|هيئة|جامعة|شركة|مؤسسة|حكومة|مجلس|وكالة|بنك)\s+[\u0600-\u06ff\s]{2,45}"
)


def _claim_type(text: str, numbers: tuple[str, ...], dates: tuple[str, ...]) -> ClaimType:
    if numbers:
        return ClaimType.STATISTIC
    if dates:
        return ClaimType.DATE
    if re.search(r"قال|أعلن|أكد|صرح|نسب", text):
        return ClaimType.ATTRIBUTION
    if re.search(r"في\s+(?:مدينة|دولة|محافظة|بلدة|قرية)", text):
        return ClaimType.LOCATION
    if FACTUAL_MARKERS.search(text):
        return ClaimType.EVENT
    return ClaimType.OTHER


class RuleBasedClaimExtractor:
    async def extract(self, text: str) -> list[ExtractedClaim]:
        output: list[ExtractedClaim] = []
        sentences = [part.strip(" ،,.;؛") for part in SENTENCE_BOUNDARY.split(text) if part.strip()]
        for sentence_index, sentence in enumerate(sentences):
            clauses = [
                part.strip(" ،,.;؛") for part in CLAUSE_BOUNDARY.split(sentence) if part.strip()
            ]
            for clause in clauses:
                cleaned = REPORTING_PREFIX.sub("", clause).strip()
                if not FACTUAL_MARKERS.search(cleaned) and len(sentences) > 1:
                    continue
                analyzed = analyze_arabic(cleaned)
                if len(analyzed.normalized) < 5:
                    continue
                entities = [normalize_arabic(item).strip() for item in ORG_PATTERN.findall(clause)]
                candidate = ExtractedClaim(
                    original_text=cleaned,
                    normalized_text=analyzed.normalized,
                    claim_type=_claim_type(cleaned, analyzed.numbers, analyzed.dates),
                    sentence_index=sentence_index,
                    entities=list(dict.fromkeys(entities)),
                    numbers=list(analyzed.numbers),
                    dates=list(analyzed.dates),
                    importance=1.0 if analyzed.numbers or entities else 0.8,
                )
                if not any(
                    is_duplicate(candidate.normalized_text, old.normalized_text) for old in output
                ):
                    output.append(candidate)
        if not output:
            analyzed = analyze_arabic(text)
            output.append(
                ExtractedClaim(
                    original_text=text.strip(),
                    normalized_text=analyzed.normalized,
                    claim_type=_claim_type(text, analyzed.numbers, analyzed.dates),
                    numbers=list(analyzed.numbers),
                    dates=list(analyzed.dates),
                )
            )
        return output
