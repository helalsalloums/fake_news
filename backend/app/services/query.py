from __future__ import annotations

import re

from app.schemas.domain import ExtractedClaim


class ArabicQueryGenerator:
    def generate(self, claim: ExtractedClaim, limit: int = 4) -> list[str]:
        phrases: list[str] = []
        entity_terms = " ".join(f'"{entity}"' for entity in claim.entities[:2])
        number_terms = " ".join(f'"{number}"' for number in claim.numbers[:3])
        date_terms = " ".join(claim.dates[:1])
        core = re.sub(r"\b(?:قال|أعلن|أكد|أوضح|ذكر|صرح)\w*\b", "", claim.normalized_text)
        core = re.sub(r"\s+", " ", core).strip()

        if entity_terms or number_terms:
            phrases.append(
                " ".join(item for item in [entity_terms, number_terms, date_terms] if item)
            )
        if claim.numbers:
            words = [word for word in core.split() if len(word) > 2][:8]
            phrases.append(f'"{claim.numbers[0]}" ' + " ".join(words))
        phrases.append(f'"{core[:120]}"')
        phrases.append(core[:180])

        unique: list[str] = []
        for phrase in phrases:
            phrase = phrase.strip()
            if phrase and phrase not in unique:
                unique.append(phrase)
        return unique[:limit]
