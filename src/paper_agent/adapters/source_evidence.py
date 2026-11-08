"""Conservative source-text helpers shared by parsing and summarization."""

import re
import unicodedata


def normalized_text(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def quote_in_source(quote: str, source: str) -> bool:
    return bool(quote.strip()) and normalized_text(quote) in normalized_text(source)


def has_substantive_description(text: str) -> bool:
    # An honest limitation after a real description is not a missing field.
    # Only a placeholder at the start means this comparison axis is absent.
    value = text.strip()
    return len(value) >= 8 and not value.startswith(
        ("材料中未明确说明", "未明确说明", "不适用", "无法获取", "无法判断")
    )


def extract_abstract(text: str) -> str | None:
    match = re.search(
        r"(?:^|\n)\s*(?:#{1,6}\s*)?(?:abstract|摘要)\s*[:：—–-]?\s*", text, re.IGNORECASE
    )
    if not match:
        # Text-layer extraction may merge the title, authors and abstract into
        # one paragraph. Require an explicit label separator for inline labels.
        match = re.search(r"\babstract\s*[:：—–-]\s*", text[:20000], re.IGNORECASE)
        if not match:
            return None
    body = text[match.end() :]
    body = re.split(
        r"\n\s*\n|\b(?:index terms|keywords)\b|\n\s*(?:#{1,6}\s*)?"
        r"(?:(?:[IVX]+|\d+)\.?\s+)?(?:introduction|引言)\b",
        body,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    body = " ".join(body.split())
    return body[:12000] if len(body) >= 60 else None


def evidence_passages(text: str, limit: int) -> dict[str, str]:
    """Stable IDs for exact source slices; models select IDs, never invent quotes."""
    blocks = [text] if len(text) <= limit else [text[: int(limit * 0.6)], text[-int(limit * 0.4) :]]
    result = {}
    for block in blocks:
        for start in range(0, len(block), 450):
            excerpt = block[start : start + 450].strip()
            if excerpt:
                result[f"E{len(result) + 1:03d}"] = excerpt
    return result


def matching_sentences(text: str, pattern: str, limit: int = 2) -> str:
    # Source excerpts only: do not translate, complete, or infer missing claims.
    selected = []
    for part in re.split(r"(?<=[.!?。！？])\s+|\n\s*\n", text):
        part = " ".join(part.split())
        if 30 <= len(part) <= 1200 and re.search(pattern, part, re.IGNORECASE):
            selected.append(part[:480])
            if len(selected) == limit:
                break
    return " ".join(selected)
