"""Pure normalization helpers; no network calls and no replacement domain models."""

import re
import unicodedata
from urllib.parse import unquote, urlsplit


def text(value):
    return " ".join(str(value).split()) if value is not None and str(value).strip() else None


def normalize_doi(value):
    value = text(value)
    if not value:
        return None
    value = re.sub(r"^doi:\s*", "", value, flags=re.IGNORECASE)
    if re.match(r"(?:https?://)?(?:dx\.)?doi\.org/", value, re.IGNORECASE):
        resolver_url = value if re.match(r"https?://", value, re.IGNORECASE) else "https://" + value
        value = unquote(urlsplit(resolver_url).path.lstrip("/"))
    value = value.strip().lower()
    return value if re.fullmatch(r"10\.\d{4,9}/\S+", value) else None


def normalize_arxiv(value, *, strip_version=False):
    value = text(value)
    if not value:
        return None
    value = re.sub(r"^arxiv:\s*", "", value, flags=re.IGNORECASE)
    if re.match(r"https?://(?:export\.)?arxiv\.org/(?:abs|pdf)/", value, re.IGNORECASE):
        value = unquote(urlsplit(value).path.split("/", 2)[2])
    value = re.sub(r"\.pdf$", "", value, flags=re.IGNORECASE)
    pattern = r"(?:\d{2}(?:0[1-9]|1[0-2])\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v[1-9]\d*)?"
    if not re.fullmatch(pattern, value, flags=re.IGNORECASE):
        return None
    return re.sub(r"v\d+$", "", value, flags=re.IGNORECASE) if strip_version else value


def normalize_title(value):
    value = unicodedata.normalize("NFKC", value).casefold()
    # Preserve mathematical distinctions such as C++ vs C and x^2 vs x2.
    value = "".join(
        " " if unicodedata.category(c).startswith("P") and c not in "+=^<>" else c for c in value
    )
    return " ".join(value.split())
