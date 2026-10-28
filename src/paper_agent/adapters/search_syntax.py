"""Compile typed alternative phrases without treating Boolean operators as text."""


def phrase(value):
    return '"' + " ".join(value.replace('"', "").split()) + '"'


def expression(source, terms, filters=None, contexts=None):
    filters = filters or {}
    field = "TITLE_ABS:" if source == "europe_pmc" else ""
    joiner = " | " if source == "semantic_scholar" else " OR "
    branches = []
    for term in terms:
        branch = field + phrase(term)
        for anchor in (contexts or {}).get(term, []):
            branch += (" + " if source == "semantic_scholar" else " AND ") + field + phrase(anchor)
        branches.append(f"({branch})" if (contexts or {}).get(term) else branch)
    value = joiner.join(branches)
    if len(terms) > 1:
        value = f"({value})"
    for term in filters.get("required_terms", []):
        value += (" + " if source == "semantic_scholar" else " AND ") + field + phrase(term)
    for term in filters.get("excluded_terms", []):
        value += (" -" if source == "semantic_scholar" else " NOT ") + field + phrase(term)
    return value
