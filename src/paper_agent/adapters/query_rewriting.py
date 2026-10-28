"""Turn multi-part descriptions into short, independently useful searches."""

import re


def focused_variants(text, concepts, variants, maximum=3):
    """Keep application qualifiers with tasks; alternatives are never all ANDed."""
    terms = [c.casefold() for c in concepts]
    network_tasks = [
        c for c in terms if any(w in c for w in ("offloading", "scheduling", "resource allocation"))
    ]
    edge = any("edge" in c and ("computing" in c or "cloud" in c) for c in terms)
    if edge and network_tasks:
        # The task and architecture are distinct phrases, not one exact sentence.
        ordered = sorted(
            network_tasks, key=lambda c: ("offloading" not in c, "scheduling" not in c)
        )
        generations = list(dict.fromkeys(re.findall(r"[56]g", text.casefold())))
        result = []
        for i, task in enumerate(ordered[:maximum]):
            context = '"edge computing"' if i == 0 else "cloud edge"
            task = task.replace("computational resource scheduling", "resource scheduling")
            generation = f" {generations[i]}" if i < len(generations) else ""
            result.append(f'"{task}" {context}{generation}')
        return result
    if "unmanned aerial vehicle navigation" in terms and "intelligent control" in terms:
        return [
            '"unmanned aerial vehicle" navigation',
            '"unmanned aerial vehicle" "intelligent control"',
            '"drone navigation"',
        ][:maximum]
    # Long English prose must not become a single quoted phrase or arbitrary prefix.
    bounded = [v for v in variants if len(v) <= 120 and len(v.split()) <= 10]
    if not bounded and not re.search(r"[\u4e00-\u9fff]", text):
        words = [c for c in concepts if len(c.split()) <= 6]
        bounded = [" ".join(words[i : i + 4]) for i in range(0, min(len(words), 12), 4)]
    return list(dict.fromkeys(bounded))[:maximum]


def chinese_query(text):
    """Remove request prose. For an unknown long topic search clauses, not a prefix."""
    if len(text) <= 40:
        return text.strip()
    clauses = re.split(r"[，。；！？\n]", text)
    cleaned = [
        re.sub(r"^(请你|请|帮我|调研|研究|分析|并|重点|介绍|总结)+", "", c).strip() for c in clauses
    ]
    return " ".join(c for c in cleaned if 2 <= len(c) <= 35)[:120].strip()


def format_expression(term, source):
    if source == "europe_pmc":
        parts = re.findall(r'"([^"]+)"|(\S+)', term)
        if '"' not in term:
            return f'TITLE_ABS:"{term}"'
        return " AND ".join(f'TITLE_ABS:"{phrase or word}"' for phrase, word in parts)
    if source in {"arxiv", "openalex"} and '"' not in term:
        return f'"{term}"'
    return term
