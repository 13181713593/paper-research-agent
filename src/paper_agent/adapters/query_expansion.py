"""Bounded query alternatives, kept separate so providers never AND synonyms."""

import re

from paper_agent.adapters._concept_extraction import _SYNONYM_GROUPS


def search_variants(concepts, synonyms, *, maximum=6):
    """Reserve slots for aliases as well as concepts and explain every omission."""
    scene = any("scene text" in term.casefold() for term in concepts)
    specific_scene = scene and any(
        re.search(r"detect|recogni|spotting", term, re.IGNORECASE) for term in concepts
    )
    notes = {}
    normalized = {}
    for original in dict.fromkeys([*concepts, *synonyms]):
        term = " ".join(original.replace('"', "").split())
        key = term.casefold()
        if scene:
            term = {
                "str": "scene text recognition",
                "ocr": "optical character recognition in natural scenes",
                "optical character recognition": "optical character recognition in natural scenes",
                "e2e text spotting": "end-to-end text spotting",
            }.get(key, term)
            if key == "scene text" and specific_scene:
                notes[original] = "主题过宽，采用具体的检测、识别或 text spotting 表达"
                continue
        if not re.search("[a-zA-Z]", term) or re.search(r"[\u4e00-\u9fff]", term):
            notes[original] = "本轮保留中文原始主题，使用英文别名扩展国际数据库检索"
            continue
        if re.fullmatch(r"[a-zA-Z]{2,5}", term):
            expansion = next(
                (g[0] for g in _SYNONYM_GROUPS if key in [v.casefold() for v in g]), None
            )
            if expansion:
                term = expansion
            elif original in synonyms:
                notes[original] = "缩写含义不明确，未单独检索；优先使用完整学术短语"
                continue
        normalized[original] = term

    def unique_terms(items):
        return list(dict.fromkeys(normalized[s].casefold() for s in items if s in normalized))

    core = unique_terms(concepts)
    aliases = [s for s in unique_terms(synonyms) if s not in core[:2]]
    # Core concepts cannot consume all slots before synonyms get considered.
    ordered = list(dict.fromkeys([*core[:2], *aliases[:3], *core[2:], *aliases[3:]]))
    chosen = ordered[:maximum]
    for original, term in normalized.items():
        if term.casefold() in chosen:
            notes[original] = f"用于检索：{term.casefold()}"
        else:
            notes[original] = f"本轮未采用：最多 {maximum} 条英文表达，优先核心词及同义短语"
    return chosen, notes
