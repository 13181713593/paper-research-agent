"""Bilingual topic matching plus corpus-aware BM25; no model download is required."""

import math
import re
from collections import Counter

from paper_agent.adapters._concept_extraction import (
    _SYNONYM_GROUPS,
    _ZH_EN_TERMS,
    HeuristicConceptExtractor,
)

STOP = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "using",
    "based",
    "research",
    "study",
    "method",
    "methods",
    "approach",
    "of",
    "in",
    "on",
    "to",
    "a",
    "an",
}


def tokens(text):
    english = [
        w[:-1] if len(w) > 4 and w.endswith("s") else w
        for w in re.findall(r"[a-z0-9]+", text.casefold())
        if w not in STOP
    ]
    chinese = re.findall(r"[\u4e00-\u9fff]+", text)
    return english + [run[i : i + 2] for run in chinese for i in range(max(1, len(run) - 1))]


def matches(term, text):
    words = set(tokens(term))
    return bool(words) and words.issubset(set(tokens(text)))


class MetadataRelevance:
    def __init__(self, idea, papers):
        self.idea = idea
        extracted = HeuristicConceptExtractor().extract(idea.text)
        self.terms = list(dict.fromkeys([*extracted.concepts, *idea.search_terms]))
        if not self.terms:
            self.terms = [idea.text]
        self.groups = []
        for term in self.terms:
            aliases = [term]
            for group in _SYNONYM_GROUPS:
                if term.casefold() in [s.casefold() for s in group]:
                    aliases.extend(group)
            aliases.extend(zh for zh, en in _ZH_EN_TERMS.items() if en == term)
            self.groups.append(list(dict.fromkeys(aliases)))
        self.anchors = [group for group in self.groups if any(len(tokens(t)) >= 2 for t in group)]
        self.documents = [Counter(tokens(p.title + " " + (p.abstract or ""))) for p in papers]
        self.df = Counter(token for doc in self.documents for token in doc)
        self.size = max(1, len(papers))
        self.average = sum(sum(d.values()) for d in self.documents) / self.size or 1
        self.query = set(tokens(" ".join(self.terms) + " " + idea.text))

    def score(self, paper):
        text = paper.title + " " + (paper.abstract or "") + " " + " ".join(paper.keywords)
        doc = Counter(tokens(text))
        hits = [next((t for t in group if matches(t, text)), None) for group in self.groups]
        anchor = not self.anchors or any(any(matches(t, text) for t in g) for g in self.anchors)
        if not self.anchors:
            # Without a phrase anchor, one shared generic word is insufficient.
            anchor = sum(bool(hit) for hit in hits) >= min(2, len(self.groups))
        coverage = sum(bool(hit) for hit in hits) / max(1, len(hits))
        # A generic alias (e.g. "scene text") in an application paper's title
        # must not count as a title match for every detection/recognition task.
        title = sum(matches(t, paper.title) for t in self.terms) / len(self.terms)
        bm25 = 0.0
        maximum = 0.0
        for term in self.query:
            idf = math.log(1 + (self.size - self.df[term] + 0.5) / (self.df[term] + 0.5))
            maximum += idf * 2.2
            tf = doc[term]
            bm25 += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * sum(doc.values()) / self.average))
        lexical = bm25 / maximum if maximum else 0.0
        score = min(1.0, 0.5 * coverage + 0.3 * title + 0.2 * lexical)
        # Exact task phrases take precedence over incidental overlaps such as "detection".
        if not anchor:
            score *= 0.15
        return score, [h for h in hits if h], anchor
