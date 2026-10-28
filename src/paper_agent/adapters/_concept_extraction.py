"""Concept extraction: turn a one-sentence research idea into searchable terms.

This module holds the heuristic (offline, deterministic) extractor. It is the
default for ``HeuristicQueryPlanner`` and doubles as the fallback for the
optional LLM extractor, mirroring the degrade-on-failure pattern used in
LLM-assisted ingestion pipelines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Chinese -> English academic terms. Longest match wins ("数学推理" beats "推理").
_ZH_EN_TERMS: dict[str, str] = {
    "大模型智能体": "large language model agents",
    "大语言模型智能体": "large language model agents",
    "大模型": "large language model",
    "无人机导航": "unmanned aerial vehicle navigation",
    "智能控制": "intelligent control",
    "云边协同": "cloud edge computing",
    "云边协作": "cloud edge computing",
    "边缘计算": "edge computing",
    "云计算": "cloud computing",
    "任务卸载": "task offloading",
    "计算卸载": "computation offloading",
    "计算资源调度": "computational resource scheduling",
    "资源调度": "resource scheduling",
    "资源分配": "resource allocation",
    "场景文本检测": "scene text detection",
    "场景文本识别": "scene text recognition",
    "场景文本": "scene text",
    "文字检测": "text detection",
    "文字识别": "text recognition",
    "文本检测": "text detection",
    "文本识别": "text recognition",
    "光学字符识别": "optical character recognition",
    "视觉语言导航": "vision language navigation",
    "无人机": "unmanned aerial vehicle",
    "长期记忆": "long term memory",
    "失败恢复": "failure recovery",
    "世界模型": "world model",
    "目标检测": "object detection",
    "图像分割": "image segmentation",
    "图神经网络": "graph neural network",
    "深度强化学习": "deep reinforcement learning",
    "强化学习": "reinforcement learning",
    "数学推理": "mathematical reasoning",
    "推理能力": "reasoning",
    "大语言模型": "large language model",
    "大型语言模型": "large language model",
    "语言模型": "language model",
    "小语言模型": "small language model",
    "小模型": "small language model",
    "多智能体": "multi-agent",
    "智能体": "agent",
    "检索增强": "retrieval augmented generation",
    "知识图谱": "knowledge graph",
    "文本分类": "text classification",
    "情感分析": "sentiment analysis",
    "机器翻译": "machine translation",
    "代码生成": "code generation",
    "图像生成": "image generation",
    "指令微调": "instruction tuning",
    "微调": "fine-tuning",
    "人类反馈": "human feedback",
    "奖励模型": "reward model",
    "对齐": "alignment",
    "幻觉": "hallucination",
    "可解释性": "interpretability",
    "联邦学习": "federated learning",
    "对比学习": "contrastive learning",
    "提示学习": "prompt learning",
    "思维链": "chain of thought",
    "摘要": "summarization",
    "问答": "question answering",
    "推荐系统": "recommendation system",
    "异常检测": "anomaly detection",
    "时间序列": "time series",
}

_SYNONYM_GROUPS: list[list[str]] = [
    ["large language model agents", "LLM agents", "language agents"],
    ["unmanned aerial vehicle navigation", "UAV navigation", "drone navigation"],
    ["cloud edge computing", "cloud edge collaboration", "edge cloud computing"],
    ["task offloading", "computation offloading", "computational offloading"],
    [
        "scene text",
        "scene text detection",
        "scene text recognition",
        "text spotting",
        "optical character recognition",
        "OCR",
        "natural scene text",
        "end-to-end text spotting",
    ],
    ["vision language navigation", "VLN", "visual navigation"],
    ["unmanned aerial vehicle", "UAV", "drone"],
    ["graph neural network", "GNN", "graph neural networks"],
    ["reinforcement learning", "RL", "policy optimization"],
    ["mathematical reasoning", "math reasoning"],
    ["large language model", "LLM"],
    ["chain of thought", "CoT"],
    ["retrieval augmented generation", "RAG"],
    ["fine-tuning", "instruction tuning"],
    ["multi-agent", "agent collaboration"],
]

_STOPWORDS = {
    "please",
    "investigate",
    "compare",
    "different",
    "across",
    "strategies",
    "environment",
    "the",
    "a",
    "an",
    "of",
    "for",
    "in",
    "on",
    "with",
    "and",
    "or",
    "to",
    "how",
    "improve",
    "improving",
    "using",
    "use",
    "based",
    "via",
    "towards",
    "toward",
    "study",
    "research",
    "method",
    "methods",
    "approach",
    "利用",
    "提升",
    "提高",
    "研究",
    "基于",
    "通过",
    "如何",
    "什么",
}

_TOKEN_RE = re.compile(r"\b[56]G\b|[a-zA-Z]{2,}", re.IGNORECASE)


@dataclass
class ExtractedConcepts:
    """Normalized, provider-agnostic concept set shared by planners."""

    concepts: list[str] = field(default_factory=list)
    synonyms: list[str] = field(default_factory=list)

    def join(self) -> str:
        return " ".join(self.concepts)


def _synonyms_for(concepts: list[str]) -> list[str]:
    """Synonyms from any group sharing a word with an extracted concept."""
    synonyms: list[str] = []
    for group in _SYNONYM_GROUPS:
        for term in group:
            if term in concepts:
                synonyms.extend(t for t in group if t != term)
                break
    seen: set[str] = set()
    return [s for s in synonyms if not (s in seen or seen.add(s))]


class HeuristicConceptExtractor:
    """Offline rule-based extractor: zh-lexicon mapping + token filtering."""

    def extract(self, text: str) -> ExtractedConcepts:
        concepts: list[str] = []

        spans: list[tuple[int, int, str]] = []
        known_phrases = set(_ZH_EN_TERMS.values()) | {s for group in _SYNONYM_GROUPS for s in group}
        for phrase in known_phrases if len(text.split()) > 10 else ():
            if len(phrase.split()) < 2:
                continue
            expression = r"(?<!\w)" + re.escape(phrase).replace(r"\ ", r"[-\s]+") + r"(?!\w)"
            for match in re.finditer(expression, text, re.IGNORECASE):
                spans.append((match.start(), match.end(), phrase))
        for zh, en in _ZH_EN_TERMS.items():
            start = 0
            while (idx := text.find(zh, start)) != -1:
                spans.append((idx, idx + len(zh), en))
                start = idx + len(zh)
        # Longest matches first; skip spans overlapped by a strictly longer one.
        for start, end, en in sorted(spans, key=lambda s: (-(s[1] - s[0]), s[0], s[2])):
            if en in concepts:
                continue
            overlapped = any(
                not (end <= s or start >= e) and (e - s) > (end - start) for s, e, _ in spans
            )
            if overlapped:
                continue
            concepts.append(en)

        for token in _TOKEN_RE.findall(text):
            lower = token.lower()
            if lower in _STOPWORDS:
                continue
            if not any(lower in c.split() for c in concepts):
                concepts.append(lower)

        # Drop concepts whose every word already appears in a longer concept
        # (e.g. "reasoning" after "mathematical reasoning"), then cap the list.
        concepts = [
            c
            for c in concepts
            if not any(o != c and all(w in o.split() for w in c.split()) for o in concepts)
        ]
        concepts = concepts[:8]
        return ExtractedConcepts(concepts=concepts, synonyms=_synonyms_for(concepts))
