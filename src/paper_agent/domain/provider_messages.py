"""User messages describe observed service states without exposing HTTP payloads."""

import re

SOURCE_LABELS = {
    "arxiv": "arXiv",
    "openalex": "OpenAlex",
    "crossref": "Crossref",
    "semantic_scholar": "Semantic Scholar",
    "europe_pmc": "Europe PMC",
    "openreview": "OpenReview",
}


def provider_message(error, source=None):
    text = f"{type(error).__name__} {error}".lower()
    label = SOURCE_LABELS.get(source, source) or "此数据源"
    if any(s in text for s in ("ratelimit", "rate limit", "429", "cooldown")):
        reason = "暂时限制了访问频率，已暂停该来源的后续请求，请稍后再试"
    elif any(s in text for s in ("timeout", "timed out", "time budget", "超时")):
        reason = "未在等待时间内完成响应，已结束本轮等待"
    elif re.search(r"\b(401|403)\b", text):
        reason = "暂不允许访问，请检查该数据源的 API 密钥或访问权限"
    elif any(
        s in text
        for s in ("connection", "unreachable", "unavailable", "server error", "503", "502")
    ):
        reason = "暂时无法连接或服务繁忙，已结束本轮尝试"
    else:
        reason = "未能完成本次请求，具体原因需查看运行记录"
    return f"{label}{reason}。已取得的结果仍会保留。"


def display_message(message, error_code=None, source=None):
    if not message:
        return message
    text = f"{error_code or ''} {message}".lower()
    if any(
        s in text
        for s in (
            "ratelimit",
            "rate limit",
            "429",
            "timeout",
            "timed out",
            "connectionerror",
            "unreachable",
            "server error",
            "cooldown",
        )
    ):
        return provider_message(text, source)
    return message
