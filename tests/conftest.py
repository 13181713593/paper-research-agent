"""默认测试隔离开发者配置；只有明确选择的联网用例可访问外部服务。"""

import os

import httpx
import pytest
import requests


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path, monkeypatch, request):
    # 付费冒烟仍须独立设置开关并显式注入凭证，不自动读取开发者 .env。
    paid_smoke = (
        request.node.name == "test_live_llm_summary_and_report_without_fallback"
        and os.getenv("RUN_LLM_SMOKE") == "1"
    )
    monkeypatch.chdir(tmp_path)
    if not paid_smoke:
        for key in tuple(os.environ):
            if key.startswith("PAPER_AGENT_") or key in {
                "OPENAI_API_KEY",
                "MINERU_API_KEY",
                "SEMANTIC_SCHOLAR_API_KEY",
                "UNPAYWALL_EMAIL",
                "CROSSREF_MAILTO",
            }:
                monkeypatch.delenv(key)
        for key in ("OPENAI_API_KEY", "PAPER_AGENT_LLM_API_KEY", "MINERU_API_KEY"):
            monkeypatch.setenv(key, "")
    network_smoke = os.getenv("RUN_NETWORK_SMOKE") == "1" and request.node.name in {
        "test_public_arxiv_workbench_smoke",
        "test_production_live_smoke",
    }
    if not paid_smoke and not network_smoke:

        def no_network(*args, **kwargs):
            raise AssertionError("离线测试禁止真实网络；请注入 mock transport")

        monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", no_network)
        monkeypatch.setattr(httpx.HTTPTransport, "handle_request", no_network)
        monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", no_network)
