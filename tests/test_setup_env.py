"""凭证自检脚本的离线单元测试。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_setup_env():
    spec = importlib.util.spec_from_file_location("setup_env", ROOT / "scripts" / "setup_env.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # dataclass 在 string 注解下需要能从 sys.modules 找到模块本身
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_read_env_skips_comments_and_normalizes_values(tmp_path):
    setup_env = load_setup_env()
    path = tmp_path / ".env"
    path.write_text("# 注释\nOPENAI_API_KEY='sk-test'\n\nEMPTY=\n  PADDED = 1 \n", encoding="utf-8")

    env = setup_env.read_env(path)

    assert env == {"OPENAI_API_KEY": "sk-test", "EMPTY": "", "PADDED": "1"}


def test_read_env_returns_empty_for_missing_file(tmp_path):
    setup_env = load_setup_env()

    assert setup_env.read_env(tmp_path / "absent.env") == {}


def test_key_status_separates_missing_from_empty(tmp_path):
    setup_env = load_setup_env()

    assert setup_env.key_status({"A": "value"}, "A") == setup_env.SET
    assert setup_env.key_status({"A": ""}, "A") == setup_env.EMPTY
    assert setup_env.key_status({}, "A") == setup_env.MISSING


def test_report_marks_capabilities_and_missing_required_keys(tmp_path):
    setup_env = load_setup_env()
    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY=sk-test\nMINERU_API_KEY=\n", encoding="utf-8")

    lines, ready = setup_env.build_report(env_path, ROOT / ".env.example")
    text = "\n".join(lines)

    assert ready is True
    assert "[可用] 查询规划与单篇总结" in text
    assert "[降级] MinerU 公式与版面解析" in text
    assert "MINERU_API_KEY（推荐）" in text
    assert "未出现" in text
    assert "sk-test" not in text


def test_report_degrades_without_any_model_key(tmp_path):
    setup_env = load_setup_env()
    env_path = tmp_path / ".env"
    env_path.write_text("UNPAYWALL_EMAIL=someone@example.com\n", encoding="utf-8")

    lines, ready = setup_env.build_report(env_path, ROOT / ".env.example")

    assert ready is False
    assert any("[降级] 查询规划与单篇总结" in line for line in lines)
    assert any("OPENAI_API_KEY（必需）" in line for line in lines)


def test_main_exit_codes(tmp_path, capsys):
    setup_env = load_setup_env()
    env_path = tmp_path / ".env"

    assert setup_env.main(["--env", str(env_path)]) == 2
    assert "未找到" in capsys.readouterr().out

    env_path.write_text("MINERU_API_KEY=token\n", encoding="utf-8")

    assert setup_env.main(["--env", str(env_path), "--template", str(ROOT / ".env.example")]) == 0
    assert (
        setup_env.main(
            ["--env", str(env_path), "--template", str(ROOT / ".env.example"), "--strict"]
        )
        == 1
    )
