"""检查本地 .env 的凭证是否齐全，并给出申请入口与缺失影响。

只用标准库读取文件，不联网、不会打印任何密钥值。
用法：python scripts/setup_env.py [--strict] [--env PATH] [--template PATH]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SET, EMPTY, MISSING = "已设置", "空值", "缺失"


@dataclass(frozen=True, slots=True)
class Spec:
    """一个环境变量的用途、级别、申请入口与缺失影响。"""

    key: str
    level: str
    purpose: str
    apply: str
    fallback: str


SPECS: tuple[Spec, ...] = (
    Spec(
        "PAPER_AGENT_ENV",
        "默认",
        "运行模式：production 启用真实检索、下载与模型",
        ".env.example 已给 production 示例",
        "缺省为 demo，仅使用演示数据",
    ),
    Spec("PAPER_AGENT_DATA_DIR", "默认", "PDF、Markdown、图片与运行产物目录", "-", "缺省为 ./data"),
    Spec("PAPER_AGENT_MAX_CONCURRENCY", "默认", "检索并发上限（1-32）", "-", "缺省为 4"),
    Spec("PAPER_AGENT_REQUEST_TIMEOUT_SECONDS", "默认", "单次检索请求超时（秒）", "-", "缺省为 30"),
    Spec(
        "OPENAI_API_KEY",
        "必需",
        "单篇总结与综合报告的主 Key，同时供查询规划复用",
        "DeepSeek: https://platform.deepseek.com/api_keys；OpenAI: https://platform.openai.com/api-keys",
        "改用 PAPER_AGENT_LLM_API_KEY；两者都空则启发式规划 + 抽取式总结 + 模板报告",
    ),
    Spec(
        "PAPER_AGENT_LLM_API_KEY",
        "可选",
        "查询规划专用 Key，缺省复用 OPENAI_API_KEY",
        "同上",
        "内置双语词典生成查询，不调用模型",
    ),
    Spec(
        "PAPER_AGENT_LLM_BASE_URL",
        "默认",
        "OpenAI-compatible 接口地址",
        "DeepSeek 填 https://api.deepseek.com",
        "缺省为 https://api.openai.com/v1",
    ),
    Spec(
        "PAPER_AGENT_LLM_MODEL",
        "默认",
        "模型名",
        "DeepSeek 可填 deepseek-flash",
        "缺省为 gpt-4o-mini，与 DeepSeek 地址不匹配时会调用失败",
    ),
    Spec("PAPER_AGENT_LLM_TIMEOUT_SECONDS", "默认", "模型单次调用超时（秒）", "-", "缺省为 120"),
    Spec(
        "PAPER_AGENT_MARKDOWN_BACKEND",
        "默认",
        "解析后端：auto / mineru / pypdf",
        ".env.example 有说明",
        "缺省为 auto，有 MinerU Key 时优先 MinerU",
    ),
    Spec(
        "MINERU_API_KEY",
        "推荐",
        "公式、表格与版面还原（MinerU 精准解析）",
        "https://mineru.net 注册登录 → API 管理 → 新建 Token（每日 1000 页额度）",
        "auto 时退回 pypdf：只能提取文本与位图，公式无法可靠还原",
    ),
    Spec(
        "UNPAYWALL_EMAIL",
        "推荐",
        "Unpaywall 公开全文定位渠道，必须填真实邮箱",
        "https://unpaywall.org/products/api",
        "跳过该渠道，只走 arXiv、OpenAlex 与公开落地页",
    ),
    Spec(
        "OPENALEX_API_KEY",
        "推荐",
        "OpenAlex 元数据与全文定位配额",
        "https://openalex.org/signup → https://openalex.org/settings/api-key",
        "匿名调用，每日额度更低，批量检索易触发限流",
    ),
    Spec(
        "PAPER_AGENT_CONTACT_EMAIL",
        "可选",
        "Crossref / OpenAlex 礼貌池联系邮箱",
        "填常用邮箱即可",
        "匿名调用，速率限制更严格",
    ),
    Spec(
        "SEMANTIC_SCHOLAR_API_KEY",
        "可选",
        "Semantic Scholar 检索配额",
        "https://www.semanticscholar.org/product/api#api-key-form",
        "匿名调用，频繁请求会被限流或返回 429",
    ),
    Spec(
        "PAPER_AGENT_EMBEDDING_MODEL",
        "可选",
        "预排序使用的向量模型",
        "需额外安装 sentence-transformers",
        "只用关键词、同义词与 BM25 预排序",
    ),
    Spec(
        "PAPER_AGENT_RERANKER_MODEL",
        "可选",
        "CrossEncoder 最终重排模型",
        "需额外安装 sentence-transformers；首次使用需下载模型",
        "不加载语义重排，使用分项加权排序",
    ),
    Spec(
        "PAPER_AGENT_RERANKER_LOCAL_ONLY",
        "默认",
        "重排模型只读本地缓存",
        "-",
        "缺省为 1，不联网下载",
    ),
    Spec(
        "PAPER_AGENT_DOWNLOAD_TIMEOUT_SECONDS", "默认", "单篇 PDF 下载超时（秒）", "-", "缺省为 60"
    ),
    Spec("PAPER_AGENT_PDF_MAX_BYTES", "默认", "单个 PDF 大小上限（字节）", "-", "缺省 30000000"),
    Spec("PAPER_AGENT_DOWNLOAD_RETRY_LIMIT", "默认", "下载失败重试次数", "-", "缺省为 2"),
    Spec("PAPER_AGENT_DOWNLOAD_MAX_REDIRECTS", "默认", "下载允许的最大跳转次数", "-", "缺省为 5"),
)

CAPABILITIES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("查询规划与单篇总结", ("OPENAI_API_KEY", "PAPER_AGENT_LLM_API_KEY"), "任一 Key 非空即可"),
    ("MinerU 公式与版面解析", ("MINERU_API_KEY",), "缺失时退回 pypdf"),
    ("Unpaywall 全文渠道", ("UNPAYWALL_EMAIL",), "缺失时跳过该渠道"),
    ("OpenAlex 配额", ("OPENALEX_API_KEY",), "缺失时匿名调用"),
)


def read_env(path: Path) -> dict[str, str]:
    """按 KEY=VALUE 解析，忽略注释与空行，去掉成对引号。"""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, _, value = text.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


def key_status(env: dict[str, str], key: str) -> str:
    if key not in env:
        return MISSING
    return SET if env[key].strip() else EMPTY


def capability_rows(env: dict[str, str]) -> list[tuple[bool, str, str]]:
    rows = []
    for name, keys, note in CAPABILITIES:
        ready = any(env.get(key, "").strip() for key in keys)
        rows.append((ready, name, note))
    return rows


def build_report(env_path: Path, template_path: Path) -> tuple[list[str], bool]:
    env = read_env(env_path)
    template = read_env(template_path)
    lines = [
        f"配置文件：{env_path}",
        f"变量模板：{template_path}（{len(template)} 个变量）",
        "",
        "关键能力检查：",
    ]
    ready_all = True
    for ready, name, note in capability_rows(env):
        mark = "[可用]" if ready else "[降级]"
        detail = "" if ready else f" —— {note}"
        lines.append(f"  {mark} {name}{detail}")
        if not ready and name.startswith("查询规划"):
            ready_all = False

    set_keys = [s.key for s in SPECS if key_status(env, s.key) == SET]
    empty_keys = [s.key for s in SPECS if key_status(env, s.key) == EMPTY]
    missing_keys = [s.key for s in SPECS if key_status(env, s.key) == MISSING]
    lines += [
        "",
        f"已设置 {len(set_keys)} 个；空值 {len(empty_keys)} 个；未出现 {len(missing_keys)} 个。",
    ]

    for title, keys in (
        ("空值（程序视为已存在，会遮住同名的真实值）", empty_keys),
        ("未出现（使用默认值或降级）", missing_keys),
    ):
        if not keys:
            continue
        lines += ["", f"{title}："]
        for key in keys:
            spec = next(s for s in SPECS if s.key == key)
            lines.append(f"  - {key}（{spec.level}）：{spec.purpose}")
            if spec.level in {"必需", "推荐"}:
                lines.append(f"    申请：{spec.apply}")
                lines.append(f"    影响：{spec.fallback}")

    untouched = [key for key in env if key not in {s.key for s in SPECS}]
    if untouched:
        lines += ["", "模板之外的自定义变量：" + "、".join(sorted(untouched))]

    lines += [
        "",
        "修复方式：把 .env.example 复制为 .env 后补齐上面的键；写入后需重启服务才会生效。",
        "模板中没有真实密钥，.env 已被 .gitignore 排除，不要提交到 Git。",
    ]
    return lines, ready_all


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检查本地 .env 的凭证配置")
    parser.add_argument("--env", type=Path, default=ROOT / ".env")
    parser.add_argument("--template", type=Path, default=ROOT / ".env.example")
    parser.add_argument("--strict", action="store_true", help="缺少模型 Key 时返回非零退出码")
    args = parser.parse_args(argv)

    if not args.env.is_file():
        print(f"未找到 {args.env}，请先复制 .env.example 为 .env")
        return 2

    lines, ready = build_report(args.env, args.template)
    print("\n".join(lines))
    return 1 if args.strict and not ready else 0


if __name__ == "__main__":
    sys.exit(main())
