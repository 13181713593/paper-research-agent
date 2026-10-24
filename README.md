# 多源论文检索与总结智能体

课程项目，采用共享数据模型、Protocol 接口和统一依赖装配，支持 Streamlit 与 MCP 接入。

## 功能与运行模式

完整流程：查询规划 → 五源检索 → 去重 → 预排序 → 全文处理 → 证据确认 → 最终重排 → 单篇总结 → 综合报告。

| `PAPER_AGENT_ENV` | 实际启用的组件 |
|---|---|
| `demo`（未配置时默认） | 全部使用离线演示组件 |
| `retrieval` | 第一组真实检索与去重，其余阶段为演示组件 |
| `ranking` | 第二组真实排序，其余阶段为演示组件 |
| `production`（推荐） | 第一组真实检索与去重、第二组排序、第三组下载解析及总结报告 |

- 五个检索源：arXiv、Crossref、OpenAlex、Semantic Scholar、Europe PMC；单来源失败保留警告，其余来源继续。
- 第三组独立工作台：按 arXiv 编号获取元数据、下载公开全文、多进程解析、并发总结，不执行主题检索或排名。
- 支持全文、单篇总结、综合报告的 Markdown 下载及 ZIP 打包，包含引文文字核对、跨论文对比和本地解析性能实验。
- 第二组支持可选 CrossEncoder，未配置模型时使用规则降级，不代表已启用语义模型，详见[排序说明](docs/RANKING.md)。

## 安装

需要 Python 3.11 或以上；以下命令均在仓库根目录执行。

```bash
python -m venv .venv
```

激活环境：macOS/Linux 使用 `source .venv/bin/activate`，Windows PowerShell 使用 `.venv\Scripts\Activate.ps1`，然后安装：

```bash
python -m pip install -e ".[dev,processing]"
```

也可使用 `conda env create -f environment.yml` 和 `conda activate paper-agent`；无需同时创建两套环境。

## 配置与启动

首次使用时复制 `.env.example` 为 `.env`，已有配置不要覆盖。推荐保留模板中的 `PAPER_AGENT_ENV=production` 和 `PAPER_AGENT_MARKDOWN_BACKEND=pypdf`。

入口自动读取当前工作目录的 `.env`，已有环境变量优先（包括显式空值）；`.env`、数据、虚拟环境均不提交到 Git。

模型配置分为两种用途，互不自动复制 Key：

- `PAPER_AGENT_LLM_API_KEY`：查询规划，把研究想法转为英文检索关键词；缺失或失败时使用启发式规划，中文召回能力有限。
- `OPENAI_API_KEY`：单篇总结和综合报告；缺失时使用抽取式总结及模板报告，模型调用失败时降级。
- 两者共用 `PAPER_AGENT_LLM_BASE_URL`、`PAPER_AGENT_LLM_MODEL`。OpenAI-compatible 与 DeepSeek 配置见 [.env.example](.env.example)，实际 Key 只填本地配置。
- `SEMANTIC_SCHOLAR_API_KEY`、`PAPER_AGENT_CONTACT_EMAIL`、`UNPAYWALL_EMAIL` 是检索/OA 定位可选配置；未配置的公开服务可能限流。

```bash
python -m streamlit run src/paper_agent/ui/app.py
```

也兼容 `python -m dotenv -f .env run -- python -m streamlit run src/paper_agent/ui/app.py`。

浏览器打开 `http://localhost:8501`，选择“完整研究流程”输入研究想法，或选择“第三组独立演示”填写 arXiv 编号（如 `1706.03762`、`1409.0473`）。更改配置后需重启并重新生成结果。

## 结果说明

- 独立入口从 arXiv 获取题名、作者和摘要，未返回的机构等字段不猜测；该入口不显示排名分数。
- 独立入口进度中的“已处理”是累计数量，“第几篇”是输入序号；日志按完成先后出现，最终结果保持输入顺序。
- 首次联网下载，重复任务可复用下载/解析缓存；缺少全文时的摘要总结会注明降级。
- 解析完整度不代表论文质量；引文文字匹配不保证结论正确。未来研究方向为系统推断。
- 多进程性能实验只比较本地 PDF 解析，不包含检索、下载或模型调用。

## 测试

```bash
python -m pytest
python -m ruff check src tests
```

离线测试隔离本机配置与密钥。真实联网及付费模型测试默认跳过，需显式开启：

- `RUN_NETWORK_SMOKE=1`：运行第三组两篇公开论文冒烟和 production 检索到报告冒烟，测试禁用模型 API。
- `RUN_LLM_SMOKE=1`：运行真实总结与报告测试，须显式加载模型配置，会消耗 API 额度，不允许将规则回退算作成功。
- 第一组 `scripts/` 中的检索验证脚本保留，手动运行会访问真实来源。

## 目录与整合文档

```text
src/paper_agent/
├── adapters/       # 检索、排序、下载、解析与总结适配器
├── application/    # 主流程、第三组工作台与性能实验
├── domain/         # 共享模型与错误类型
├── ports/          # 各组 Protocol 接口
├── ui/             # Streamlit 与 Markdown 导出
├── bootstrap.py    # 统一依赖装配
├── config.py       # 环境配置
└── mcp_server.py   # MCP 工具入口
```

- [总体架构](docs/ARCHITECTURE.md)
- [接口契约与联调清单](docs/API_CONTRACTS.md)
- [数据模型与空值规范](docs/DATA_MODEL.md)
- [MCP Host 配置](docs/MCP_HOST_CONFIG.md)
- [第二组排序说明](docs/RANKING.md)

组间复用公共模型，在 `bootstrap.py` 接线，不在 UI/MCP 复制业务编排。MCP 使用 `mcp>=2,<3`，通过 `python -m paper_agent.mcp_server` 启动 stdio 服务，与 Streamlit 共享主 Agent 门面。
