# 多源论文检索与总结智能体

课程项目，采用共享数据模型、Protocol 接口和统一依赖装配，支持 Streamlit 与 MCP 接入。

## 当前功能与边界

- 第三组独立工作台：输入 arXiv 编号，获取元数据、下载公开全文、多进程解析 Markdown、并发生成单篇总结与综合报告。
- 支持全文、单篇总结、综合报告的 Markdown 下载及 ZIP 打包；提供引文文字核对、跨论文对比和串行/多进程解析实验。
- 第二组两级排序已实现，支持可选 CrossEncoder 和规则降级，详见[排序说明](docs/RANKING.md)。
- 完整流程仍使用演示检索器；`production` 仅替换第三组处理组件，不代表已接入真实多源检索。

完整流程：查询规划 → 多源检索 → 去重 → 预排序 → 全文处理 → 证据确认 → 最终重排 → 单篇总结 → 综合报告。

## 安装

需要 Python 3.11 或以上；以下命令均在仓库根目录执行。

```bash
python -m venv .venv
```

激活环境：macOS/Linux 使用 `source .venv/bin/activate`，Windows PowerShell 使用 `.venv\Scripts\Activate.ps1`。然后安装：

```bash
python -m pip install -e ".[dev,processing]"
```

也可使用现有 Conda 配置：`conda env create -f environment.yml`，然后 `conda activate paper-agent`；无需同时创建两套环境。

## 配置与启动

首次使用时复制 `.env.example` 为 `.env`，已有配置不要覆盖。`.env`、本地数据及虚拟环境均不提交到 Git。

推荐第三组配置：

```dotenv
PAPER_AGENT_ENV=production
PAPER_AGENT_MARKDOWN_BACKEND=pypdf
```

大模型配置使用 `OPENAI_API_KEY`、`PAPER_AGENT_LLM_BASE_URL`、`PAPER_AGENT_LLM_MODEL`；DeepSeek 示例见 [.env.example](.env.example)。密钥只填在本地 `.env`。

`.env` 不会自动加载，使用下面的命令显式加载并启动：

```bash
python -m dotenv -f .env run -- python -m streamlit run src/paper_agent/ui/app.py
```

浏览器打开 `http://localhost:8501`，选择“第三组独立演示”，填写研究想法和 arXiv 编号（如 `1706.03762`、`1409.0473`），点击“处理论文并生成报告”。更改模型配置后需重启并重新生成。

## 结果说明

- **总结模式**：独立工作台及 `production` 处理组件在有 Key 时优先调用模型，失败后降级；无 Key 时使用抽取式总结和模板报告。页面入口不是模型选择开关。
- **元数据**：作者、摘要等来自 arXiv；未返回的机构等字段不猜测。独立入口不做排名，因此不显示排名分数。
- **并发进度**：“已处理”是累计数量，“第几篇”是输入序号；日志按完成先后出现，最终结果保持输入顺序。
- **全文与缓存**：首次按编号联网下载，重复任务可复用本地下载/解析缓存；缺少全文时的摘要总结会注明降级。
- **验证边界**：解析完整度不是论文质量；引文文字匹配不保证结论正确。性能实验只比较本地 PDF 解析，不包含下载或模型调用。

## 测试

```bash
python -m pytest
python -m ruff check src tests
```

真实联网和付费模型测试默认跳过，统一保留在 `tests/test_processing_workbench.py`：

- `test_public_arxiv_workbench_smoke`：显式设置 `RUN_NETWORK_SMOKE=1` 后运行；下载两篇公开论文，测试中禁用模型 API。
- `test_live_llm_summary_and_report_without_fallback`：显式设置 `RUN_LLM_SMOKE=1` 并加载模型配置后运行；会消耗 API 额度，且不允许将规则回退算作成功。

## 目录与整合文档

```text
src/paper_agent/
├── adapters/       # 检索演示、排序及第三组真实适配器
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

组间复用公共模型，在 `bootstrap.py` 接线；不要为独立展示复制或改变主流水线。MCP 使用 `mcp>=2,<3`，与 Streamlit 共享主 Agent 门面。
