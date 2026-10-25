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
- 查询成功但返回 0 篇属于“未匹配”，不计为来源失败；超时、网络故障和限流才记为请求失败。arXiv 的中文原句可能没有匹配，系统仍保留英文改写召回的文献。
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

模型配置支持两个用途，密钥只在本地读取：

- `PAPER_AGENT_LLM_API_KEY`：查询规划，生成英文专业术语和同义表达；同时保留中文查询。缺失或失败时使用内置双语词典，无法翻译的主题会明确告警。
- `OPENAI_API_KEY`：单篇总结和综合报告优先使用此密钥；未填写时复用 `PAPER_AGENT_LLM_API_KEY`。两者都缺失或调用失败时，使用抽取式总结及注明限制的模板报告。
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

## 检索质量与证据保留

- 默认按运行日滚动检索近5年，近2年文献优先；按约70%/30%分配近2年与其余3年的召回预算，日期范围互不重叠。每个请求都携带日期筛选，默认不扩展到五年前；明确指定历史日期时保留指定范围。相关候选充足时，候选池与最终结果至少70%来自近2年；不足时使用范围内的其他相关论文补充，避免凑入无关论文。
- arXiv 使用 `submittedDate` 和 `submittedDate` 倒序，避免把旧论文更新日期当作新发表；OpenAlex 使用发表日期过滤及 `publication_date:desc`；Europe PMC 使用 `FIRST_PDATE` 与 `sort_date:y`；Semantic Scholar 使用支持 `publicationDateOrYear` 与 `publicationDate:desc` 的 `/paper/search/bulk`（服务端固定批次，客户端只保留本次配额）。Crossref 支持 `published` 倒序，但泛词查询按日期排列容易失去主题准确度，因此默认使用发表日期过滤后在时间窗口内按相关性检索。
- 查询规划和逐来源记录会展示时间窗口、日期范围、排序方式及近2年实际召回数量。API 返回的日期仍可能缺失或不一致；未知日期文献保留在元数据中，不用于凑近2年配额。
- 中文主题会同时检索中文原词和英文改写。每个来源分别运行多条查询：arXiv 使用短语、OpenAlex 使用短语匹配、Crossref 使用 bibliographic 检索、Europe PMC 使用标题/摘要字段、Semantic Scholar 使用相关性检索。界面展示实际查询、语言、条数和失败原因。
- 检索采用最多6条英文核心词与同义短语，中文输入同时保留中文主题。每条表达独立请求各数据库，结果取并集去重，避免同义词被错误地全部 AND。缩写优先展开；场景文本主题下 OCR 加入自然场景限定。“查询规划”显示实际采用词、改写和未采用原因，“多源数据库检索”记录各词请求状态及返回数量；候选同义词列表不代表全部已执行。扩展会增加请求次数，每个来源最终保留数量仍受侧栏设置限制。
- 预排序优先匹配任务与研究对象，结合中英同义词、标题权重和候选集 BM25，避免仅因命中“检测”等通用词或引用数高而入选。相关性不足时允许少于 Top-k；默认没有加载语义重排模型。
- 全文阶段保留所有公开 PDF 候选，尝试 OpenAlex/Unpaywall、arXiv 和公开落地页中的 PDF 地址；首个地址失败可换用其他公开副本，每篇最多尝试四个地址并限制总耗时。登录限制、付费墙和来源限流仍可能导致失败。
- 全文失败仍保留文献元数据，允许用摘要继续分析；没有摘要时仅保留元数据并说明无法核实方法。即使选择“必须有全文”，被排除的条目仍能在“全部文献元数据”中查看和导出。
- 综合报告正文为 **300–500 字中文综述**（按非空白字符计数），附编号引文和独立参考文献。正文仅根据实际材料综合，模型失败会明确注明回退；无相关文献时说明证据不足，不编造综述。

可复现实测（会联网，默认会调用已配置模型）：

```powershell
python scripts/verify_research_quality.py --idea "场景文本检测与识别"
# 禁用模型，仅检验真实检索、相关性排序与全文处理：
python scripts/verify_research_quality.py --no-llm
```

结果保存在 `data/quality-evaluation/`，包含九阶段输出、逐条查询、全文尝试记录和保留的元数据。小样本实测不替代多主题的人工相关性评估。

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


## 本地实时进度展示

完整研究流程使用九阶段导航和总进度条，区分已完成、进行中、待执行与失败。
点击已完成或进行中的阶段，下方只展示该阶段的状态、耗时、输入输出数量与结果快照。
默认跟随当前进度；手动选择后固定查看，点击“跟随当前进度”可恢复自动切换。
研究任务在后台继续执行，切换详情或操作页面不会重复启动检索；运行期间禁用重复提交。
检索部分分别列出实际查询的数据库、检索式、文献题目、作者、年份、DOI、来源链接与摘要；
每个来源完成后立即显示结果，不必等待最慢来源。全文处理会逐篇展示成功、摘要降级或排除原因。
后续阶段失败时，之前已完成的中间结果仍可查看，并包含在完整 JSON 导出中。

`PipelineResult.events` 保持九个阶段汇总的原有契约；`progress_events` 保存完整的阶段、
数据库及单篇文献事件时间线。总耗时只统计阶段汇总，不重复累计并发子任务耗时。
