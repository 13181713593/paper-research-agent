# 多源论文检索与总结智能体

课程项目，采用共享数据模型、Protocol 接口和统一依赖装配，支持 Streamlit 与 MCP 接入。

## 功能与运行模式

完整流程：查询规划 → 五源检索 → 去重 → 预排序 → 论文下载 → 转换为 Markdown → 证据确认（并入最终重排） → 最终重排 → 单篇总结 → 综合报告。

| `PAPER_AGENT_ENV` | 实际启用的组件 |
|---|---|
| `demo`（未配置时默认） | 全部使用离线演示组件 |
| `retrieval` | 第一组真实检索与去重，其余阶段为演示组件 |
| `ranking` | 第二组真实排序，其余阶段为演示组件 |
| `production`（推荐） | 第一组真实检索与去重、第二组排序、第三组下载解析及总结报告 |

- 五个检索源：arXiv、Crossref、OpenAlex、Semantic Scholar、Europe PMC；单来源失败保留警告，其余来源继续。
- 查询成功但返回 0 篇属于“未匹配”，不计为来源失败；超时、网络故障和限流才记为请求失败。arXiv 的中文原句可能没有匹配，系统仍保留英文改写召回的文献。
- 第三组独立工作台：每批支持 1–100 篇 arXiv 论文，元数据分批获取，下载与总结并发可设为 1–8；不执行主题检索或排名。
- 完整流程支持 Top-k 1–100、全文候选池 1–200；下载和总结均遵守 `PAPER_AGENT_MAX_CONCURRENCY`（默认 4，最高 32）。扩大数量不会放开并发。
- 支持单篇和批量原始 PDF、全文与图片资源 ZIP，以及单篇总结和综合报告 Markdown 下载；已下载但未进入 Top-k 的候选也可导出，解析失败仍保留原始 PDF。下载与预览就近放在对应阶段：05 论文下载提供批量与候选 PDF，06 转换为 Markdown 提供整包 ZIP 与逐篇预览/下载，07–09 分别展示最终结果、单篇总结与综合报告。
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

仓库根目录的 `.env` 是**唯一凭证来源**，启动时自动读取，不再从其他项目补加载配置。

```bash
cp .env.example .env
python scripts/setup_env.py        # 列出缺失/为空的键、申请入口与缺失后的降级行为
```

自检脚本只读本地文件、不联网、不打印密钥值；`--strict` 在缺少模型 Key 时返回非零退出码，可用于 CI。配置写入后需重启服务生效。

### 凭证清单

| 变量 | 级别 | 用途 | 申请入口 | 缺失时 |
| --- | --- | --- | --- | --- |
| `OPENAI_API_KEY` | 必需 | 单篇总结与综合报告；查询规划复用它 | [DeepSeek](https://platform.deepseek.com/api_keys) / [OpenAI](https://platform.openai.com/api-keys) | 改用 `PAPER_AGENT_LLM_API_KEY`；两者皆空则抽取式总结 + 模板报告 |
| `PAPER_AGENT_LLM_API_KEY` | 可选 | 查询规划专用 Key | 同上 | 内置双语词典规划 |
| `PAPER_AGENT_LLM_BASE_URL` / `PAPER_AGENT_LLM_MODEL` | 默认 | OpenAI-compatible 地址与模型 | DeepSeek 填 `https://api.deepseek.com` + `deepseek-flash` | 缺省为 OpenAI 地址与 `gpt-4o-mini`，与 DeepSeek 不匹配会调用失败 |
| `MINERU_API_KEY` | 推荐 | 公式、表格与版面还原 | [mineru.net](https://mineru.net) → API 管理 → 新建 Token | `auto` 时退回 `pypdf`，公式无法可靠还原 |
| `UNPAYWALL_EMAIL` | 推荐 | Unpaywall 公开全文定位，需真实邮箱 | [unpaywall.org](https://unpaywall.org/products/api) | 跳过该渠道 |
| `OPENALEX_API_KEY` | 推荐 | OpenAlex 元数据与全文定位配额 | [openalex.org](https://openalex.org/settings/api-key) | 匿名调用，限额更低 |
| `PAPER_AGENT_CONTACT_EMAIL` | 可选 | Crossref / OpenAlex 礼貌池 | 填常用邮箱 | 匿名调用，限流更严 |
| `SEMANTIC_SCHOLAR_API_KEY` | 可选 | Semantic Scholar 检索配额 | [申请页](https://www.semanticscholar.org/product/api#api-key-form) | 匿名调用，易 429 |

其余带默认值的变量（并发、超时、下载上限、重排模型等）见 [.env.example](.env.example)，`python scripts/setup_env.py` 会逐条说明用途。

**分发方式**：`.env` 已被 `.gitignore` 排除，模板中没有任何真实密钥；合作者 clone 后由项目负责人通过私有渠道（私聊、团队网盘、密码管理器）提供一份 `.env`，或各自按上表申请自己的 Key。密钥不得提交到 Git，一旦进入提交历史即视为泄露，需要轮换。

`auto` 配置了 `MINERU_API_KEY` 时会将 PDF 上传到 MinerU，识别公式、表格和版面并保存图片；无 Key 时使用本地 `pypdf`。工作台也可直接选择后端。`pypdf` 会合并文本硬换行、提取嵌入位图，但不能可靠恢复公式、扫描正文或矢量图；仅规范化已有 LaTeX，不编造缺失公式。

全文资源 ZIP 应完整解压，保持 Markdown 与图片目录相对位置；单独 `.md` 内嵌图片，阅读器不支持 data URI 时请使用资源 ZIP。批量 ZIP 总大小限制为 200 MB，超限时请分篇下载。

已有环境变量优先（包括显式空值），因此不要用 `export` 预注入空值，否则会遮住 `.env` 里的真实值；`.env`、`data/`、`artifacts/`、虚拟环境均不提交到 Git。

模型有两个用途，密钥只在本地读取：

- `PAPER_AGENT_LLM_API_KEY`（缺省复用 `OPENAI_API_KEY`）：查询规划，生成英文专业术语和同义表达，同时保留中文查询；缺失或失败时使用内置双语词典，无法翻译的主题会明确告警。
- `OPENAI_API_KEY`：单篇总结和综合报告的主 Key；两者都缺失或调用失败时，使用抽取式总结及注明限制的模板报告。
- 两者共用 `PAPER_AGENT_LLM_BASE_URL`、`PAPER_AGENT_LLM_MODEL`，DeepSeek 与 OpenAI-compatible 示例见 [.env.example](.env.example)。

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
- 全文失败仍保留文献元数据，允许用摘要继续分析；没有摘要时仅保留元数据并说明无法核实方法。即使选择“必须有全文”，被排除的条目仍能在“05 论文下载 · 全部文献元数据”中查看和导出。
- 综合报告正文为 **1000–1500 字中文综述**（按非空白字符计数），附编号引文和独立参考文献。正文仅根据实际材料综合，模型失败会明确注明回退；无相关文献时说明证据不足，不编造综述。

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
每个来源完成后立即显示结果，不必等待最慢来源。“论文下载”逐篇展示是否找到公开全文与取得渠道（直链/Unpaywall/OpenAlex/落地页）并给出候选池进度条；“转换为 Markdown”逐篇展示解析器、质量与失败原因。
后续阶段失败时，之前已完成的中间结果仍可查看，并包含在完整 JSON 导出中。

`PipelineResult.events` 保持九个阶段汇总的原有契约；`progress_events` 保存完整的阶段、
数据库及单篇文献事件时间线。总耗时只统计阶段汇总，不重复累计并发子任务耗时。
