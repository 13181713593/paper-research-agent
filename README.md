# MCP多源论文检索与总结智能体

这是一个接口先行、可运行的课程项目骨架。它已经完成：

- MCP Python SDK v2 Server、主 Agent 门面与结构化工具输出；
- Streamlit 页面和端到端进度展示；
- 完整论文领域模型、空值策略、来源追踪和评分拆解；
- 多源并发、部分失败、全文失败降级和候选补位所需的编排框架；
- 单一依赖装配入口，1–3 组实现无需修改 UI、MCP 或主流程；
- 检索、去重、下载解析、证据确认、两级排序、单篇总结、综合报告接口；
- 离线 Demo 适配器与基础测试。

真实论文 API、OCR/Markdown 解析器、学术重排模型和 LLM 报告器是各小组要补齐的实现。当前 Demo 数据带有醒目标记，不会冒充真实检索结果。

## 流程：

> 查询规划 → 多源并发检索 → 合并去重 → 元数据预排序取 N → 并发全文处理 → 证据确认 → 最终重排取 k → 单篇总结 → 综合报告

完整问题清单见 [docs/ASSIGNMENT_REVIEW.md](docs/ASSIGNMENT_REVIEW.md)，

总体架构见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)，

组间接口见 [docs/API_CONTRACTS.md](docs/API_CONTRACTS.md)。
模型空值规范见 [docs/DATA_MODEL.md](docs/DATA_MODEL.md)，

各组交付检查见[docs/INTEGRATION_CHECKLIST.md](docs/INTEGRATION_CHECKLIST.md)，

Host 配置见[docs/MCP_HOST_CONFIG.md](docs/MCP_HOST_CONFIG.md)。

## 环境要求

- Anaconda 或 Miniconda；
- 推荐使用 Python 3.11；
- 所有命令均在项目根目录 `paper-research-agent` 中执行。

本项目使用 Conda 管理 Python 环境，通过环境内的 `pip` 对当前项目进行可编辑安装。
可编辑安装是 `src/` 目录布局正常导入 `paper_agent` 所必需的步骤。

## 使用 Conda 创建环境（推荐）

### 1. 进入项目目录

PowerShell：

```powershell
cd paper-research-agent
```

### 2. 根据 `environment.yml` 创建环境

```powershell
conda env create -f environment.yml
conda activate paper-agent
```

`environment.yml` 会创建 Python 3.11 环境，并在该环境中执行可编辑安装
`pip install -e ".[dev]"`。

如果环境已经存在，需要同步新增依赖，可以执行：

```powershell
conda activate paper-agent
conda env update -f environment.yml --prune
```

### 3. 验证安装

```powershell
python -c "import paper_agent; print(paper_agent.__file__)"
python -c "import streamlit, mcp; print('依赖安装成功')"
```

两条命令都正常执行后，说明当前 Conda 环境和项目包安装正确。

## 使用已有 Conda 环境

如果希望继续使用已有的 `LLM2` 环境，不需要重新创建环境：

```powershell
conda activate LLM
cd paper-research-agent
python -m pip install -e ".[dev]"
```

注意：必须在项目根目录执行 `pip install -e .`。

## 运行项目

### 运行 Streamlit 页面

```powershell
conda activate paper-agent
python -m streamlit run .\src\paper_agent\ui\app.py
```

浏览器默认访问：`http://localhost:8501`。

建议使用 `python -m streamlit`，而不是直接使用 `streamlit`，以确保 Streamlit
来自当前激活的 Conda 环境。

### 运行测试

```powershell
conda activate paper-agent
python -m pytest
```

### 使用 MCP Inspector

```powershell
conda activate paper-agent
mcp dev .\src\paper_agent\mcp_server.py
```

### 通过 stdio 启动 MCP Server

```powershell
conda activate paper-agent
python -m paper_agent.mcp_server
```

## 常见问题

## 当前 MCP 工具

`get_pipeline_contract()` 返回可供主 Agent/联调程序读取的阶段契约。

`research_papers(research_idea, top_k, pre_rank_pool_size, require_fulltext)` 返回：

- Top-k 标准化论文；
- 评分解释与来源记录；
- 单篇论文总结；
- Markdown 综合报告；
- 每阶段耗时、输入/输出量、警告和最终状态。

## 接入真实组件

1. 阅读 `ports/contracts.py` 与 `docs/API_CONTRACTS.md`。
2. 在 `adapters/` 中实现对应 Protocol。
3. 新建生产装配函数，例如 `build_production_dependencies(settings)`。
4. 只修改 `paper_agent/bootstrap.py` 的生产装配分支；MCP 和 Streamlit 无需改动。
5. 每个适配器补齐单元测试，并使用固定查询集做端到端评测。

## 数据模型

`Paper` 包括题目、作者、单位、日期、出版状态、venue、摘要、评论、arXiv ID、DOI、OpenAlex/Semantic Scholar ID、BibTeX、项目页、数据集、PDF/OA 状态、代码开源状态、引用数、来源记录、别名、冲突、解析质量和分项分数。模型同时区分：

- 元数据声称的 `is_open_access`；
- 本次任务实际验证的 `fulltext.access_status`；
- 有 PDF URL 与是否成功下载/解析；
- 全文总结与摘要降级总结。

## 目录结构

```text
src/paper_agent/
├── application/      # 主 Agent 门面与确定性流水线
├── domain/           # 共享实体、枚举、错误类型
├── ports/            # 1–3 组必须实现的 Protocol 接口
├── adapters/         # Demo 及各组真实实现
├── ui/               # Streamlit，仅调用主 Agent
├── bootstrap.py      # 唯一依赖装配入口
├── config.py         # 环境配置
└── mcp_server.py     # MCP v2 工具边界
```

## 版本说明

项目按 2026 年官方 MCP Python SDK v2 接口编写，并将依赖限制为 `mcp>=2,<3`。不要继续将 2024-11-05 规范和早期 SDK 示例作为唯一实现依据。
