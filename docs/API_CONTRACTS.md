# 接口契约与组间联调规则

接口源码位于 `src/paper_agent/ports/contracts.py`，公共实体位于 `domain/models.py`。任何组不得在自己的模块重新定义 `Paper`。

## 通用约定

- 所有 I/O 接口均为 `async`；同步 SDK 由适配器用线程池封装。
- 组件不得吞掉异常。可恢复的单项失败由编排层转换为 warning；整体不可继续时抛异常。
- 外部字段未知时返回 `None`，禁止用 `"unknown"`、空字符串或虚构值代替。
- 所有分数归一化到 `[0, 1]`，并填写 `formula_version`。
- 每条来源数据写入 `source_records`；合并后也不得丢失。
- 真实 LLM 输出必须先通过 Pydantic 校验，失败最多修复/重试有限次数。

## 检索组

### `QueryPlanner.plan`

输入：用户研究想法、每源结果上限。  
输出：每个数据源对应的查询、同义词、纳入/排除标准。  
注意：不同 API 查询语法不同，不应把同一原始字符串直接广播。

### `PaperRetriever.search`

输入：`SearchPlan`。  
输出：标准化的元数据 `Paper[]`。  
不负责：PDF 下载、最终 Top-k、总结。  
验收：超时、429、空结果、字段缺失、非法日期均有测试。

### `BibTexMapper.to_paper`

输入：单条 BibTeX、来源名、可选补充元数据。  
输出：校验后的 `Paper`。  
注意：BibTeX 通常没有可靠摘要和单位，应使用提供方补充字段，不得从作者名推测单位。

### `PaperMerger.merge_and_deduplicate`

输入：全部来源的论文。  
输出：规范化、去重、合并后的论文。  
验收：DOI 格式差异、arXiv 版本号、标题标点差异、会议/期刊扩展版。

## 论文处理组

### `OpenAccessResolver.resolve` / `PdfDownloader.download` / `MarkdownParser.parse`

这三个细粒度接口分别负责合法 OA 地址解析、安全下载和 Markdown 转换；
`FullTextProcessor` 是流水线调用的组合门面。拆开后便于分别替换 MinerU、本地解析器或
对象缓存，也便于单元测试下载安全边界。

### `FullTextProcessor.process`

输入：单篇论文元数据。  
输出：更新 `fulltext` 的论文。  
处理链：OA 解析 → HEAD/GET → MIME/魔数校验 → 哈希缓存 → PDF 解析 → Markdown 质量检测。  
不得绕过付费墙；下载失败直接抛出带类型的异常，由编排器决定摘要降级或排除。

### `PaperSummarizer.summarize`

输入：研究想法 + 单篇论文。  
输出：固定栏目总结与证据引用。  
必须填写 `generated_from_fulltext`；摘要模式不得生成原文不存在的实验细节。

## 排序组

### `PreRanker.rank`

只使用低成本元数据证据，目标是高召回地选出全文处理池，不是最终结论。

### `EvidenceVerifier.verify`

根据全文或摘要判定研究对象、任务、方法是否满足纳入标准，并更新 `evidence_coverage`。不直接截断 Top-k。

### `FinalRanker.rank`

输入已确认候选，输出 Top-k。必须返回分项得分、公式版本与简短解释。时间和 venue 质量只能作为明确的小权重特征，不得把“新”或“高引用”直接等同于“相关”。

## 下载与总结组

### `ReportWriter.write`

只能基于 `PaperSummary` 与其 evidence 进行综合。未来方向需标注为系统推断，不应伪称论文作者结论。

## 建议目录命名

各组实现放在 `adapters/` 下，例如：

- `adapters/arxiv_retriever.py`
- `adapters/openalex_retriever.py`
- `adapters/mineru_processor.py`
- `adapters/bge_reranker.py`
- `adapters/llm_report_writer.py`

完成后只修改依赖装配函数，不修改 `ResearchPipeline`。
