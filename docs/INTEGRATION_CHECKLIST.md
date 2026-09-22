# 1–3 组联调清单

## 组 1：检索与去重

- 实现 `QueryPlanner`、至少两个 `PaperRetriever`、`BibTexMapper`、`PaperMerger`；
- 每个来源只返回元数据，不下载 PDF；
- 覆盖超时、429、空结果、非法日期、缺摘要和 BibTeX 多作者测试；
- 记录来源 ID、检索时间和原始元数据；
- DOI、arXiv 版本号、标题标点差异和扩展版去重均有固定样例。

## 组 2：筛选与重排

- 实现 `RankingIntentRewriter`（可选）、`PreRanker`、`EvidenceVerifier`、`FinalRanker`；
- 预排序只选 N 个全文候选，不提前截断最终 Top-k；
- 所有分数归一化，记录公式/模型/提示词版本和解释；
- 时间、venue、引用量不得压过主题相关性；
- 在固定标注查询集上报告 Precision@k、nDCG@k 或人工相关性均值。

## 组 3：下载、解析与总结

- 实现 `OpenAccessResolver`、`PdfDownloader`、`MarkdownParser`，组合成 `FullTextProcessor`；
- 实现 `PaperSummarizer` 与 `ReportWriter`；
- 下载器限制协议、重定向、响应大小、MIME、PDF 魔数、超时与并发；
- 使用内容哈希缓存，重复运行不重复下载/解析；
- 摘要降级必须显式标记，不从摘要虚构实验细节；
- 关键总结结论提供 `EvidenceRef`。

## 合并规则

每组只向 `adapters/` 添加实现和测试。最终接线只修改 `bootstrap.py` 的生产分支；
不得修改 `ResearchPipeline` 的阶段顺序，也不得在 UI 或 MCP 服务器复制业务逻辑。

