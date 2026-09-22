# 数据模型与空值规范

所有组必须复用 `paper_agent.domain.models`，禁止复制一个“自己的 Paper”。这可以避免
BibTeX、排序、下载和总结阶段之间出现字段名与空值语义不一致。

## Paper 关键字段

| 字段 | 类型 | 可为空 | 设计说明 |
|---|---|---:|---|
| `paper_id` | UUID | 否 | 系统内部稳定标识，不等同于任何外部 ID |
| `title` | str | 否 | 唯一强制元数据，不能为空字符串 |
| `authors` | `list[Author]` | 列表可空 | API 未返回作者时为空列表，不伪造“Unknown”作者 |
| `institutions` | `list[str]` | 列表可空 | 论文级去重单位；作者级单位保留在 `Author.affiliations` |
| `publication_date` | date | 是 | 只知道年份时保持 `None`，使用 `year` |
| `year` | int | 是 | 若日期存在则必须与日期年份一致 |
| `publication_status` | enum | 否 | 默认 `unknown`；preprint 不等于已正式发表 |
| `venue` | Venue | 是 | arXiv 分类不能冒充会议/期刊 |
| `abstract` / `comments` | str | 是 | 未返回时用 `None`，不用空字符串 |
| `arxiv_id` / `doi` | str | 是 | DOI 应在组 1 中去 URL 前缀并转小写；arXiv 去版本后缀用于去重 |
| `bibtex` | str | 是 | 保存原始条目，映射后字段仍需独立验证 |
| `is_open_access` | bool | 是 | 第三方元数据声明；未知必须是 `None` |
| `open_access_url` | HttpUrl | 是 | 元数据提供的 OA 地址，尚不代表下载成功 |
| `fulltext.access_status` | enum | 否 | 本次处理实际观察，默认 `unknown` |
| `fulltext.pdf_url` | HttpUrl | 是 | 可能存在但下载/解析失败 |
| `code.status` | enum | 否 | 代码开源与论文 OA 是两个独立维度 |
| `code.repository_url` | HttpUrl | 是 | 只有确认仓库后填写 |
| `source_records` | `list[SourceRecord]` | 列表可空 | 真实检索结果必须至少一条，手工构造/测试可为空 |
| `scores` | ScoreBreakdown | 否 | 分项可空，使用过的分数必须落在 `[0,1]` |

## 合并优先级建议

1. DOI 完全匹配；
2. arXiv ID 去版本号后匹配；
3. 外部来源稳定 ID；
4. 规范化标题 + 第一作者 + 年份的谨慎匹配；
5. 模糊标题只能生成“疑似重复”候选，不能自动合并期刊扩展版。

字段冲突不能静默覆盖：选择规范值后，将说明写入 `metadata_conflicts`，并保留每个
`SourceRecord.raw_metadata` 供审计。

