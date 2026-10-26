# 论文筛选与排序（第 2 组）

实现位于 `src/paper_agent/adapters/academic_ranker.py`，主流水线接口保持不变。

## 两级排序

- `AcademicIntentRewriter` 保留原始问题，使用中英文规则提取时间、引用、venue、代码侧重。明确的 `exclude "term"` 或 `排除“术语”` 会提取为排除条件；已有结构化约束保留。这是规则实现，并未调用 LLM。
- `AcademicPreRanker` 只读标题、摘要、关键词，结合双语任务词、标题权重、候选集 BM25 和可选 embedding 计算相关性，取全文处理候选池 N。配置 embedding 后使用 `0.50 semantic + 0.30 lexical + 0.20 anchor`，未配置时回退 BM25。核心对象/任务未命中的条目会被过滤，避免通用词命中造成误选。排除撤稿、排除词和日期范围外论文；必需词延后到最终排序检查，以免错过仅在全文出现的术语。
- `AcademicEvidenceVerifier` 尝试读取实际 Markdown 文件；只有路径而没有文件时回退摘要。返回证据可用性/解析质量分数。
- `AcademicFinalRanker` 选择问题相关段落，结合元数据执行 CrossEncoder 重排，检查必需词并取 Top-k。没有模型或模型失败时回退词项覆盖率，解释中明确记录原因。词项回退不能替代语义模型，尤其对跨语言或同义表达有限。

CrossEncoder 查询由 `build_ranking_prompt()` 生成，包含原始问题、必需/排除概念和侧重，版本为 `academic-intent-v1`。这是给重排器的结构化查询，未实现 LLM 提示词重写。对无指令训练的重排器，这种查询格式是否优于原始问题需要在实际数据上评测。

## 默认分数

预排序：`0.90 relevance + 0.06 intent + relevance * (0.02 recency + 0.01 venue + 0.01 citations)`。

`relevance` 由双语词典、短语覆盖、标题加权和候选集 BM25 组成；查询规划的改写概念也参与匹配。标题核心短语优先于泛化单词。分数低于 0.08 或缺少任务锚点的条目不会为了凑够数量进入结果。此方法仍是轻量文本匹配，不等价于语义模型；陌生领域和多约束长问题仍需固定查询集与人工标注评测。

最终排序仍使用归一化的 0～1 内部分数；`display_score = 2 × final` 作为满分 2 的展示分数，不改变排序顺序。分项仍保持 0～1，避免不同指标尺度混乱。

- `recency`：相对运行年份，5 年半衰期；未知年份为 0。
- `venue_quality`：仅表示用户 `preferred_venues` 的匹配程度，不冒充期刊质量评价。
- `citation_signal`：`log(1 + citations) / log(1001)`，最大为 1；未知为 0。
- `intent`：必需概念覆盖率；未指定时为问题词项覆盖率。代码侧重时，20% 来自确认的开源状态。
- 最终 `evidence_coverage`：证据可用性/解析质量乘问题覆盖率；摘要证据可用性为 0.35。
- 时间、venue、引用侧重分别将对应权重增加 0.02，从证据权重扣除，保证总权重为 1，相关性始终占主要部分。

每篇结果包含公式版本、实际后端、提示版本、权重和分项解释；分数是排序启发式，不是相关概率。缺失元数据可能带来偏差。模型输出要求单个 logit，使用 sigmoid 归一化。

用户界面和报告显示 `display_score`，范围为 0～2；内部 `final` 仍为 0～1，换算关系为 `display_score = 2 × final`。两者表示同一排序结果，满分换算不会改变论文顺序。

## 联调与 BGE

在项目根目录使用 PowerShell：

```powershell
$env:PAPER_AGENT_ENV = "ranking"
python -m streamlit run .\src\paper_agent\ui\app.py
```

此模式使用新排序组件；检索、全文处理和总结仍是 Demo 数据。默认不加载模型，适合先联调。

可选安装：

CrossEncoder 的加载与预测参数参照 [Sentence Transformers 官方接口文档](https://www.sbert.net/docs/package_reference/cross_encoder/model.html)，可选依赖限制为 5.x。

可选的 embedding 模型用于预排序，例如 `BAAI/bge-m3` 或其他支持 `SentenceTransformer.encode()` 的多语言模型；`bge-reranker-v2-m3` 属于最终重排模型，不作为 embedding 模型使用。

```powershell
python -m pip install -e ".[dev,ranking]"
$env:PAPER_AGENT_RERANKER_MODEL = "D:\models\your-academic-reranker"
$env:PAPER_AGENT_EMBEDDING_MODEL = "D:\models\your-multilingual-embedding"
```

模型须兼容 `sentence_transformers.CrossEncoder` 的单输出相关性重排接口。项目没有假设 `bge-reranker-academic` 是某个确定存在的公开模型标识；请将选定的 BGE 模型路径或仓库 ID 填入此配置。本地缓存优先，默认禁止模型下载；需要从仓库下载时显式设置：

```powershell
$env:PAPER_AGENT_RERANKER_LOCAL_ONLY = "0"
```

修改配置后重启进程；首次加载会耗时，失败后该实例保持降级状态。模型加载/推理在工作线程执行，同一实例串行推理。无 embedding 时预排序继续使用 BM25，无 reranker 时最终排序继续使用混合文本匹配。

## 与其他组集成及评测

生产装配时在 `bootstrap.py` 中替换 `pre_ranker`、`verifier`、`final_ranker` 即可。第 1 组提供标准化 `Paper` 元数据，第 3 组提供可读的 Markdown 文件和解析质量。约束筛选后可能不足 k，调用方应检查实际结果数。

全文最多读取前 120000 字符，选前 6 个相关段落，证据截取 6000 字符；模型输入上限 512 tokens。因此末尾概念、很长摘要和跨语言词项匹配存在限制。严格必需词检查仅覆盖选中的文本，不能作为全文事实核验。

运行 `python -m pytest` 检查排序行为和端到端装配。正式课程评测应准备固定查询集和人工相关性标签，比较原 Demo、词项回退与实际 BGE 的 Precision@k、nDCG@k、候选池召回率与耗时。当前未给出真实论文数据上的模型效果结论。
