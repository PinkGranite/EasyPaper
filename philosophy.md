# 论文生成系统 — 架构与设计哲学

> 本文档梳理 EasyPaper 多智能体论文生成的完整流程、组织逻辑与关键设计方案。

---

## 1. 多智能体系统 (MAS) 组织逻辑

### 1.1 Agent 总览

系统包含 9 个 Agent，按职能分为四层：

| 层级 | Agent | 核心职责 |
|------|-------|---------|
| **编排层** | `MetaDataAgent` | 主编排器，驱动论文生成全生命周期 |
| **规划层** | `PlannerAgent` | 段落级论文规划、引用策略、VLM 图表分析 |
| **执行层** | `WriterAgent` | 生成 LaTeX 正文，支持 ReAct 工具调用与 mini-review |
| **执行层** | `TypesetterAgent` | 资源获取、模板注入、LaTeX 编译、自愈式编译 |
| **执行层** | `CommanderAgent` | 从 FlowGram 画布图组装上下文，构建 `SectionWritePayload` |
| **质控层** | `ReviewerAgent` | 协调多种 Checker（Style/Logic/Structure），输出结构化反馈 |
| **质控层** | `VLMReviewAgent` | 基于 VLM 的 PDF 页面分析（溢出/空白/布局问题） |
| **辅助层** | `TemplateParserAgent` | 解析 LaTeX 模板，提取 document_class/citation_style 等 |

### 1.2 Agent 之间的关系

```
                          ┌──────────────────────┐
                          │    MetaDataAgent      │
                          │    (主编排器/Orchestrator)│
                          └──────────┬───────────┘
                ┌────────────┬───────┼───────┬────────────┐
                ▼            ▼       ▼       ▼            ▼
         ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
         │ Planner  │ │  Writer  │ │ Reviewer │ │VLMReview │
         │  Agent   │ │  Agent   │ │  Agent   │ │  Agent   │
         └──────────┘ └──────────┘ └──────────┘ └──────────┘
                                        │
                          ┌─────────────┼─────────────┐
                          ▼             ▼             ▼
                   ┌────────────┐ ┌────────────┐ ┌────────────┐
                   │StyleChecker│ │LogicChecker│ │StructChecker│
                   └────────────┘ └────────────┘ └────────────┘
```

### 1.3 两种生成模式

| 模式 | 适用场景 |
|------|---------|
| **Simple Mode** | 用户提供 5 个自然语言字段 + BibTeX 引用，独立 API |
| **FlowGram Mode ** | 用户在画布上构建研究图谱，Commander 从图中提取上下文 |

---

## 2. 论文生成流水线 — 核心流程

### 2.1 全流程概览 (Seven-Phase Pipeline)

```
用户输入
   │
   ▼
Phase 0e-pre: Code Context (可选，pre-plan)
   │
   ▼
Phase 0: Planning
   ├── 0a: Research Context v1
   │       ← discover seed references
   ├── 0b: Reference Discovery
   │       ← 新发现的论文注入 ReferencePool
   ├── 0c: Reference Assignment
   │       → 每个 section 分配引用
   └── 0d: Research Context v2
           ← generate research context
   │
   ▼
Phase 0.5: Table Conversion
   │  ← convert_tables(): CSV/Markdown → LaTeX
   ▼
Phase 1: Introduction (Leader Section)
   │  ← WriterAgent.run() with ReAct AskTool
   │  → 提取 contributions 供后续 section 参考
   ▼
Phase 2: Body Sections (顺序生成)
   │  ← 逐个 section: related_work → method → experiment → result → ...
   │  ← 每个 section 经过 Two-Phase Pattern:
   │     Phase A: Search Judgment (LLM 判断是否需要补充引用)
   │     Phase B: Pure Writing (无工具，纯 LLM 生成)
   │  ← WriterAgent.run() with mini-review
   ▼
Phase 3: Synthesis Sections
   │  ← Abstract (基于所有已写 section 综合生成)
   │  ← Conclusion (如果 Planner 规划了 conclusion section)
   ▼
Phase 3.5: Unified Review Orchestration
   │  ├── ReviewerAgent: 文本级反馈 (Style/Logic/Structure Checkers)
   │  ├── VLMReviewAgent: PDF 布局级反馈 (可选)
   │  ├── 反馈合并 → RevisionTask 生成
   │  ├── WriterAgent: 执行修订 (段落级/section 级)
   │  ├── Semantic Consistency Guard: 修订前后语义一致性检查
   │  └── 循环直到 pass 或达到 max_review_iterations
   ▼
Phase 4: PDF Compilation
   │  ← TypesetterAgent: 模板注入 + LaTeX 编译 + 自愈式编译
   ▼
Phase 5: VLM Review
   │  ← VLMReviewAgent: 页面溢出/空白/布局检测
   ▼
输出: PaperGenerationResult
```

---

## 3. 核心组件与功能

### 3.1 ReferencePool — 持久引用池

贯穿整个生成流程的核心数据结构，管理引用的完整生命周期：

```
初始化: 用户 BibTeX/纯文本引用 → ReferencePool.create()
         ├── 纯文本引用自动通过 PaperSearchTool 搜索解析为 BibTeX
         ├── BibTeX 引用通过搜索结果 enrich (补充 abstract/venue/citation_count)
         └── 解析后生成 core_refs + valid_citation_keys

生长:    Planning 阶段 → PlannerAgent.discover_references()
         → add_discovered() 注入新引用
         → valid_citation_keys 自动扩展

使用:    Writing 阶段 → get_all_refs() 提供给 prompt
         → valid_citation_keys 约束 Writer 的引用范围
         → extract_cite_keys() 验证生成内容中的引用合法性

输出:    to_bibtex() → 生成最终 .bib 文件
```

**核心设计原则：**
- Core refs (用户提供) 不可变，discovered refs (搜索发现) 持续增长
- 两层验证：LLM judgment (是否需要搜索) + system cross-reference (引用 key 是否有效)
- `valid_citation_keys` 随流程实时增长，后续 section 和 mini-review 始终看到最新引用集

### 3.2 SessionMemory — 会话记忆

跨 Agent 协调的共享状态对象：

| 字段 | 类型 | 用途 |
|------|------|------|
| `plan` | `PaperPlan` | Planner 创建的论文计划 |
| `generated_sections` | `Dict[str, str]` | 所有已生成 section 的 LaTeX 内容 |
| `contributions` | `List[str]` | 论文贡献点列表 |
| `review_history` | `List[ReviewRecord]` | 完整的审稿历史 |
| `agent_logs` | `List[AgentLogEntry]` | Agent 活动日志（含 narrative） |
| `issue_store` | `Dict[str, Dict]` | Issue Lifecycle 全量状态 |

**关键能力：**
- 写作上下文: 为特定 section 构建写作上下文（计划指导 + 已写 section 摘要 + 贡献点）
- 修订追踪: 构建修订上下文（修订次数 + 历史反馈 + 防回归提示）
- 动态上下文索引: 两阶段搜索（规则过滤 + 可选 LLM 精炼），供 AskTool 使用

---
