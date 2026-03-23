### Phase 1: 数据集构建 (Dataset Construction) —— "Reverse-Engineering"

为了让实验公平且可复现，不能随便想的 idea 测试，必须使用真实的高质量论文来构建基准测试集。

* **动作**：从 ArXiv 或近两年的顶会（如 NeurIPS 2024/2025, ICLR 2025）中，随机抽取 100 篇高质量的计算机科学/机器学习论文源文件（LaTeX 源码）。
* **处理（Reverse-Engineering）**：
1. 提取原始论文的 Metadata：标题、Idea/Hypothesis、Method 核心公式/伪代码、实验数据（核心表格和图表说明）、真实引用的 BibTeX。
2. 提取目标约束：将原论文的页数（如 8 页）和排版格式（如双栏 NeurIPS 模板）作为约束条件。


* **目标**：用这 100 个真实的 Metadata 作为系统的输入，让各种模型/系统重新生成这 100 篇论文，从而与“人类 Ground Truth”以及其他 Baseline 进行客观对比。

---

### Phase 2: 确立对比基线 (Baselines)

不只和弱模型比，要和当前最主流的方案比，证明你的特定 MAS 编排是不可替代的。

* **Baseline 1: Zero-shot LLM (GPT-4o 或 Claude 3.5 Sonnet)**
* *设定*：将 Metadata 和所有的 BibTeX、图表要求一次性放入超级长 Prompt，要求模型直接输出完整的 LaTeX 源码。
* *目的*：证明在没有 MAS 拆解和多轮 Review 的情况下，单一强大模型依然无法搞定物理约束和长程逻辑。


* **Baseline 2: RAG-based Pipeline (e.g., LlamaIndex/LangChain 原生方案)**
* *设定*：使用标准的检索增强生成。先根据 Metadata 检索提供的 BibTeX，然后按章节顺序单向生成（无 Review 闭环，无 VLM）。
* *目的*：证明单向流水线极易产生引用脱节和排版溢出。


* **Baseline 3: SOTA General MAS (e.g., MetaGPT 或 ChatDev)**
* *设定*：使用业界公认的软件工程通用多智能体框架，自定义角色让其写论文。
* *目的*：证明“通用 MAS”解决不了学术写作的特化问题（如 LaTeX 编译闭环和 Citation Budget）。



---

### Phase 3: 核心量化指标计算 (Quantitative Metrics)

这是实验章节的重头戏，对应三个核心学术贡献。需要写自动化脚本来跑这三组数据。

**1. 物理排版与约束遵循 (Physical Layout & Constraint Satisfaction)**

* **Compilation Success Rate (编译成功率)**：一次性通过 `pdflatex` 或 `xelatex` 编译生成 PDF 且没有 Fatal Error 的比例。（证明代码鲁棒性）
* **Constraint Violation Error (CVE / 页数误差)**：生成的 PDF 页数与目标页数（如 8 页）的绝对误差均值。如果超页或少页，都要扣分。（证明 VLM 闭环的有效性）
* **Visual Overlap / Underfill (视觉异常率)**：通过脚本检查 PDF，是否存在文字覆盖图片、表格超出边界等严重排版事故的比例。

**2. 事实性与引用准确度 (Factuality & Citation Grounding)**

* **Citation Precision / Recall (引用精确率与召回率)**：
* *Precision*：生成的文本中使用的 `\cite{}`，有多少是真实存在于给定 BibTeX 列表中的（非捏造）。
* *Recall*：Planner 强制分配的 Citation Budget，有多少真正被 Writer 合理用在了最终文本中。


* **Claim-Evidence Entailment (主张-证据蕴含得分)**：
* *自动化打分*：抽取生成的论文中带有引用的句子（Claims），与其引用的原论文 Abstract 进行对比，使用 NLI（自然语言推理）模型或 GPT-4 裁判，打分（1-5 分）判断该主张是否真的被证据支持。



**3. 结构连贯性与写作质量 (Structural Coherence)**

* **G-Eval (LLM-as-a-Judge)**：采用业界标准的 LLM 裁判法，围绕三个维度打分：逻辑连贯性（Coherence）、学术语气（Academic Tone）、冗余度（Redundancy）。
* **Structure Preservation (结构保留率)**：与人类撰写的 Ground Truth 对比，提取双方的章节目录树，计算结构相似度。

---

### Phase 4: 烧脑消融实验 (Ablation Studies)

在证明了完整系统碾压 Baseline 之后，必须拆解系统，证明你写的每一个 Agent 模块都是“必须的”。

* **w/o VLM Review**：关闭 VLM 视觉排版检查，仅依赖文本 Reviewer。预期结果：Compilation Rate 可能还在，但 **Constraint Violation Error (超页率)** 和 **Visual Overlap** 将大幅飙升。
* **w/o Citation Budget**：关闭 Planner 里的强证据分配，让 Writer 自由调用文献。预期结果：**Citation Precision** 下降，幻觉率上升，证明预算分配理论是防幻觉的关键。
* **w/o Hierarchical Conflict Resolution**：关闭你系统里复杂的优先级和层级消解（把所有 Review 意见平铺给 Writer）。预期结果：修改容易发散（Divergence），系统可能在多次迭代后陷入死循环，或者写作质量反而下降。

---

### Phase 5: 定性案例分析 (Qualitative Case Studies)

量化数据跑完后，需要在论文中放 1-2 个直观的图表或截图，给 Reviewer 带来视觉冲击。

* **Case 1: VLM 挽救排版灾难**：展示一个中间态截图中，表格严重超出了右边界或超出了第 8 页。然后展示你的系统如何生成了 `StructuralAction (RESIZE/DOWNGRADE)`，最终态完美对齐双栏边界。
* **Case 2: 复杂的冲突消解过程可视化**：画一个流程图，展示一个段落既被要求扩写（逻辑补充），又被要求压缩（排版限制），系统最终是如何聪明地做出“将次要图表移入附录，保留文本扩写”的决策的。