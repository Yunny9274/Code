# LaTeX 参考文献手动插入指南

本项目使用 BibTeX。正文只写引用命令，完整的作者、题名和出版信息统一放在
`references.bib`。不要手工编辑 `samplepaper.bbl`，它是 BibTeX 每次编译生成的文件。

## 1. 在 `references.bib` 中增加条目

会议论文通常使用 `@inproceedings`：

```bibtex
@inproceedings{author2026shortname,
  author    = {Author, Alice and Author, Bob},
  title     = {{GNN}-Based Counterfactual Explanations},
  booktitle = {Proceedings of the ACM Web Conference 2026},
  pages     = {100--110},
  year      = {2026},
  doi       = {10.xxxx/xxxxx}
}
```

期刊论文通常使用 `@article`：

```bibtex
@article{author2026shortname,
  author  = {Author, Alice and Author, Bob},
  title   = {Topology-Aware Explanations for Graph Neural Networks},
  journal = {Neural Networks},
  volume  = {190},
  pages   = {100--115},
  year    = {2026},
  doi     = {10.xxxx/xxxxx}
}
```

花括号后的 `author2026shortname` 是 citation key，由你自己命名，但必须唯一。建议统一用
“第一作者姓 + 年份 + 方法简称”，例如 `lucic2022cfgnn`。

## 2. 在正文中引用

```latex
GNNExplainer learns an instance-specific edge mask
\cite{ying2019gnnexplainer}.

Several counterfactual explainers have been proposed
\cite{lucic2022cfgnn,ma2022clear,huang2023d4explainer}.
```

引用应紧跟它所支撑的事实或论断，不要把一串文献统一堆在段末。`~` 用于防止名称和
编号断行，例如 `Table~\ref{tab:datasets}`；它与文献引用无关。

## 3. 应该在哪些内容后插入引用

- 方法首次出现：说明是谁提出、解决什么问题。
- 数据集、模型或指标首次出现：引用其原始论文或正式数据页面对应论文。
- 可验证的背景事实：例如某类方法通常优化边掩码。
- 与已有工作的比较：每一类关键基线至少有一条原始文献。

不需要引用你自己的方法定义、实验观察和从公式直接推出的结论。避免引用二手博客；优先从
出版社页面、DBLP、Crossref 或论文首页核对 DOI、作者顺序、年份、卷期和页码。

## 4. BibTeX 容易出错的地方

- 多位作者用 `and` 分隔，不使用逗号分隔整个人名列表。
- 方法名和缩写用额外花括号保护大小写，如 `{{GECo}}`、`{{GNN}-Based}`。
- 页码范围用两个连字符，如 `100--110`。
- 会议名称填 `booktitle`，期刊名称填 `journal`。
- 已正式发表的论文优先记录正式会议/期刊版本，不再只写 arXiv。
- LaTeX 特殊字符要转义，例如 `\&`；重音姓名使用 LaTeX 写法。

## 5. 编译顺序

新增引用后运行：

```powershell
pdflatex samplepaper.tex
bibtex samplepaper
pdflatex samplepaper.tex
pdflatex samplepaper.tex
```

首次 LaTeX 编译收集 citation key，BibTeX 生成参考文献，后两次 LaTeX 更新编号和交叉引用。
如果显示 `Citation ... undefined`，先检查正文 key 与 `references.bib` 是否完全一致，再按上述
顺序重新编译。

## 6. 投稿模板说明

当前 `splncs04` 是 Springer LNCS 的数字编号格式。The Web Conference 使用 ACM 模板，
Neural Networks 使用 Elsevier 模板；真正投稿时需要切换文档类和官方模板。通常
`references.bib` 与 `\cite{...}` 可以继续复用，但参考文献样式应由目标模板控制，不要手工
改造成某个期刊的显示格式。
