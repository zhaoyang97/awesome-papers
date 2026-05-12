# shared/ — 跨笔记共享素材 / Cross-note shared assets

每篇 deep note 的「思想史脉络（Idea Lineage）」section 里都有一张 Mermaid 引用图，中英两版**字符级相同**。这里保存的是这些 Mermaid 图的**源文件**（`<slug>/lineage.mmd`），方便：

- Fork 本仓的人单独提取 / 修改某张引用图
- 未来用静态生成器把 Mermaid 渲染成 SVG / PNG
- 验证已发布笔记里的 Mermaid 块与源文件是否一致

> 这些 `.mmd` 文件**不**被 mkdocs 当作页面渲染（mkdocs 只扫描 `docs_zh/` 和 `docs_en/`）。它们是源真值，对应的 Mermaid block 已经内嵌到每篇笔记的 markdown 里。
>
> 维护规则：当 [.github/skills/write-deep-note](../../.github/skills/write-deep-note) 的 Round 4 跑完时，会**同时**写入这里的 `.mmd` 和两版笔记里的 ` ```mermaid ` 块；二者必须保持一致。

---

For each deep note, the *Idea Lineage* section contains one Mermaid graph, kept **character-identical** between the Chinese and English versions. This folder stores the **source-of-truth Mermaid files** (`<slug>/lineage.mmd`) so downstream readers can:

- extract or modify a single lineage graph in isolation,
- pre-render the Mermaid into SVG / PNG with a static-site generator,
- verify the published notes' inlined Mermaid against the canonical source.

> These `.mmd` files are **not** rendered as pages by mkdocs (mkdocs only scans `docs_zh/` and `docs_en/`). They are the source-of-truth, and the corresponding Mermaid block is already inlined inside every published note.
>
> Maintenance contract: when Round 4 of the [write-deep-note](../../.github/skills/write-deep-note) skill finishes, it writes both the `.mmd` here and the ` ```mermaid ` block in the two language versions; the two must stay in sync.
