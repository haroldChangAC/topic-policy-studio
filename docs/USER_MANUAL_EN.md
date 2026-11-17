# Humanities & Policy Text Studio user manual

V0.1 bilingual personal research trial · 17 November 2026

**Save before closing, refreshing or rerunning an analysis.** For a topic project, go to **Save and export → Project ZIP → Prepare export → Download prepared file**. The automatic run folder holds the import and the run log. To keep the complete analysis, download the ZIP. In the policy workspace, select **Download policy project ZIP** directly in **Save and export**.

## Author and project background

**Jinghao Chang (常景皓)** is a doctoral student in the Program of East Asian Studies at Chonnam National University. He enrolled in 2026.

The author developed this project during doctoral study and research to support research on humanities, area studies and policy texts.

The About section is available from the sidebar.

## 1. What the Studio does

Use the Studio to explore topics in humanities texts, policies, news or other collections, then verify model output against the sources. Basic PMC policy evaluation lets researchers define indicators, code policies and record their evidence. BERTopic clusters text segments. PMC evaluates whole policies. The two workspaces save and reopen their projects separately.

Use **Workspace** in the sidebar to start a new analysis, open a topic project or enter policy evaluation. Once a project is ready, **On this page** links to its results, sources, charts, annual tools, coding or exports. Expand the sidebar if it is hidden. Long tables scroll within their own area. Long texts appear in a scrollable viewer from which you can select and copy text.

## 2. System requirements and first launch

The tested environment is Apple Silicon macOS with Python 3.11.18 and Streamlit 1.66.1. The project's virtual environment is in `.venv/`, with dependencies pinned in `uv.lock`. The model lock at `models/model-lock.json` pins `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, a multilingual encoder with 384-dimensional embeddings. New analyses require a complete local cache in `models/cache/`. You can reopen a saved project to view its existing results.

On this configured Mac, double-click `启动工作台.command` in Finder or open a terminal in the project folder and run:

```bash
bash scripts/start.sh
```

Visit [http://127.0.0.1:8501](http://127.0.0.1:8501). The service binds to loopback and starts the model in offline mode. For a new installation with permission to install and connect, run `bash scripts/setup.sh`, `bash scripts/download_model.sh` and `bash scripts/check.sh`. On the configured machine, use `bash scripts/check.sh` to diagnose missing components. Plot rendering uses locally available fonts.

After downloading your project, press Control+C in the terminal that started the server to stop it. To resume, start it again and upload the saved ZIP. If the browser reports a refused connection, check the terminal for startup errors and the actual port. If 8501 is occupied, identify the listener with `lsof -nP -iTCP:8501 -sTCP:LISTEN`. Start on an alternate port with `bash scripts/start.sh --server.port 8502` and visit [http://127.0.0.1:8502](http://127.0.0.1:8502).

## 3. Prepare, import and inspect data

Select **New analysis** and upload a CSV, XLSX or TXT up to 10 MiB. Confirm the text column and preview before modeling.

| Format | Prepare | Check |
| --- | --- | --- |
| CSV | Use unique, nonempty headers and one record per document | UTF-8 is recommended. GB18030 and UTF-16 are available. Quote multiline fields correctly |
| XLSX | Select a worksheet with headers in the first row and one document per data row | The Studio reads cached formula values. Recalculate and save the workbook or paste values if a text formula has no cached result |
| TXT | Choose the actual text encoding. Each file is one document | Decoded spaces, punctuation and line breaks are preserved. The UTF-8 BOM is treated as an encoding marker |

Suggested columns include `title`, `text`, `year`, `group` and `source`. Record verifiable four-digit years or explicit dates. Groups can be places, publishers or other defensible categories. Keep provenance in a source field. The app leaves an unknown year unknown. The preview shows up to 100 rows and modeling uses the entire file.

Blank text and excluded duplicate rows remain in the source document table. The default keeps exact duplicate texts. You can choose to model only the first occurrence. Changing the import, text column or analysis settings can clear the current result from the page. Save any result you need first.

## 4. Long documents and source offsets

**Stop for overlong text (default)** reports documents over the model budget. **Split overlong text automatically** prefers paragraph and sentence boundaries, then divides longer runs as needed. Segments do not overlap. The current model limit is 128 tokens per segment including special tokens. The input check shows the effective limit. Splitting preserves original whitespace and punctuation.

A document is an imported record. A segment is a piece of text that the model clusters. In **Topics and source text**, select a topic and a segment to read the segment and the full document. The document and segment IDs identify the exact records. Offsets count Unicode characters from zero and exclude the end. For example, `[1,3)` in `ABCDE` is `BC`. Apply offsets to the saved original text.

## 5. Local BERTopic, clustering and keywords

1. Start with CPU. Apple Silicon can use MPS to compute embeddings. Dimensionality reduction and clustering still use the CPU.
2. **Legacy (default)** preserves the original jieba behavior. **Chinese segmentation** retains single-character words, filters standalone punctuation and accepts custom dictionaries. **English tokenization** retains ASCII English words, numbers and internal apostrophes and hyphens. For mixed text where you want Chinese keywords, use Chinese tokenization.
3. Stopwords are off by default. Enter one word per line or upload a UTF-8 list, then enable stopwords to filter exact lowercased terms. The jieba modes can also use a custom dictionary. English tokenization ignores custom dictionaries. Keyword processing feeds the c-TF-IDF representation. The semantic encoder receives the original segments. Preview the tokenizer before running.
4. HDBSCAN's default minimum cluster size is 4. In advanced settings, UMAP defaults to 8 neighbors and minimum distance 0. HDBSCAN's minimum samples defaults to 2. Neighbors affect the local structure in the projection. Minimum distance affects packing. Cluster size and minimum samples affect density-based grouping and outlier assignment. Use at least 10 valid segments and 5 distinct segments, and follow the parameter checks.
5. Start the analysis. The run records seed 42. Inspect document counts, modeled segments, topic counts, outliers and keywords, then choose segments for source verification. Outliers are labeled `-1`. HDBSCAN membership strength describes cluster assignment. Topic importance and calibrated probability are separate concepts.

Local technical tests used this encoder with fictional English and mixed Chinese–English corpora. Assess topic usefulness on your own sources and language mix before drawing research conclusions.

## 6. Name and group topics

Under **Topic names and groups**, set or clear a display name. The original ID and keywords stay available. Create a group from at least two ungrouped, non-outlier topics. You can edit the name or members, dissolve the group and review the history. Ungrouped topics and outliers remain separate. The app continues to display the members' original c-TF-IDF keywords.

Group segment counts sum the member topic segments. Document mentions count each distinct document ID once, even when the document contains several member topics. Grouping and renaming preserve original topic assignments.

## 7. Topic charts and source review

**Topic charts** provides topic distribution, keyword weights, topic similarity, vector hierarchy and keyword co-occurrence. The distribution counts segments. Weights are derived from model c-TF-IDF. Similarity compares topic embeddings using cosine similarity. Hierarchies use average linkage on `1 − cosine similarity`. A historical project without weights or vectors cannot produce the charts that require them.

The co-occurrence network counts a pair of different words once per segment. Node size represents the number of eligible segments containing a word. Links represent co-occurrence counts. Choose filters, then select a word pair to inspect matching sources. Use publication color or monochrome, adjust size, font and DPI, then inspect the exported figure. Some existing fixed chart text is still in Chinese.

## 8. Annual statistics, independent annual models and cross-year similarity

- **Annual statistics.** Select a year or date field and an optional group field. Choose handling for missing values and outliers, then calculate the yearly topic panel. Segment share divides by included segments in that group and year. Document mention rate divides by distinct documents with at least one included segment. Excluding outliers also excludes documents containing only outlier segments from the document denominator. Documents can mention several topics, so rates may sum to more than 100%. Rates are missing when a denominator is zero. A topic absent from an otherwise eligible sample has a count of zero.
- **Grouped annual charts.** Switch to **Researcher-defined groups** for grouped trends and heatmaps. For a line point, use the same group, year and topic in the heatmap's source review controls. Whole-corpus views count distinct document IDs. Missing years are omitted from plots.
- **Independent models by year.** Run full-corpus analysis first, or open a saved project with modeled segments. Select the date field, years and device. Each year gets its own model result. Years without enough samples show the reason. Topic IDs are local to a year. The original full-corpus analysis and its annual panel remain in the project.
- **Cross-year topic links.** Choose consecutive calendar years, topics, a cosine similarity threshold and a link limit. Links cover comparable adjacent years and exclude outliers. Positive cosine similarity sets link width. Connected weights set node height. Segment and document counts appear separately. Pick a link or comparable pair to review both source and target texts. Similarity offers a comparison for human interpretation. Historical continuation or movement needs independent evidence.

## 9. Basic PMC, coding, evidence and model assistance

1. Open **Policy evaluation**. Import policies and select the text column and optional title, date, agency, region and source. Open an existing policy project to append policies. From an open topic project you can transfer whole documents and keep a snapshot of their source topic associations.
2. Under **Indicator framework**, create first- and second-level indicators with definitions, then confirm. Confirmation records an indicator version. Later edits create a new draft and, when confirmed, another version. Existing coding stays linked to its original version.
3. Under **Coding and evidence**, select a policy and a second-level indicator. Choose 1 (Present), 0 (Absent), Needs review or Not applicable, then record evidence and rationale. Exact evidence has to match the original substring at the specified offsets. A text revision keeps the imported original and past coding. Review stale codes before using the revised document in a complete score.
4. Divide the sum of binary codes under a first-level indicator by its number of second-level indicators. Sum first-level scores for raw PMC. The number of first-level indicators is the maximum. Divide by that maximum and multiply by 10 for the optional ten-point value. Missing, needs-review, not-applicable or stale codes block a complete score. For example, first-level values of 1, 0.5 and 0 give raw PMC 1.5 out of 3, or 5 out of 10.
5. A radar chart needs at least three first-level indicators and can compare up to eight fully coded policies under the same version. The axes span 0–1. A basic PMC surface requires exactly nine first-level indicators with complete coding. You can adjust the 3 × 3 order. Its nine vertices are the observed values; flat triangles connect them.
6. Optional model assistance uses an installed Ollama model through loopback, normally `http://127.0.0.1:11434`. One earlier technical test used `qwen2.5:3b-instruct-q4_K_M`. Confirm your Ollama configuration and the text range before sending private material. For long policies you can create a segmented job and inspect returned coverage. Exact matching locates a quote. Relevance still needs judgment. Accept, edit or reject suggestions in review. Only formal coding goes into PMC.

## 10. Prepare, download and reopen files

| To keep | Action |
| --- | --- |
| Complete topic project | **Save and export → Project ZIP → Prepare export → Download prepared file**. Save the `.hpts.zip` and choose **Open existing project** to reopen |
| Topic, text and parameters | Download small CSV and JSON files from the results area. Choose segment CSV or Excel in the on-demand export controls |
| Annual and cross-year data | Download CSV beside the charts. Use the export area to prepare the annual or cross-year data ZIP |
| Interactive cross-year chart | Extract the cross-year data and chart ZIP. Open `graph.html` with its embedded Plotly script. Node, edge, similarity and settings files are in the same ZIP |
| PNG, SVG and chart CSV | Use chart-level download controls for most figures. Prepare cross-year PNG and SVG in **Save and export**. PNG offers 150, 300 and 600 DPI. SVG Chinese labels are commonly outlines |
| Complete policy project | Select **Download policy project ZIP** directly. CSV and a multi-sheet Excel workbook are in the same section. In **Policy evaluation**, use **Open a policy project** to upload `.pmc.zip` |

The topic workspace keeps one prepared export at a time. Changes to names, groups or chart settings can invalidate the file. Release an unused export or prepare a new one. Project uploads are limited to 50 MiB. Excel cannot represent every unusually long field or control character losslessly. Use CSV and ZIP to preserve those values. Reduce the canvas or resolution if a PNG exceeds 36 million pixels. Static cross-year figures support up to 30 connected nodes and 80 links.

## 11. Language and troubleshooting

- **语言 / Language** in the sidebar switches between Chinese and English. Your document text, filenames, columns, names, keywords and saved records retain their original values. Choose English keyword tokenization separately for English analysis.
- If no keywords remain, examine the tokenizer preview and stopword list. Use Chinese tokenization for mixed data when you want Chinese terms.
- Some historical technical records, saved column values, fixed chart text and original diagnostics remain in their source language. The independent validation lists the remaining interface gaps.
- If the browser session has ended, reopen the last downloaded project ZIP. A CSV or the automatic run log contains only part of the analysis.

## 12. Compatibility, demos and limitations

V0.1 reads topic projects 1–6 and policy projects 1–6. The current topic writer uses version 5 for ordinary results and version 6 when independent annual models are included. Policy format selection depends on content, with version 6 for segmented model tasks. English tokenization settings are saved with the output. Older software cannot recompute that tokenizer. Topic ZIPs preserve results and source references for review. They do not contain a trainable model.

The public repository includes synthetic import examples. Store research corpora and personal project files locally.

Sentiment analysis, PMC-AE and automatic Word reporting remain outside this release. Hover and drag in the standalone Sankey HTML have yet to be tested in a browser with networking isolated. Static and interactive figures share data but compute their layouts separately. Large Excel exports may require considerable memory. Save new copies alongside your original project files.
