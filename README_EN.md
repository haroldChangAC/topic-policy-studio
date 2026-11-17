# Humanities & Policy Text Studio

[中文](README.md) · [English](README_EN.md)

Humanities & Policy Text Studio is a bilingual, local research workbench for exploring text collections and conducting basic policy evaluation. The current V0.1 is a personal research trial.

## Author and project background

**Jinghao Chang (常景皓)** is a doctoral student in the Program of East Asian Studies at Chonnam National University. He enrolled in 2026.

The author developed this project during doctoral study and research to support research on humanities, area studies and policy texts.

## Features

- Import CSV, XLSX and TXT, inspect documents, and preserve source text and metadata.
- Run BERTopic locally, split long documents into traceable segments, and review and group topics.
- Examine topic charts, annual statistics, independently modeled years and cross-year similarity links.
- Define PMC indicators, record manual coding and evidence, and calculate scores from completed coding.
- Save and reopen separate topic and policy projects, with on-demand data and figure exports.

Topic modeling uses text segments. Basic PMC evaluation uses complete policy documents. Interpret results with reference to their source material.

## Start on the configured Mac

In the project directory, run:

```bash
bash scripts/start.sh
```

Open [http://127.0.0.1:8501](http://127.0.0.1:8501). You can also double-click `启动工作台.command`. The existing Python environment and locally cached model are required for analysis. The user manual covers first-time setup and troubleshooting.

**Save a topic project before closing or refreshing the page.** Select **Project ZIP**, choose **Prepare export**, then **Download prepared file**. In the policy workspace, choose **Download policy project ZIP** directly.

## Documentation

- [English user manual](docs/USER_MANUAL_EN.md) and [PDF](docs/USER_MANUAL_EN.pdf)
- [中文使用手册](docs/USER_MANUAL_ZH.md) and [PDF](docs/USER_MANUAL_ZH.pdf)
- [Chinese README](README.md)

The current application reads topic and policy project versions 1–6. Some historical technical records and fixed labels in standalone Sankey exports retain their original language. English corpus testing has used synthetic texts. Sentiment analysis, PMC-AE and automated Word reports are not implemented.
