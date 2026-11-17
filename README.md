# Humanities & Policy Text Studio

[中文](README.md) · [English](README_EN.md)

人文与政策文本研究工作台提供中英文界面，支持在本机进行主题分析与基础政策评价。当前版本为 V0.1 个人研究试用版。

## 作者与项目背景

**常景皓 / Jinghao Chang**，全南大学东亚学博士在读，2026 年入学。英文署名为 Chonnam National University, Program of East Asian Studies。

本项目由作者在博士学习与研究期间开发，服务于人文、区域国别与政策文本研究。

## 功能

- 导入 CSV、XLSX、TXT，检查数据并保存原文和元数据。
- 使用本地 BERTopic 分析文本，切分长文并回查原文位置，支持人工命名和归组主题。
- 查看主题图表、年度统计、分年模型以及跨年主题相似关联。
- 制定 PMC 指标，人工记录编码和证据，按完整编码计算基础分数。
- 分别保存和重新打开主题项目与政策项目，按需导出数据和图表。

主题分析以文本片段为单位，基础 PMC 评价以整份政策为单位。研究者应结合原文解释结果。

## 在已配置的 Mac 上启动

在项目目录执行：

```bash
bash scripts/start.sh
```

打开 [http://127.0.0.1:8501](http://127.0.0.1:8501)，或双击 `启动工作台.command`。运行主题分析需要配置 Python 环境及本地模型。首次安装和故障排查请参阅使用手册。

**关闭或刷新页面前保存主题项目。** 选择“项目 ZIP”，点击“准备导出”，再点击“下载已准备文件”。政策项目在政策评价模块直接点击“下载政策评价项目 ZIP”。

## 使用手册

- [中文使用手册](docs/USER_MANUAL_ZH.md) · [PDF](docs/USER_MANUAL_ZH.pdf)
- [English user manual](docs/USER_MANUAL_EN.md) · [PDF](docs/USER_MANUAL_EN.pdf)

当前工作台可读取主题和政策项目版本 1–6。部分历史技术记录及独立桑基图固定文字保留原语言。英文语料流程已使用合成文本测试。情感分析、PMC-AE 和自动 Word 报告尚未实现。
