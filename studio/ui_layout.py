"""Native Streamlit navigation and bounded source-text viewers."""

from html import escape

import streamlit as st

from studio.i18n import tr
from studio.runtime import ROOT


def anchor(name):
    st.html(f'<div id="studio-{name}" style="scroll-margin-top:4rem"></div>')


def source_text(text):
    long = len(text) > 350 or text.count('\n') >= 6
    if long:
        st.caption(tr('长原文可在下框滚动查看。'))
    # Streamlit text and code widgets strip boundary whitespace. A span inside
    # the pre element preserves it, including an initial line break.
    source = escape(text).replace('\r', '&#13;')
    height = '15rem' if long else 'none'
    st.html(f'<pre data-studio-source="true" tabindex="0" '
            f'style="max-height:{height};overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;'
            'font:inherit;padding:.7rem .85rem;margin:0;border:1px solid rgba(125,135,130,.32);'
            f'border-radius:.45rem"><span>{source}</span></pre>')


def section_links(mode):
    if mode == '基础政策评价':
        loaded = st.session_state.get('pmc_project') is not None
        links = [('top', '项目与导入')]
        if loaded:
            links += [('policy-source', '政策原文'), ('policy-indicators', '指标体系'),
                      ('policy-coding', '编码与证据'), ('policy-results', 'PMC 计分与图表'),
                      ('policy-export', '保存与导出')]
    else:
        loaded = st.session_state.get('analysis_result' if mode == '新建分析' else 'opened_project') is not None
        links = [('top', '项目与导入')]
        if loaded:
            links += [('topics', '主题与原文'), ('groups', '主题命名与归组'), ('charts', '主题图表'),
                      ('annual', '年度统计'), ('yearly', '按年份分别建模'),
                      ('cross-year', '跨年主题关联'), ('exports', '保存与导出')]
    st.sidebar.markdown('**' + tr('本页导航') + '**')
    for target, label in links:
        st.sidebar.html(f'<a href="#studio-{target}" style="display:block;padding:.18rem 0">{tr(label)}</a>')


def help_menu():
    with st.sidebar.popover(tr('关于本项目'), use_container_width=True):
        st.markdown(tr('**作者**：常景皓 / Jinghao Chang'))
        st.write(tr('全南大学东亚学博士在读，2026 年入学。'))
        st.write(tr('本项目由作者在博士学习与研究期间开发，服务于人文、区域国别与政策文本研究。'))
    with st.sidebar.popover(tr('帮助与使用手册'), use_container_width=True):
        st.caption(tr('先保存研究结果，再关闭页面。主题项目须准备并下载 ZIP；政策项目可直接下载 ZIP。'))
        for filename, label in [('USER_MANUAL_ZH.md', '中文使用手册'), ('USER_MANUAL_EN.md', 'English user manual')]:
            path = ROOT / 'docs' / filename
            if path.is_file():
                st.download_button(label, path.read_bytes(), filename, 'text/markdown', key='manual_'+filename,
                                   on_click='ignore')
