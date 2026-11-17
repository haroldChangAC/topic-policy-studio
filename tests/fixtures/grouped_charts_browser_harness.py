"""Synthetic-only browser harness for actual grouped annual UI."""
import streamlit as st
from studio.grouped_annual_ui import show
from studio.project import open_project,save_project
from studio.runtime import ROOT
if 'grouped_chart_browser' not in st.session_state:
    st.session_state.grouped_chart_browser=open_project((ROOT/'outputs/grouped_charts_validation/synthetic_grouped_charts.hpts.zip').read_bytes())
p=st.session_state.grouped_chart_browser
st.title('合成项目 · 人工归组年度图验证')
show(p,'browser',p['visualization']['annual_panel'])
st.download_button('下载归组图验证项目',save_project(p),'grouped_charts_browser.hpts.zip','application/zip')
