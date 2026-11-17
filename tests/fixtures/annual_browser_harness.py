"""Browser QA harness: preload a synthetic, previously analyzed project only."""
from pathlib import Path
import streamlit as st
from studio.project import open_project,save_project
from studio.annual_ui import show
from studio.runtime import ROOT

if 'annual_browser' not in st.session_state:
    st.session_state.annual_browser=open_project((ROOT/'outputs/browser_chart_project.hpts.zip').read_bytes())
st.title('合成语料 · 年度面板浏览器验证')
show(st.session_state.annual_browser,'browser_test')
st.download_button('下载验证项目',save_project(st.session_state.annual_browser),'annual_browser.hpts.zip','application/zip')
