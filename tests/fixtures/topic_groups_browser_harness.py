"""Synthetic-only UI harness for manual grouping browser checks."""
import hashlib
import streamlit as st
from studio.project import open_project,save_project
from studio.runtime import ROOT
from studio.topic_groups_ui import show
from studio.grouped_annual_ui import show as annual_show

if 'groups_browser_project' not in st.session_state:
    st.session_state.groups_browser_project=open_project((ROOT/'outputs/annual_validation/synthetic_annual.hpts.zip').read_bytes())
st.title('合成项目 · 主题归组浏览器验证')
st.caption('仅使用合成输入；这是实际工作台归组和年度面板组件。')
project=st.session_state.groups_browser_project
show(project,'browser')
annual_show(project,'browser',project['visualization']['annual_panel'])
st.download_button('下载归组验证项目',save_project(project),'topic_groups_browser.hpts.zip','application/zip')
uploaded=st.file_uploader('重新打开验证项目',type='zip',key='groups_reopen')
if uploaded:
    digest=hashlib.sha256(uploaded.getvalue()).hexdigest()
    if st.session_state.get('groups_reopen_digest')!=digest:
        st.session_state.groups_browser_project=open_project(uploaded.getvalue());st.session_state.groups_reopen_digest=digest;st.rerun()
    st.success('已重新打开验证项目')
