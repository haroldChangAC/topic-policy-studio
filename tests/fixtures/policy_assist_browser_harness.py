"""Browser QA harness: preload only the frozen synthetic project into real UI."""
from pathlib import Path
import sys

import streamlit as st

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from studio.policy_project import open_policy_project
from studio.policy_ui import workspace

st.title('合成样例 · 本机模型浏览器验证')
if 'pmc_project' not in st.session_state:
    path=ROOT/'outputs/policy_local_eval/generated.pmc.zip'
    project=open_policy_project(path.read_bytes())
    if not all(p['title'].startswith('S0') for p in project['policies']):
        raise ValueError('仅允许预先固定的合成验证材料。')
    st.session_state['pmc_project']=project
workspace(lambda _device: None)
