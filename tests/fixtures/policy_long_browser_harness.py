"""Browser-only QA harness, preloading the frozen synthetic long policy."""
import json
import sys
from pathlib import Path

import streamlit as st

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from studio.policy_project import open_policy_project
from studio.policy_ui import workspace

st.title('合成长政策 · 本机模型浏览器验证')
if 'pmc_project' not in st.session_state:
    project=open_policy_project((ROOT/'outputs/policy_long_validation/browser_input.pmc.zip').read_bytes())
    fixture=json.loads((ROOT/'tests/fixtures/policy_long_real_synthetic_2026-11-07.json').read_text())
    if len(project['policies'])!=1 or project['policies'][0]['original_text']!=fixture['text']:
        raise ValueError('验证页面只允许固定的合成长政策。')
    st.session_state['pmc_project']=project
workspace(lambda _device:None)
