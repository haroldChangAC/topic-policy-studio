"""Check existing local examples and Streamlit save/reopen entry points offline."""
import hashlib,json
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from studio.runtime import ROOT
from studio.project import open_project
from studio.policy_project import open_policy_project,save_policy_project

out=ROOT/'outputs/v0_1_release_validation';out.mkdir(parents=True,exist_ok=True)
theme=ROOT/'outputs/cross_year_trial/relations_project.hpts.zip'
policy=ROOT/'outputs/policy_validation/手算示例.pmc.zip'
traw=theme.read_bytes();praw=policy.read_bytes();sha=lambda x:hashlib.sha256(x).hexdigest()
with patch('socket.socket.connect',side_effect=AssertionError('offline test forbids network')):
    t=open_project(traw);p=open_policy_project(praw)
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=180).run()
    app.radio(key='work_mode').set_value('打开已有项目').run()
    app.file_uploader(key='project_upload').set_value(('example.hpts.zip',traw,'application/zip')).run(timeout=180)
    assert not app.exception and not app.error,([x.value for x in app.exception],[x.value for x in app.error])
    result=app.session_state['opened_project']['result']
    assert len(result['rows'])==len(t['rows']) and len(result['segments'])==len(t['segments'])
    assert 'export_prepare_restored' in [x.key for x in app.button]
    app.selectbox(key='export_kind_restored').select('project').run(timeout=180)
    app.button(key='export_prepare_restored').click().run(timeout=180)
    assert not app.exception and not app.error,([x.value for x in app.exception],[x.value for x in app.error])
    saved=app.session_state['prepared_export_restored']['data'];restored=open_project(saved)
    assert restored['yearly_analysis']==t['yearly_analysis'] and len(restored['segments'])==len(t['segments'])
    assert 'export_download_restored' in [x.key for x in app.get('download_button')]
    app.radio(key='work_mode').set_value('基础政策评价').run(timeout=180)
    app.radio(key='pmc_mode').set_value('打开政策评价项目').run(timeout=180)
    app.file_uploader(key='pmc_project_upload').set_value(('example.pmc.zip',praw,'application/zip')).run(timeout=180)
    assert not app.exception and not app.error,([x.value for x in app.exception],[x.value for x in app.error])
    project=app.session_state['pmc_project']
    for key in ('policies','versions','encodings','active_version','rule'):
        assert project[key]==p[key],key
    assert 'pmc_download_project' in [x.key for x in app.get('download_button')]
    assert open_policy_project(save_policy_project(project))==project
assert sha(theme.read_bytes())==sha(traw) and sha(policy.read_bytes())==sha(praw)
report={'theme':{'path':str(theme),'source_sha256':sha(traw),'version':t['project_info']['version'],'documents':len(t['rows']),'segments':len(t['segments']),'apptest_open_prepare_save_reopen':True},
        'policy':{'path':str(policy),'source_sha256':sha(praw),'schema_version':p['schema_version'],'policies':len(p['policies']),'indicator_versions':len(p['versions']),'apptest_open_save_button_and_backend_roundtrip':True},
        'source_files_unchanged':True,'offline_network_blocked':True,'browser_manual':False}
(out/'examples.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
