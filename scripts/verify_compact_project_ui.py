"""Open the real compact project in the Streamlit testing interface offline."""
import json
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from studio.project import open_project,save_project
from studio.runtime import ROOT

out=ROOT/'outputs/project_compact_validation'
with patch('socket.socket.connect',side_effect=AssertionError('network prohibited')):
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=300).run()
    app.radio(key='work_mode').set_value('打开已有项目').run()
    app.file_uploader(key='project_upload').set_value(('trial-v5.hpts.zip',(out/'real_policy_grouped_v5.hpts.zip').read_bytes(),'application/zip')).run(timeout=300)
    if app.exception or app.error:raise AssertionError([str(x.value) for x in app.exception]+[str(x.value) for x in app.error])
    result=app.session_state['opened_project']['result']
    assert result['project_info']['version']==5
    assert result['visualization']['annual_view']=='grouped'
    labels={x.label for x in app.get('download_button')}
    assert {'下载修改后的新项目 ZIP','下载片段结果 CSV','下载结果 Excel','下载归组热力图数据 CSV'}.issubset(labels)
    charts=[x.value for x in app.dataframe if 'scheme_signature' in x.value.columns and 'topic_group_id' in x.value.columns]
    assert len(charts)==2
    again=open_project(save_project(result))
    assert again['visualization']['grouped_annual_charts']==result['visualization']['grouped_annual_charts']
    report={'real_project_opened_in_apptest':True,'version':5,'grouped_chart_tables_shown':len(charts),
            'export_buttons_present':True,'opened_and_saved_again':True,'browser_clicked':False}
    (out/'ui_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)
