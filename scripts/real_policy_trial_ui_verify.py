"""Exercise saved real-data project and compare rendered tables with chart CSV."""
import json
from unittest.mock import patch
import pandas as pd
from streamlit.testing.v1 import AppTest
from studio.runtime import ROOT
from studio.project import open_project,save_project

out=ROOT/'outputs/real_policy_2022_2024'
with patch('socket.socket.connect',side_effect=AssertionError('Offline UI attempted network')):
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=600).run()
    app.radio(key='work_mode').set_value('打开已有项目').run(timeout=600)
    app.file_uploader(key='project_upload').set_value(('real_policy_grouped.hpts.zip',(out/'real_policy_grouped.hpts.zip').read_bytes(),'application/zip')).run(timeout=600)
    if app.exception or app.error:raise AssertionError([str(x.value) for x in app.exception]+[str(x.value) for x in app.error])
    saved=app.session_state['opened_project']['result']
    assert saved['visualization']['annual_view']=='grouped'
    labels=[x.label for x in app.get('download_button')]
    for label in ['下载归组折线图 PNG','下载归组折线图数据 CSV','下载归组热力图 PNG','下载归组热力图数据 CSV','下载修改后的新项目 ZIP']:
        assert label in labels,label
    chart_tables=[x.value for x in app.dataframe if 'scheme_signature' in x.value.columns and 'topic_group_id' in x.value.columns]
    assert len(chart_tables)==2,[(list(x.columns),len(x)) for x in chart_tables]
    for frame,name in zip(chart_tables,['grouped_line.csv','grouped_heatmap.csv']):
        disk=pd.read_csv(out/name,keep_default_na=False)
        cols=['group','year','topic_group_id','member_topic_ids','metric','numerator','denominator','value','missing_status','scheme_signature']
        assert len(frame)==len(disk)
        for i in range(len(frame)):
            for col in cols:
                a=frame.iloc[i][col];b=disk.iloc[i][col]
                if col=='value':assert abs(float(a)-float(b))<1e-12,(col,a,b)
                else:assert str(a)==str(b),(col,a,b)
    back=open_project(save_project(saved))
    assert back['visualization']['grouped_annual_charts']==saved['visualization']['grouped_annual_charts']
    report={'app_test':True,'real_project_uploaded_and_reopened':True,'chart_tables_match_exported_csv':True,
            'chart_png_csv_download_buttons_present':True,'download_click_tested':False,'project_reopened_after_app':True}
    (out/'app_test_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))
