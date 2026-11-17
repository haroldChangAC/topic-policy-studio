"""Run the real-corpus import and offline fit through the actual Streamlit app."""
import json
import time
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from studio.runtime import ROOT
from studio.project import save_project

out=ROOT/'outputs/real_policy_2022_2024'
start=time.perf_counter()
with patch('socket.socket.connect',side_effect=AssertionError('离线测试禁止网络连接')):
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=900).run()
    assert not app.exception, list(app.exception)
    app.file_uploader(key='csv_upload').set_value(('real_reports.csv',(out/'real_reports.csv').read_bytes(),'text/csv')).run(timeout=900)
    assert not app.exception and app.selectbox(key='text_column').value=='正文'
    assert app.radio(key='duplicate_policy').value=='保留全部（默认）'
    print('导入预览',[(x.label,x.value) for x in app.metric],flush=True)
    app.radio(key='split_mode').set_value('自动切分超长文本').run(timeout=900)
    print('切分预览',[(x.label,x.value) for x in app.metric],flush=True)
    assert not app.button(key='run_analysis').disabled
    app.button(key='run_analysis').click().run(timeout=900)
    if app.exception or app.error:
        raise RuntimeError('页面错误：'+repr([str(x.value) for x in app.exception]+[str(x.value) for x in app.error]))
    result=app.session_state['analysis_result']['result']
    (out/'before_grouping.hpts.zip').write_bytes(save_project(result))
    result['topics'].to_csv(out/'topics_original.csv',index=False,encoding='utf-8-sig')
    result['segments'].to_csv(out/'segments_original.csv',index=False,encoding='utf-8-sig')
    result['rows'].to_csv(out/'documents_original.csv',index=False,encoding='utf-8-sig')
    (out/'run_metadata.json').write_text(json.dumps(result['metadata'],ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('模型与 UI 分析完成',json.dumps({'seconds':round(time.perf_counter()-start,2),'counts':result['metadata']['counts'],'non_outlier_topics':result['metadata']['topic_count_excluding_outliers'],'outliers':result['metadata']['outlier_count']},ensure_ascii=False),flush=True)
