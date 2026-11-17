"""Measure a full Streamlit AppTest render; no browser or network used."""
import argparse,json,threading,time
from pathlib import Path
from unittest.mock import patch
import psutil
from streamlit.testing.v1 import AppTest
import studio.runtime as runtime
ROOT=Path(__file__).resolve().parent.parent
runtime.ROOT=ROOT

parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);parser.add_argument('--kind',choices=['real','large'],required=True);args=parser.parse_args()
out=ROOT/'outputs/memory_validation'
path=ROOT/'outputs/project_compact_validation/real_policy_grouped_v5.hpts.zip' if args.kind=='real' else out/'synthetic_large.hpts.zip'
raw=path.read_bytes();proc=psutil.Process();start=proc.memory_info().rss;peak=[start];stop=threading.Event()
def poll():
 while not stop.wait(.005):peak[0]=max(peak[0],proc.memory_info().rss)
th=threading.Thread(target=poll,daemon=True);th.start();t=time.perf_counter()
with patch('socket.socket.connect',side_effect=AssertionError('offline UI attempted network')):
 app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=600).run()
 app.radio(key='work_mode').set_value('打开已有项目').run(timeout=600)
 app.file_uploader(key='project_upload').set_value(('trial.hpts.zip',raw,'application/zip')).run(timeout=600)
 elapsed=time.perf_counter()-t;end=proc.memory_info().rss;peak[0]=max(peak[0],end)
 if app.exception or app.error:raise RuntimeError([str(x.value) for x in app.exception]+[str(x.value) for x in app.error])
 result=app.session_state['opened_project']['result'];f=result['metadata']['generated_fields']
 assert result['segments'].iloc[0][f['segment_text']] in [v.value for v in app.text]
 data={'kind':args.kind,'elapsed_seconds':round(elapsed,4),'rss_before_bytes':start,'rss_after_bytes':end,
       'rss_sampled_peak_bytes':peak[0],'sample_interval_ms':5,'segments':len(result['segments']),
       'open_and_default_trace_visible':True,'browser':False,'download_buttons':[v.label for v in app.get('download_button')]}
stop.set();th.join();(out/f'{args.label}_ui_{args.kind}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');print(json.dumps(data,ensure_ascii=False))
