"""Measure Streamlit result-page reruns and export preparation in a local process."""
import argparse,json,threading,time
from pathlib import Path
from unittest.mock import patch
import psutil
from streamlit.testing.v1 import AppTest
import studio.runtime as runtime
ROOT=Path(__file__).resolve().parent.parent
runtime.ROOT=ROOT
import studio.workflow as workflow
import studio.project as project
import studio.topic_groups as groups

parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);parser.add_argument('--kind',choices=['real','small'],required=True);parser.add_argument('--app',default=str(ROOT/'app.py'));parser.add_argument('--deferred',action='store_true');args=parser.parse_args()
out=ROOT/'outputs/deferred_export_validation';out.mkdir(parents=True,exist_ok=True)
path=ROOT/'outputs/project_compact_validation/real_policy_grouped_v5.hpts.zip' if args.kind=='real' else ROOT/'outputs/memory_validation/synthetic_small.hpts.zip'
raw=path.read_bytes();proc=psutil.Process();events=[];stages={}
def wrap(name,fn):
 def run(*args,**kwargs):
  start=time.perf_counter();before=proc.memory_info().rss;peak=[before];stop=threading.Event()
  def poll_export():
   while not stop.wait(.005):peak[0]=max(peak[0],proc.memory_info().rss)
  thread=threading.Thread(target=poll_export,daemon=True);thread.start()
  try:value=fn(*args,**kwargs)
  finally:stop.set();thread.join()
  after=proc.memory_info().rss
  item={'kind':name,'seconds':round(time.perf_counter()-start,5),'rss_before':before,'rss_after':after,'rss_sampled_peak':max(peak[0],after)}
  if isinstance(value,bytes):item['bytes']=len(value)
  if name=='csv' and args:
   item['rows']=len(args[0]);item['columns']=list(args[0].columns[:5])
  events.append(item);return value
 return run

def measure(label,fn):
 start=proc.memory_info().rss;peak=[start];stop=threading.Event()
 def poll():
  while not stop.wait(.005):peak[0]=max(peak[0],proc.memory_info().rss)
 thread=threading.Thread(target=poll,daemon=True);thread.start();t=time.perf_counter();offset=len(events)
 try:result=fn()
 finally:
  elapsed=time.perf_counter()-t;stop.set();thread.join();end=proc.memory_info().rss;peak[0]=max(peak[0],end)
 stages[label]={'seconds':round(elapsed,4),'rss_before':start,'rss_after':end,'rss_sampled_peak':peak[0],
                'events':events[offset:]}
 return result

with patch.object(workflow,'csv_bytes',wrap('csv',workflow.csv_bytes)),\
     patch.object(workflow,'excel_bytes',wrap('excel',workflow.excel_bytes)),\
     patch.object(project,'save_project',wrap('zip',project.save_project)),\
     patch.object(groups,'trace',wrap('trace',groups.trace)),\
     patch('socket.socket.connect',side_effect=AssertionError('offline UI attempted network')):
 app=AppTest.from_file(args.app,default_timeout=600).run()
 app.radio(key='work_mode').set_value('打开已有项目').run(timeout=600)
 measure('open',lambda:app.file_uploader(key='project_upload').set_value(('trial.hpts.zip',raw,'application/zip')).run(timeout=600))
 assert not app.exception and not app.error,[str(x.value) for x in app.exception]+[str(x.value) for x in app.error]
 # Changing topic for inspection must not regenerate any prepared large export.
 measure('unrelated_rerun',lambda:app.selectbox(key='restored_topic').select(int(app.session_state['opened_project']['result']['topics']['Topic'].iloc[0])).run(timeout=600))
 assert not app.exception
 if args.deferred:
  for kind in ('segments','excel','project','group_trace'):
   measure('select_'+kind,lambda k=kind:app.selectbox(key='export_kind_restored').select(k).run(timeout=600))
   measure('prepare_'+kind,lambda:app.button(key='export_prepare_restored').click().run(timeout=600))
   assert not app.exception and not app.error,[str(x.value) for x in app.exception]+[str(x.value) for x in app.error]
   cache=app.session_state['prepared_export_restored'];stages['prepare_'+kind]['prepared_bytes']=len(cache['data'])
   measure('reuse_'+kind,lambda:app.run(timeout=600))
   assert not app.exception
 data={'kind':args.kind,'mode':args.label,'project_bytes':len(raw),'documents':len(app.session_state['opened_project']['result']['rows']),
       'segments':len(app.session_state['opened_project']['result']['segments']),'stages':stages,'browser_download':False}
 (out/f'{args.label}_{args.kind}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');print(json.dumps(data,ensure_ascii=False))
