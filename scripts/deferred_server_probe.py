"""Sample the real local Streamlit server RSS for browser verification."""
import argparse,os,signal,subprocess,sys,time
from pathlib import Path
import psutil
from studio.runtime import ROOT
p=argparse.ArgumentParser();p.add_argument('--app',default=str(ROOT/'app.py'));p.add_argument('--label',required=True);p.add_argument('--port',type=int,default=8765);args=p.parse_args()
out=ROOT/'outputs/deferred_export_validation';out.mkdir(parents=True,exist_ok=True)
log=(out/f'{args.label}_server.log').open('w',encoding='utf-8')
child=subprocess.Popen([sys.executable,'-m','streamlit','run',args.app,'--server.headless=true',
                        '--server.address=127.0.0.1',f'--server.port={args.port}','--browser.gatherUsageStats=false'],stdout=log,stderr=subprocess.STDOUT)
(out/f'{args.label}_server.pid').write_text(str(child.pid))
stop=False
def stop_handler(*_):
 global stop
 stop=True
signal.signal(signal.SIGINT,stop_handler);signal.signal(signal.SIGTERM,stop_handler)
try:
 process=psutil.Process(child.pid)
 with (out/f'{args.label}_server_rss.csv').open('w',encoding='utf-8') as f:
  f.write('epoch_seconds,rss_bytes\n')
  while not stop and child.poll() is None:
   try:f.write(f'{time.time():.6f},{process.memory_info().rss}\n')
   except psutil.Error:break
   f.flush();time.sleep(.01)
finally:
 if child.poll() is None:child.terminate()
 try:child.wait(timeout=10)
 except subprocess.TimeoutExpired:child.kill();child.wait()
 log.close()
