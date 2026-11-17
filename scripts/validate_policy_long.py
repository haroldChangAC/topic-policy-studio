"""Actual local Qwen test on a pre-frozen synthetic long policy, no reviews."""
import csv
import hashlib
import io
import json
import time
import urllib.request
from pathlib import Path
from unittest.mock import patch

from studio.corpus import read_csv
from studio.policy import add_indicator, confirm_indicators, create_project, score
from studio.policy_long import PARAMETERS, candidates, coverage, create_job, process_segment
from studio.policy_project import open_policy_project, save_policy_project

ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/'tests/fixtures/policy_long_real_synthetic_2026-11-07.json'
SHA='c7b16b137a91852882d4d9aa5553bf343d2e28462969a9ac56d6e46faf4a1ffd'
MODEL='qwen2.5:3b-instruct-q4_K_M'
MODEL_DIGEST='357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b'


def local_get(path):
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open('http://127.0.0.1:11434'+path,timeout=10) as response:
        return json.loads(response.read(1024*1024).decode())


def trace_opener(observation):
    factory=urllib.request.build_opener
    class ObservedResponse:
        def __init__(self,response):self.response=response
        def __enter__(self):self.response.__enter__();return self
        def __exit__(self,*args):return self.response.__exit__(*args)
        def read(self,n):
            raw=self.response.read(n)
            try:
                data=json.loads(raw.decode('utf-8'))
                observation['response']={k:data[k] for k in ('model','done','done_reason','total_duration','load_duration','prompt_eval_count','eval_count') if k in data}
            except (ValueError,UnicodeError):observation['response_error']='响应元数据无法解析'
            return raw
    class ObservedOpener:
        def __init__(self,opener):self.opener=opener
        def open(self,request,*args,**kwargs):
            payload=json.loads(request.data);observation['request_options']=payload['options']
            observation['prompt_utf8_bytes']=len(payload['prompt'].encode('utf-8'))
            return ObservedResponse(self.opener.open(request,*args,**kwargs))
    return lambda *handlers:ObservedOpener(factory(*handlers))


def main():
    raw=FIXTURE.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SHA:raise SystemExit('预先标定的样例哈希不符。')
    item=json.loads(raw);models=local_get('/api/tags')['models']
    if not any(m['name']==MODEL and m['digest']==MODEL_DIGEST for m in models):raise SystemExit('找不到已核对摘要的本地模型。')
    buf=io.StringIO(newline='');w=csv.writer(buf);w.writerow(['标题','正文']);w.writerow([item['name'],item['text']])
    p=create_project(read_csv(buf.getvalue().encode()),'正文',{'title':'标题'},'合成长政策.csv')
    parent=add_indicator(p,1,'合成测试一级','仅用于合成测试。')
    indicator=add_indicator(p,2,item['indicator']['name'],item['indicator']['definition'],parent);confirm_indicators(p)
    job=create_job(p,p['policies'][0]['policy_id'],indicator,MODEL,item['overlap_characters'])
    out=ROOT/'outputs/policy_long_validation';out.mkdir(parents=True,exist_ok=True)
    if (out/'real_generated.pmc.zip').exists():raise SystemExit('验证输出已存在，未覆盖。')
    before=score(p);observations=[]
    for segment in job['segments']:
        observed={};started=time.monotonic()
        with patch('studio.policy_assist.urllib.request.build_opener',side_effect=trace_opener(observed)):
            suggestion=process_segment(p,job,segment['order'])
        observed['wall_seconds']=round(time.monotonic()-started,3)
        observed['order']=segment['order'];observed['start']=segment['start'];observed['end']=segment['end']
        observed['result']=suggestion['result'];observed['suggestion_id']=suggestion['suggestion_id']
        observations.append(observed)
        if observed.get('request_options')!=PARAMETERS:raise RuntimeError('实际请求参数与任务记录不一致。')
        if p['encodings'] or score(p)!=before:raise RuntimeError('未经审核的建议改变了正式编码或得分。')
        (out/'real_generated.pmc.zip').write_bytes(save_policy_project(p))
        print('segment',segment['order'],segment['start'],segment['end'],segment['status'],suggestion['match_status'],flush=True)
    found,unresolved=candidates(p,job);runtime=local_get('/api/ps')
    record={'synthetic':True,'fixture_sha256':SHA,'model':MODEL,'model_digest':MODEL_DIGEST,
            'settings':job['settings'],'segments':job['segments'],'coverage':coverage(job),
            'observations':observations,'suggestions':p['ai_suggestions'],'candidates':found,'unresolved':unresolved,
            'pre_model_expected_positions':item['expected_positions'],'runtime':runtime,
            'before_human_review_no_formal_encodings':not p['encodings'],
            'disclaimer':'全部返回只表示收到响应，不能证明模型理解或完整处理了全文。'}
    (out/'real_generated.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    assert open_policy_project((out/'real_generated.pmc.zip').read_bytes())==p


if __name__=='__main__':main()
