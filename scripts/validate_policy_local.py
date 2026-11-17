"""Run frozen synthetic policy cases against a user-installed local Ollama model.

No real policies are read.  Per-call metadata is observed without changing the
workbench request.  Generation does not perform any human review or scoring.
"""
import argparse
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
from studio.policy_assist import generate, request_ollama
from studio.policy_project import open_policy_project, save_policy_project

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/policy_local_eval_2026-11-07.json'
FIXTURE_SHA = '07d31c521f897d77e808f861ed5e30121dd2dc83674700e5e7c6d3f0d7c32f8a'


def local_get(url):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=10) as response:
        return json.loads(response.read(1024*1024).decode('utf-8'))


def observed_requester(observation):
    original_factory = urllib.request.build_opener
    class ObservedResponse:
        def __init__(self, response): self.response = response
        def __enter__(self): self.response.__enter__(); return self
        def __exit__(self, *args): return self.response.__exit__(*args)
        def read(self, n):
            raw = self.response.read(n)
            try:
                data = json.loads(raw.decode('utf-8'))
                for key in ('model','done','done_reason','total_duration','load_duration',
                            'prompt_eval_count','prompt_eval_duration','eval_count','eval_duration'):
                    if key in data: observation[key] = data[key]
            except (ValueError, UnicodeError):
                observation['telemetry_error'] = '无法解析响应元数据'
            return raw
    class ObservedOpener:
        def __init__(self, opener): self.opener = opener
        def open(self, *args, **kwargs): return ObservedResponse(self.opener.open(*args, **kwargs))
    def factory(*handlers): return ObservedOpener(original_factory(*handlers))
    def request(url, model, prompt):
        with patch('studio.policy_assist.urllib.request.build_opener', side_effect=factory):
            return request_ollama(url, model, prompt)
    return request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    raw = FIXTURE.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != FIXTURE_SHA: raise SystemExit('测试样例哈希不符，拒绝运行。')
    fixture = json.loads(raw); cases = fixture['cases']
    csv_buffer = io.StringIO(newline='')
    writer = csv.writer(csv_buffer); writer.writerow(['case_id','正文'])
    writer.writerows((case['case_id'],case['text']) for case in cases)
    project = create_project(read_csv(csv_buffer.getvalue().encode('utf-8')),'正文',{'title':'case_id'},'合成政策样例.csv')
    parent = add_indicator(project,1,'合成样例：生活补贴','用于验证的示例，不代表通用评价体系。')
    indicator = add_indicator(project,2,cases[0]['indicator_name'],cases[0]['indicator_definition'],parent)
    confirm_indicators(project)
    directory = ROOT / 'outputs/policy_local_eval'; directory.mkdir(parents=True,exist_ok=True)
    model_info = local_get('http://127.0.0.1:11434/api/tags')
    matching = [m for m in model_info.get('models',[]) if m.get('name')==args.model]
    if len(matching)!=1: raise SystemExit('本机模型列表未找到唯一指定模型，不进行推理。')
    out = {'fixture_sha256':digest,'synthetic_only':True,'model':matching[0],
           'service_version':local_get('http://127.0.0.1:11434/api/version'),
           'cases':[],'scope_notice':'已发送整段不证明模型完整处理，须人工判断。'}
    for case, policy in zip(cases,project['policies']):
        before = score(project); old = len(project['encodings']); info={}; started=time.monotonic()
        item=generate(project,policy['policy_id'],indicator,args.model,0,len(case['text']),
                      requester=observed_requester(info))
        seconds=round(time.monotonic()-started,3)
        if len(project['encodings'])!=old or score(project)!=before:
            raise RuntimeError('未经审核的建议改变了正式编码或得分。')
        out['cases'].append({'case_id':case['case_id'],'pre_model_human_assessment':case['pre_model_human_assessment'],
                             'pre_model_exact_evidence':case['pre_model_exact_evidence'],
                             'pre_model_evidence_positions':case['pre_model_evidence_positions'],
                             'suggestion':item,'server_telemetry':info,'wall_seconds':seconds})
        (directory/'generated.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        (directory/'generated.pmc.zip').write_bytes(save_policy_project(project))
        print(case['case_id'],item['result'],item['model_status'],item['suggested_status'],item['match_status'],seconds,flush=True)
    out['running_models'] = local_get('http://127.0.0.1:11434/api/ps')
    (directory/'generated.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    assert open_policy_project((directory/'generated.pmc.zip').read_bytes()) == project


if __name__=='__main__': main()
