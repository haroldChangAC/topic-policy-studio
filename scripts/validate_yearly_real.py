"""Offline per-year technical trial on six actual municipal reports."""
import copy,hashlib,json,time
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from studio.runtime import ROOT,configure
from studio.project import open_project,save_project
from studio.modeling import load_encoder
from studio.yearly import run,tables
from studio.deferred_exports import create_export
from studio.annual import calculate
configure()
out=ROOT/'outputs/yearly_trial';out.mkdir(parents=True,exist_ok=True)
source=ROOT/'outputs/project_compact_validation/real_policy_grouped_v5.hpts.zip'
raw=source.read_bytes();result=open_project(raw);before=copy.deepcopy(result)
start=time.perf_counter()
with patch('socket.socket.connect',side_effect=AssertionError('offline network forbidden')):
 encoder=load_encoder('cpu');result['yearly_analysis']=run(result,'年份',[2022,2023,2024],encoder,progress=lambda s:print(s,flush=True))
print('model elapsed',round(time.perf_counter()-start,2),flush=True)
summary,parts,vectors=tables(result);f=result['metadata']['generated_fields'];lookup=result['rows'].set_index(f['document_id'])
for _,part in parts.iterrows():
 doc=lookup.loc[part[f['document_id']]];assert doc[f['original_text']][int(part[f['start_char']]):int(part[f['end_char']])]==part[f['segment_text']]
for k in ('rows','segments','topics'):pd.testing.assert_frame_equal(before[k],result[k])
settings=before['visualization']['annual_panel'];pd.testing.assert_frame_equal(calculate(before,settings)[1],calculate(result,settings)[1])
project=save_project(result);(out/'six_reports_yearly.hpts.zip').write_bytes(project);opened=open_project(project)
assert opened['yearly_analysis']==result['yearly_analysis']
for k in ('rows','segments','topics'):
 pd.testing.assert_frame_equal(before[k],opened[k],check_dtype=False)
pd.testing.assert_frame_equal(calculate(opened,settings)[1],calculate(before,settings)[1])
export=create_export(opened,'yearly');(out/'yearly_data.zip').write_bytes(export['data'])
summary.to_csv(out/'yearly_topics.csv',index=False,encoding='utf-8-sig')
examples=[]
for runitem in opened['yearly_analysis']['results']:
 if runitem['status']=='completed':
  first=next((t for t in runitem['topics'] if t['topic_id']!=-1),runitem['topics'][0]);matching=parts[(parts.year==runitem['year'])&(parts.topic_id==first['topic_id'])]
  for _,p in matching.head(2).iterrows():
   examples.append({'year':runitem['year'],'topic':first['topic_id'],'keywords':first['keywords'],'document_id':p[f['document_id']],'segment_id':p['segment_id'],'start':int(p[f['start_char']]),'end':int(p[f['end_char']]),'excerpt':p[f['segment_text']]})
report={'dataset':'北京上海六份真实政府工作报告，功能试用，非政策变化结论','elapsed_seconds':round(time.perf_counter()-start,2),'source_sha256':hashlib.sha256(raw).hexdigest(),'source_unchanged':hashlib.sha256(source.read_bytes()).hexdigest()==hashlib.sha256(raw).hexdigest(),'project_version':opened['project_info']['version'],'project_bytes':len(project),'export_bytes':len(export['data']),'old_global_result_and_annual_panel_unchanged':True,'all_yearly_excerpts_verified':True,'reopened_equal':True,'results':[{'year':x['year'],'documents':x['documents'],'segments':x['segments'],'status':x['status'],'non_outlier_topics':x['non_outlier_topics'],'outliers':x['outliers'],'error':x['error'],'timings':x['timings_seconds'],'topics':x['topics']} for x in opened['yearly_analysis']['results']],'examples':examples,'model':opened['yearly_analysis']['model_id'],'revision':opened['yearly_analysis']['model_revision'],'device':'cpu'}
(out/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
