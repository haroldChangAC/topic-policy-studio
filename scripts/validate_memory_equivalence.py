"""Verify real v4/v5 projects, exports and transfer after memory optimization."""
import hashlib,io,json,uuid
from pathlib import Path
import pandas as pd
from openpyxl import load_workbook
from studio.annual import calculate
from studio.grouped_charts import source,for_scope,cells
from studio.policy_project import open_policy_project,save_policy_project
from studio.policy_transfer import apply_transfer,plan_transfer
from studio.project import open_project,save_project
from studio.runtime import ROOT
from studio.topic_groups import mapping,summary
from studio.workflow import csv_bytes,excel_bytes

out=ROOT/'outputs/memory_validation';out.mkdir(parents=True,exist_ok=True)
old_path=ROOT/'outputs/real_policy_2022_2024/real_policy_grouped.hpts.zip'
current_path=ROOT/'outputs/project_compact_validation/real_policy_grouped_v5.hpts.zip'
before_hash=hashlib.sha256(old_path.read_bytes()).hexdigest();v5_hash=hashlib.sha256(current_path.read_bytes()).hexdigest()
a=open_project(old_path.read_bytes());b=open_project(current_path.read_bytes());f=b['metadata']['generated_fields']
for k in ('rows','segments','topics'):
 x=json.loads(a[k].to_json(orient='split',force_ascii=False,double_precision=15));y=json.loads(b[k].to_json(orient='split',force_ascii=False,double_precision=15))
 assert x['columns']==y['columns'] and x['data']==y['data'],k
assert a['metadata']==b['metadata'] and a['topic_edits']==b['topic_edits'] and a['visualization']==b['visualization']
for _,p in b['segments'].iterrows():
 doc=b['rows'].set_index(f['document_id']).loc[p[f['document_id']]]
 assert doc[f['original_text']][p[f['start_char']]:p[f['end_char']]]==p[f['segment_text']]
 assert all(p[c]==doc[c] for c in b['metadata']['original_columns'])
settings=b['visualization']['annual_panel'];pd.testing.assert_frame_equal(calculate(a,settings)[1],calculate(b,settings)[1]);pd.testing.assert_frame_equal(mapping(a),mapping(b));pd.testing.assert_frame_equal(summary(a),summary(b))
ad,ap,aseg,am=source(a,settings);bd,bp,bseg,bm=source(b,settings);pd.testing.assert_frame_equal(ap,bp);assert am==bm
chart=b['visualization']['grouped_annual_charts']
for k in ('line','heat'):
 conf=chart[k]
 for scope in conf['scopes']:
  left=cells(for_scope(a,ad,ap,aseg,am,scope),conf['metric'],conf['years'],conf['topics'],settings,am)
  right=cells(for_scope(b,bd,bp,bseg,bm,scope),conf['metric'],conf['years'],conf['topics'],settings,bm)
  pd.testing.assert_frame_equal(left,right)
print('真实项目文档、片段及年度归组一致',flush=True)
source_csv=csv_bytes(a['segments']);new_csv=csv_bytes(b['segments']);assert source_csv==new_csv
csv_hash=hashlib.sha256(new_csv).hexdigest();csv_size=len(new_csv);del source_csv,new_csv
excel=excel_bytes(b);wb=load_workbook(io.BytesIO(excel),read_only=True)
for sheet_name,frame in (('records',b['rows']),('segments',b['segments'])):
 it=wb[sheet_name].values;headers=list(next(it));columns=b['metadata']['original_columns']+[f['document_id'],f['original_text']]
 if sheet_name=='segments':columns += [f[x] for x in ('segment_id','segment_order','start_char','end_char','segment_text','topic_id')]
 count=0
 for i,row in enumerate(it):
  count+=1
  for col in columns:
   actual=row[headers.index(col)];expected=frame.iloc[i][col]
   assert actual==expected or (actual is None and expected==''),(sheet_name,i,col)
 assert count==len(frame)
wb.close();excel_size=len(excel);print('CSV 逐字节、Excel 原文和位置回读一致',flush=True)
doc=b['rows'].iloc[0];pid=doc[f['document_id']]
project=apply_transfer(plan_transfer(b,v5_hash,[pid],{'title':'标题','date':'官网刊载日期','region':'分组','source':'来源网址'}))
policy=open_policy_project(save_policy_project(project));assert policy['policies'][0]['original_text']==doc[f['original_text']]
assert len(policy['policies'][0]['theme_origin']['segments'])==int(b['segments'][f['document_id']].eq(pid).sum())
new=save_project(b);new_path=out/f'reopened-{uuid.uuid4().hex[:12]}.hpts.zip';new_path.write_bytes(new);again=open_project(new)
assert again['visualization']==b['visualization'] and again['topic_edits']==b['topic_edits']
assert again['segments'][f['segment_text']].tolist()==b['segments'][f['segment_text']].tolist()
assert hashlib.sha256(old_path.read_bytes()).hexdigest()==before_hash and hashlib.sha256(current_path.read_bytes()).hexdigest()==v5_hash
report={'real_corpus':True,'documents':len(b['rows']),'segments':len(b['segments']),'old_version':4,'new_version':5,
        'all_table_columns_and_values_equal':True,'all_offsets_and_original_fields_equal':True,
        'topic_edits_groups_annual_chart_inputs_equal':True,'csv_byte_identical':True,'csv_bytes':csv_size,'csv_sha256':csv_hash,
        'xlsx_bytes':excel_size,'xlsx_all_document_fields_and_offsets_checked':True,
        'policy_transfer_document_and_segments_checked':True,'old_files_unchanged':True,'new_project':str(new_path.relative_to(ROOT)),
        'new_project_bytes':len(new),'project_reopened_without_model':True}
(out/'real_equivalence.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
