"""Validate requested exports and edits on six real reports; no model calls."""
import copy,hashlib,io,json,uuid
from pathlib import Path
import pandas as pd
from openpyxl import load_workbook
from studio.annual import calculate
from studio.deferred_exports import create_export,fingerprint
from studio.project import open_project
from studio.runtime import ROOT
from studio.topic_groups import trace,update
from studio.topic_names import materialize,rename
from studio.workflow import csv_bytes

root=ROOT/'outputs/deferred_export_validation';root.mkdir(parents=True,exist_ok=True)
source=ROOT/'outputs/project_compact_validation/real_policy_grouped_v5.hpts.zip'
source_sha=hashlib.sha256(source.read_bytes()).hexdigest();r=open_project(source.read_bytes());before=copy.deepcopy(r)
f=r['metadata']['generated_fields'];view,_=materialize(r);settings=r['visualization']['annual_panel'];old_panel=calculate(r,settings)[1]
segment=create_export(r,'segments');expected=csv_bytes(view['segments']);assert segment['data']==expected
csv_sha=hashlib.sha256(expected).hexdigest();del expected
network=create_export(r,'group_trace');assert network['data']==csv_bytes(trace(r))
excel=create_export(r,'excel');book=load_workbook(io.BytesIO(excel['data']),read_only=True,data_only=False)
for sheet_name,frame in [('records',view['rows']),('segments',view['segments'])]:
 rows=book[sheet_name].values;headers=list(next(rows));cols=r['metadata']['original_columns']+[f['document_id'],f['original_text']]
 if sheet_name=='segments':cols += [f[x] for x in ('segment_id','segment_order','start_char','end_char','segment_text','topic_id')]
 count=0
 for i,row in enumerate(rows):
  count+=1
  for col in cols:
   actual=row[headers.index(col)];expected_value=frame.iloc[i][col]
   assert actual==expected_value or (actual is None and expected_value==''),(sheet_name,i,col)
 assert count==len(frame)
book.close();z=create_export(r,'project');reopened=open_project(z['data'])
for key in ('rows','segments','topics'):
 left=json.loads(r[key].to_json(orient='split',force_ascii=False,double_precision=15))
 right=json.loads(reopened[key].to_json(orient='split',force_ascii=False,double_precision=15))
 assert left['columns']==right['columns'] and left['data']==right['data'],key
assert r['visualization']==reopened['visualization'] and r['topic_edits']==reopened['topic_edits']
old_stamp=fingerprint(r);rename(r,40,'更新后主题名称（验证）');gid=r['visualization']['topic_groups']['groups'][0]['group_id']
update(r,gid,'更新后归组（验证）',[8,40,56]);assert old_stamp!=fingerprint(r)
updated=create_export(r,'project');back=open_project(updated['data'])
assert back['topic_edits']['labels']['40']=='更新后主题名称（验证）'
assert back['visualization']['topic_groups']['groups'][0]['name']=='更新后归组（验证）'
new_segment=create_export(r,'segments');assert '更新后主题名称（验证）'.encode() in new_segment['data']
new_trace=create_export(r,'group_trace');assert '更新后归组（验证）'.encode() in new_trace['data']
assert hashlib.sha256(source.read_bytes()).hexdigest()==source_sha
for key in ('rows','segments','topics'):
 pd.testing.assert_frame_equal(r[key],before[key])
pd.testing.assert_frame_equal(calculate(r,settings)[1].drop(columns=['topic_label']),old_panel.drop(columns=['topic_label']))
report={'documents':len(r['rows']),'segments':len(r['segments']),'project_version':back['project_info']['version'],
        'source_unchanged':True,'segment_csv_byte_equal':True,'segment_csv_sha256':csv_sha,
        'group_trace_csv_byte_equal':True,'excel_document_and_segment_fields_roundtrip':True,
        'zip_reopened_data_and_settings_equal':True,'name_and_group_updated_in_reprepared_files':True,
        'data_assignments_and_annual_numbers_unchanged':True,
        'files':{name:{'bytes':len(item['data']),'sha256':item['content_sha256'],'filename':item['filename']}
                 for name,item in [('segments',segment),('group_trace',network),('excel',excel),('project',z)]}}
(root/'real_export_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
