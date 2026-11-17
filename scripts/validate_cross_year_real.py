"""Check adjacent-year links using the six-report project without model calls."""
import copy, json, hashlib, io, zipfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from studio.runtime import ROOT
from studio.project import open_project,save_project
from studio.cross_year import calculate,defaults,evidence,graph
from studio.deferred_exports import create_export
from studio.annual import calculate as annual
src=ROOT/'outputs/yearly_trial/six_reports_yearly.hpts.zip';out=ROOT/'outputs/cross_year_trial';out.mkdir(parents=True,exist_ok=True)
raw=src.read_bytes();r=open_project(raw);before=copy.deepcopy(r);settings=defaults(r)
with patch('socket.socket.connect',side_effect=AssertionError('offline')):
 nodes,frame,excluded,warnings=calculate(r,settings)
 assert not excluded and not warnings
 for row in frame.itertuples():
  a=next(n for n in nodes if n['year']==row.source_year and n['topic_id']==row.source_topic_id)
  b=next(n for n in nodes if n['year']==row.target_year and n['topic_id']==row.target_topic_id)
  av=np.asarray(a['vector']);bv=np.asarray(b['vector']);expected=av.dot(bv)/np.linalg.norm(av)/np.linalg.norm(bv)
  assert abs(expected-row.similarity)<1e-12
 drawn=frame[frame.in_graph];fig=graph(nodes,frame,settings)
 if fig is not None:assert list(fig.data[0].link.value)==drawn.line_weight.tolist()
 f=r['metadata']['generated_fields'];examples=[]
 for row in drawn.head(4).itertuples():
  ends=[]
  for yr,topic in [(row.source_year,row.source_topic_id),(row.target_year,row.target_topic_id)]:
   parts=evidence(r,yr,topic);seg=parts.iloc[0];doc=r['rows'].set_index(f['document_id']).loc[seg[f['document_id']]]
   start,end=int(seg[f['start_char']]),int(seg[f['end_char']]);assert doc[f['original_text']][start:end]==seg[f['segment_text']]
   ends.append({'year':int(yr),'topic_id':int(topic),'document_id':seg[f['document_id']],'segment_id':seg[f['segment_id']],'start':start,'end':end,'excerpt':seg[f['segment_text']]})
  examples.append({'source_keywords':row.source_keywords,'target_keywords':row.target_keywords,'similarity':row.similarity,'ends':ends})
 r['visualization']['cross_year_links']=settings
 data=save_project(r);reopened=open_project(data);assert reopened['visualization']['cross_year_links']==settings
 n2,f2,e2,w2=calculate(reopened,settings);pd.testing.assert_frame_equal(frame,f2)
 for name in ('rows','segments','topics'):pd.testing.assert_frame_equal(before[name],r[name])
 assert before['yearly_analysis']==r['yearly_analysis']
 pd.testing.assert_frame_equal(annual(before,before['visualization']['annual_panel'])[1],annual(r,r['visualization']['annual_panel'])[1])
 export=create_export(reopened,'cross_year');(out/'relations.zip').write_bytes(export['data']);(out/'relations_project.hpts.zip').write_bytes(data)
 with zipfile.ZipFile(io.BytesIO(export['data'])) as z:
  ef=pd.read_csv(io.BytesIO(z.read('graph_edges.csv')))
  assert len(ef)==len(drawn) and np.allclose(ef['line_weight'].to_numpy(),drawn.line_weight.astype(float).to_numpy())
  assert 'graph.html' in z.namelist() if not drawn.empty else 'graph.html' not in z.namelist()
 details=[]
 for y in (2022,2023):
  subset=frame[frame.source_year.eq(y)];d=subset[subset.in_graph]
  details.append({'years':[y,y+1],'selected_source_topics':sum(n['year']==y for n in nodes),'selected_target_topics':sum(n['year']==y+1 for n in nodes),'compared_pairs':len(subset),'above_threshold':int((subset.passes_threshold & subset.similarity.gt(0)).sum()),'drawn_edges':len(d)})
 report={'model_id':r['yearly_analysis']['model_id'],'revision':r['yearly_analysis']['model_revision'],'settings':settings,'pairs':details,'examples':examples,'vector_pairs_independently_checked':len(frame),'project_version':reopened['project_info']['version'],'source_unchanged':hashlib.sha256(src.read_bytes()).digest()==hashlib.sha256(raw).digest(),'global_results_annual_unchanged':True,'zip_roundtrip':True,'html_and_csv_checked':True,'source':'六份真实政府工作报告，技术试用而非政策变化结论'}
 (out/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
