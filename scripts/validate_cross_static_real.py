"""Validate publication Sankey exports against the six-report analysis."""
import io, json, hashlib, zipfile, xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from studio.runtime import ROOT,configure
from studio.project import open_project,save_project
from studio.cross_year import calculate,graph
from studio.cross_year_static import render,selected_data,default_export
from studio.deferred_exports import create_export
configure();out=ROOT/'outputs/cross_static_validation';out.mkdir(parents=True,exist_ok=True)
src=ROOT/'outputs/cross_year_trial/relations_project.hpts.zip';original=src.read_bytes();r=open_project(original)
s=r['visualization']['cross_year_links'];nodes,frame,excluded,warnings=calculate(r,s);active,edges=selected_data(nodes,frame);interactive=graph(nodes,frame,s)
frame_before=frame.copy(deep=True)
assert [e['weight'] for e in edges]==list(interactive.data[0].link.value)
assert [e['weight'] for e in edges]==frame.loc[frame.in_graph,'line_weight'].astype(float).tolist()
r['visualization']['cross_year_export']=default_export();checks={}
for style in ('color','mono'):
 r['visualization']['cross_year_links']['style']=style
 for fmt in ('png','svg'):
  data,meta=render(nodes,frame,r['visualization']['cross_year_links'],default_export(),fmt)
  (out/f'sankey_{style}.{fmt}').write_bytes(data)
  assert meta['edge_weights']==[e['weight'] for e in edges] and meta['node_ids']==[[n['year'],n['topic_id']] for n in active]
  if fmt=='png':
   im=Image.open(io.BytesIO(data));assert im.size==(2100,1350) and abs(im.info['dpi'][0]-150)<.1
   if style=='mono':
    rgb=np.asarray(im.convert('RGB'));assert np.array_equal(rgb[:,:,0],rgb[:,:,1]) and np.array_equal(rgb[:,:,1],rgb[:,:,2])
  else:assert ET.fromstring(data).tag.endswith('svg') and '相邻自然年份主题相似关联' in data.decode()
  checks[f'{style}_{fmt}']={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'metadata':meta}
r['visualization']['cross_year_links']['style']='color';high={'width':10.,'height':8.,'dpi':600,'font_size':7}
data,meta=render(nodes,frame,s,high,'png');(out/'sankey_600dpi.png').write_bytes(data)
im=Image.open(io.BytesIO(data));assert im.size==(6000,4800) and abs(im.info['dpi'][0]-600)<.1
bundle=create_export(r,'cross_year')['data'];(out/'data_html.zip').write_bytes(bundle)
with zipfile.ZipFile(io.BytesIO(bundle)) as z:
 html=z.read('graph.html').decode();(out/'graph.html').write_text(html)
 marker=html.rfind('Plotly.newPlot(');assert marker>=0
 parse=json.JSONDecoder();pos=marker+len('Plotly.newPlot(');_,n=parse.raw_decode(html[pos:].lstrip());start=pos+len(html[pos:])-len(html[pos:].lstrip())+n
 start=html.index(',',start)+1;payload,_=parse.raw_decode(html[start:].lstrip());trace=payload[0]
 assert trace['link']['value']==[e['weight'] for e in edges]
 assert trace['node']['label']==list(interactive.data[0].node.label)
 expected={((n['year'],n['topic_id'])):i for i,n in enumerate(active)}
 assert trace['link']['source']==[expected[e['source']] for e in edges]
 assert trace['link']['target']==[expected[e['target']] for e in edges]
 exported=pd.read_csv(io.BytesIO(z.read('graph_edges.csv')))
 assert np.allclose(exported['line_weight'].to_numpy(),np.array([e['weight'] for e in edges]))
 assert list(zip(exported.source_year,exported.source_topic_id,exported.target_year,exported.target_topic_id)) == [(a['source'][0],a['source'][1],a['target'][0],a['target'][1]) for a in edges]
 class Scripts(HTMLParser):
  def __init__(self):super().__init__();self.sources=[]
  def handle_starttag(self,t,attrs):
   if t=='script' and dict(attrs).get('src'):self.sources.append(dict(attrs)['src'])
 parser=Scripts();parser.feed(html);assert not parser.sources
saved=save_project(r);back=open_project(saved);assert back['visualization']['cross_year_export']==default_export()
pd.testing.assert_frame_equal(frame_before,frame)
assert hashlib.sha256(src.read_bytes()).digest()==hashlib.sha256(original).digest()
report={'project_documents':6,'segments':685,'selected_nodes':len(active),'edges':len(edges),'html_plotly_static_csv_nodes_and_edges_equal':True,
 'styles':checks,'dpi600':{'pixels':list(im.size),'dpi':list(im.info['dpi']),'bytes':len(data)},'html_inline_script':True,'external_script_tags':parser.sources,
 'project_roundtrip':True,'source_unchanged':True,'underlying_similarity_unchanged':True,'html_offline_browser_verified':False,'browser_static_download_verified':False}
(out/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
