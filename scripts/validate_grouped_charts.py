"""Independent checks of grouped annual chart inputs on an existing synthetic project."""
import copy
import io
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from PIL import Image
from studio.annual import calculate,group_key
from studio.annual_heatmap import ALL,render as render_heat
from studio.grouped_charts import basis,cells,for_scope,line_arrays,render_line,signature,source
from studio.project import open_project,save_project
from studio.runtime import ROOT
from studio.topic_groups import create
from studio.visualization import find_chinese_font
from studio.workflow import csv_bytes

out=ROOT/'outputs/grouped_charts_validation';out.mkdir(parents=True,exist_ok=True)
p=open_project((ROOT/'outputs/annual_validation/synthetic_annual.hpts.zip').read_bytes());settings=p['visualization']['annual_panel']
original=p['segments'].copy(deep=True);original_topics=p['topics'].copy(deep=True);original_metadata=copy.deepcopy(p['metadata']);original_weights=copy.deepcopy(p['visualization'].get('keyword_weights'));original_panel=calculate(p,settings)[1];gid=create(p,'合成年度归组',[0,1]);docs,panel,parts,members=source(p,settings)
years=sorted(int(x) for x in panel.year.dropna().unique());scope=group_key('北京');topics=[gid]+[t for t in members if t.startswith('topic:')][:2]
scoped=for_scope(p,docs,panel,parts,members,scope);f=p['metadata']['generated_fields']
for metric in ('segment_count','segment_share','document_count','document_mention_rate'):
    frame=cells(scoped,metric,years,topics,settings,members)
    for row in frame.itertuples():
        selected=docs[docs.selected_by_metadata & docs.group_key.eq(scope) & docs.year.eq(row.year)]
        seg=parts[parts[f['document_id']].isin(selected.document_id)];chosen=seg[seg[f['topic_id']].isin(members[row.topic_group_id])]
        num=len(chosen) if metric.startswith('segment_') else chosen[f['document_id']].nunique()
        den=len(seg) if metric.startswith('segment_') else seg[f['document_id']].nunique()
        assert (row.numerator,row.denominator)==(num,den)
        assert (pd.isna(row.value) if den==0 else abs(float(row.value)-(num/den if metric in ('segment_share','document_mention_rate') else num))<1e-12)
frame=cells(scoped,'segment_share',years,topics,settings,members);(out/'heatmap.csv').write_bytes(csv_bytes(frame));(out/'line.csv').write_bytes(csv_bytes(frame))
read=pd.read_csv(io.BytesIO((out/'heatmap.csv').read_bytes()),keep_default_na=False)
for i,row in read.iterrows():
    expected=frame.iloc[i];assert row.topic_group_id==expected.topic_group_id and row.member_topic_ids==expected.member_topic_ids and int(row.denominator)==expected.denominator
    assert (row.value=='' and pd.isna(expected.value)) or abs(float(row.value)-expected.value)<1e-12
font,family=find_chinese_font();stats={};matrix=frame.assign(topic_id=frame.topic_group_id,topic_label=frame.topic_group_name)
for style in ('color','mono'):
    for chart in ('line','heat'):
        for fmt in ('png','svg'):
            if chart=='line':blob,missing=render_line(frame,10,6,150,font,{'style':style},fmt)
            else:blob,missing=render_heat(matrix,10,6,150,font,{'style':style},fmt,grouped=True)
            assert not missing
            file=out/f'{style}_{chart}.{fmt}';file.write_bytes(blob);stats[f'{style}_{chart}_{fmt}']={'bytes':len(blob)}
            if fmt=='png':
                image=Image.open(io.BytesIO(blob));image.verify();image=Image.open(io.BytesIO(blob));stats[f'{style}_{chart}_{fmt}']['pixels']=image.size
            else:assert b'<svg' in blob
b=basis(p,settings);conf={'schema_version':1,'basis':b,'signature':signature(b),
    'line':{'scopes':[scope],'years':years,'topics':topics,'metric':'segment_share'},
    'heat':{'scopes':[scope],'years':years,'topics':topics,'metric':'segment_share'}}
p['visualization']['grouped_annual_charts']=conf;p['visualization']['annual_view']='grouped'
raw=save_project(p);(out/'synthetic_grouped_charts.hpts.zip').write_bytes(raw);back=open_project(raw)
assert back['project_info']['version']==4 and back['visualization']['grouped_annual_charts']==conf and back['visualization']['annual_view']=='grouped'
pd.testing.assert_frame_equal(original,p['segments']);pd.testing.assert_frame_equal(original_topics,p['topics']);assert original_metadata==p['metadata'] and original_weights==p['visualization'].get('keyword_weights');pd.testing.assert_frame_equal(original_panel,calculate(p,settings)[1])
d,panel2,parts2,mem=source(back,settings);again=cells(for_scope(back,d,panel2,parts2,mem,scope),'segment_share',years,topics,settings,mem);pd.testing.assert_frame_equal(frame,again)
report={'synthetic':True,'input_documents':len(docs),'modeled_segments':len(original),'topic_group_id':gid,'members':[0,1],
        'scope':'北京','years':years,'cells':len(frame),'no_samples':int(frame.value.isna().sum()),'font':family,
        'independent_segment_and_unique_document_counts_verified':True,'csv_values_and_ids_verified':True,'original_panel_and_assignments_unchanged':True,
        'project_reopen_settings_and_cells_equal':True,'project_format_version':back['project_info']['version'],'outputs':stats}
(out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
