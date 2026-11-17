"""Validate against the existing, explicitly synthetic offline BERTopic project."""
import io
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from PIL import Image

from studio.annual import RULE,calculate,group_key
from studio.annual_heatmap import ALL,data,evidence,render,samples
from studio.project import open_project,save_project
from studio.runtime import ROOT
from studio.visualization import find_chinese_font
from studio.workflow import csv_bytes

out=ROOT/'outputs/annual_heatmap_validation';out.mkdir(parents=True,exist_ok=True)
source=ROOT/'outputs/annual_validation/synthetic_annual.hpts.zip';project=open_project(source.read_bytes())
settings=project['visualization']['annual_panel'];docs,panel,parts=calculate(project,settings)
years=sorted(panel.year.dropna().astype(int).unique().tolist());topics=sorted(panel.topic_id.astype(int).unique().tolist())
heat=data(panel,'segment_share',ALL,years,topics);sample=samples(project,docs,ALL,years)
f=project['metadata']['generated_fields']
for row in heat.itertuples(index=False):
    eligible=docs[docs.selected_by_metadata & docs.year.eq(row.year)];seg=parts[parts[f['document_id']].isin(eligible.document_id)]
    chosen=seg[seg[f['topic_id']].eq(row.topic_id)]
    assert (row.segment_count,row.segment_denominator)==(len(chosen),len(seg))
    assert (row.document_count,row.document_denominator)==(chosen[f['document_id']].nunique(),seg[f['document_id']].nunique())
    matched,found=evidence(project,docs,parts,ALL,row.year,row.topic_id)
    assert len(found)==len(chosen) and len(matched)==chosen[f['document_id']].nunique()
    for _,segment in found.iterrows():
        original=project['rows'].set_index(f['document_id']).loc[segment[f['document_id']],f['original_text']]
        assert original[int(segment[f['start_char']]):int(segment[f['end_char']])]==segment[f['segment_text']]
    assert (pd.isna(row.value) if not len(seg) else abs(row.value-len(chosen)/len(seg))<1e-12)
obj={'schema_version':1,'group_key':ALL,'years':years,'topics':topics,'metric':'segment_share'}
project['visualization']['annual_heatmap']=obj
raw=save_project(project);(out/'synthetic_heatmap.hpts.zip').write_bytes(raw);reopened=open_project(raw)
_,again,_=calculate(reopened,settings);pd.testing.assert_frame_equal(panel,again);pd.testing.assert_frame_equal(heat,data(again,'segment_share',ALL,years,topics))
export=heat.drop(columns=['group_key']).copy()
for k in ('year_field','group_field','exclude_missing_year','exclude_missing_group','include_outliers'):export[k]=settings[k]
export['denominator_rule']=RULE;(out/'heatmap.csv').write_bytes(csv_bytes(export));(out/'samples.csv').write_bytes(csv_bytes(sample))
frame=pd.read_csv(io.BytesIO((out/'heatmap.csv').read_bytes()),keep_default_na=False)
assert len(frame)==len(heat)
for i,row in frame.iterrows():
    expected=heat.iloc[i];assert int(row.numerator)==int(expected.numerator) and int(row.denominator)==int(expected.denominator)
    assert row.missing_status==expected.missing_status
    if pd.isna(expected.value):assert row.value==''
    else:assert abs(float(row.value)-float(expected.value))<1e-12
font,family=find_chinese_font();output={};plotted=[];original_plot=Axes.imshow
def spy(ax,values,*args,**kwargs):
    plotted.append(np.ma.filled(values,np.nan));return original_plot(ax,values,*args,**kwargs)
expected=heat.pivot(index='topic_id',columns='year',values='plot_value').sort_index().sort_index(axis=1).to_numpy(float)
for style in ['color','mono']:
    for fmt in ['png','svg']:
        with patch.object(Axes,'imshow',spy):content,missing=render(heat,9,6,150,font,{'style':style},fmt)
        assert np.allclose(plotted[-1],expected,equal_nan=True)
        (out/f'{style}.{fmt}').write_bytes(content);output[f'{style}_{fmt}']={'bytes':len(content),'missing_glyphs':missing}
        if fmt=='png':
            image=Image.open(io.BytesIO(content));image.verify();image=Image.open(io.BytesIO(content));output[f'{style}_{fmt}']['pixels']=image.size;output[f'{style}_{fmt}']['dpi']=image.info.get('dpi')
        else:assert b'<svg' in content
missing_frame=data(panel,'segment_share',group_key('北京'),years,topics)
assert missing_frame['plot_value'].isna().any() and (missing_frame['plot_value'].eq(0)).any()
for style in ['color','mono']:
    content,missing=render(missing_frame,9,6,150,font,{'style':style},'png')
    (out/f'{style}_missing.png').write_bytes(content)
    assert not missing
report={'synthetic':True,'input_project':str(source.relative_to(ROOT)),'input_documents':len(docs),'modeled_segments':len(project['segments']),
        'settings':settings,'heatmap_settings':obj,'sample_table':sample.to_dict(orient='records'),'heatmap_rows':len(heat),
        'panel_and_chart_and_csv_match':True,'evidence_offsets_checked':True,'project_roundtrip':True,'font':family,'output':output}
(out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
