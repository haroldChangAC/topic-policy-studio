"""Technical run using a previously modeled, explicitly synthetic project."""
import io
import json
from pathlib import Path

from PIL import Image
import pandas as pd
from unittest.mock import patch
from matplotlib.axes import Axes

from studio.annual import RULE,calculate,render,validate_settings
from studio.project import open_project,save_project
from studio.visualization import find_chinese_font
from studio.workflow import csv_bytes

root=Path(__file__).resolve().parents[1];out=root/'outputs/annual_validation';out.mkdir(parents=True,exist_ok=True)
project=open_project((root/'outputs/browser_chart_project.hpts.zip').read_bytes())
settings={'year_field':'年份','group_field':'地区','exclude_missing_year':True,'exclude_missing_group':True,
          'include_outliers':True,'metric':'segment_share','selected_groups':[],'selected_topics':[],'rule_version':1}
docs,panel,parts=calculate(project,settings)
settings['selected_groups']=docs.loc[docs.selected_by_metadata,'group_key'].drop_duplicates().tolist()[:2]
settings['selected_topics']=sorted(int(x) for x in panel.topic_id.unique() if x!=-1)[:3]
project['visualization']['annual_panel']=settings;validate_settings(project,settings)
f=project['metadata']['generated_fields'];lookup=docs.set_index('document_id')
for row in panel.itertuples(index=False):
    selected=docs[docs.group_key.eq(row.group_key)&docs.year.eq(row.year)&docs.selected_by_metadata]
    seg=parts[parts[f['document_id']].isin(selected.document_id)]
    topic=seg[seg[f['topic_id']].eq(row.topic_id)]
    assert row.segment_denominator==len(seg) and row.segment_count==len(topic)
    assert row.document_denominator==seg[f['document_id']].nunique() and row.document_count==topic[f['document_id']].nunique()
    assert (pd.isna(row.segment_share) if not len(seg) else abs(row.segment_share-len(topic)/len(seg))<1e-12)
    for _,part in topic.iterrows():
        source=project['rows'].set_index(f['document_id']).loc[part[f['document_id']],f['original_text']]
        assert source[int(part[f['start_char']]):int(part[f['end_char']])]==part[f['segment_text']]
raw=save_project(project);(out/'synthetic_annual.hpts.zip').write_bytes(raw);reopened=open_project(raw)
_,again,_=calculate(reopened,settings);pd.testing.assert_frame_equal(panel,again)
(out/'annual_panel.csv').write_bytes(csv_bytes(panel.drop(columns=['group_key'])))
(out/'settings.json').write_text(json.dumps({'settings':settings,'formula':RULE},ensure_ascii=False,indent=2)+'\n')
font,family=find_chinese_font();info={}
plotted=[]
original_plot=Axes.plot
def capture_plot(ax,x,y,*args,**kwargs):
    plotted.append((list(x),list(y),kwargs.get('label')))
    return original_plot(ax,x,y,*args,**kwargs)
with patch.object(Axes,'plot',capture_plot):
    render(panel,'segment_share',settings['selected_groups'],settings['selected_topics'],8,5,150,font)
selected=panel[panel.group_key.isin(settings['selected_groups']) & panel.topic_id.isin(settings['selected_topics']) & panel.year.notna()]
expected=[]
for _,df in selected.groupby(['group','group_missing','topic_id','topic_label'],sort=True,dropna=False):
    df=df.sort_values('year');expected.append((df.year.astype(int).tolist(),df.segment_share.astype(float).tolist()))
assert len(plotted)==len(expected)
for (xs,ys,_),(ex,ey) in zip(plotted,expected):
    assert xs==ex and all((pd.isna(a) and pd.isna(b)) or a==b for a,b in zip(ys,ey))
for fmt in ['png','svg']:
    data,missing=render(panel,'segment_share',settings['selected_groups'],settings['selected_topics'],8,5,150,font,fmt=fmt)
    (out/f'annual.{fmt}').write_bytes(data);info[fmt]={'bytes':len(data),'missing_glyphs':missing}
    if fmt=='png':
        im=Image.open(io.BytesIO(data));im.verify();im=Image.open(io.BytesIO(data));info[fmt]['size']=im.size;info[fmt]['dpi']=im.info.get('dpi')
    else:assert b'<svg' in data
summary={'synthetic':True,'source':'outputs/browser_chart_project.hpts.zip；既有合成语料离线 BERTopic 项目','documents':len(docs),
         'segments':len(parts),'panel_rows':len(panel),'groups':docs.group.drop_duplicates().tolist(),
         'years':sorted(int(x) for x in docs.year.dropna().unique()),'formula':RULE,'chart':info,'font':family,
         'project_roundtrip':True,'independent_counts_and_offsets':True,'plotted_values_match_panel':True,'settings':settings}
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False,indent=2))
