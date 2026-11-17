"""Grouped annual chart inputs, recomputed from saved segments and document IDs."""
import copy
import hashlib
import io
import json
import math

import pandas as pd
from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text

from studio.annual import METRICS, RULE
from studio.annual_heatmap import ALL
from studio.plot_style import resolve_appearance, style_axes
from studio.topic_groups import annual, mapping
from studio.visualization import missing_glyphs

ANNUAL_KEYS=('year_field','group_field','exclude_missing_year','exclude_missing_group','include_outliers')


def basis(result,settings):
    return {'groups':copy.deepcopy((result.get('visualization') or {}).get('topic_groups',{}).get('groups',[])),
            'annual':{k:settings[k] for k in ANNUAL_KEYS}}


def signature(value):
    raw=json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def validate(result,visual,obj):
    if obj is None:return
    if not isinstance(obj,dict) or set(obj)!={'schema_version','basis','signature','line','heat'} or obj['schema_version']!=1 or not visual.get('annual_panel'):
        raise ValueError('归组年度图表设置格式无效。')
    b=obj['basis']
    if not isinstance(b,dict) or set(b)!={'groups','annual'} or not isinstance(b['groups'],list) or not isinstance(b['annual'],dict) or set(b['annual'])!=set(ANNUAL_KEYS):raise ValueError('归组年度图表依据无效。')
    if len(b['groups'])>100000 or obj['signature']!=signature(b):raise ValueError('归组年度图表依据校验失败。')
    if any(not isinstance(x,dict) or set(x)!={'group_id','name','members'} or not isinstance(x['group_id'],str) or not isinstance(x['name'],str) or not isinstance(x['members'],list) for x in b['groups']):raise ValueError('归组年度图表快照无效。')
    for key in ('line','heat'):
        item=obj[key]
        if not isinstance(item,dict) or set(item)!={'scopes','years','topics','metric'} or item['metric'] not in METRICS:raise ValueError('归组年度图表选项无效。')
        for name in ('scopes','years','topics'):
            seq=item[name]
            if not isinstance(seq,list) or len(seq)>10000 or len(set(seq))!=len(seq):raise ValueError('归组年度图表选项重复或超限。')
        if any(not isinstance(x,str) or len(x)>20000 for x in item['scopes']+item['topics']) or any(isinstance(x,bool) or not isinstance(x,int) or not 1000<=x<=9999 for x in item['years']):raise ValueError('归组年度图表筛选值无效。')
        if key=='heat' and len(item['scopes'])>1:raise ValueError('热力图只能选择一个分组。')


def display_labels(names, *, translate=str):
    """Find a unique, stable ID prefix for each displayed group; exports keep the full ID."""
    ids=[x for x in names if x.startswith('group-')];out={}
    for key,name in names.items():
        if key.startswith('group-'):
            length=8
            while length<32 and sum(other[6:6+length]==key[6:6+length] for other in ids)>1:length+=1
            out[key]=translate('{0} [组ID:{1}]').format(name,key[6:6+length])
        elif key=='topic:-1':out[key]=translate('离群主题 -1 · {0}').format(name)
        else:out[key]=translate('未归组主题 {0} · {1}').format(key.split(":",1)[-1],name)
    return out


def source(result,settings):
    docs,panel,parts=annual(result,settings);m=mapping(result);members={}
    for row in m.itertuples():members.setdefault(row.view_id,[]).append(int(row.original_topic_id))
    return docs,panel,parts,members


def for_scope(result,docs,panel,parts,members,scope):
    if scope!=ALL:return panel[panel.group_key.eq(scope)].copy()
    f=result['metadata']['generated_fields'];eligible=docs[docs.selected_by_metadata];rows=[]
    years=panel.year.drop_duplicates().tolist();categories=panel[['view_id','view_name','is_outlier']].drop_duplicates('view_id')
    for year in years:
        selected=eligible[eligible.year.isna() if pd.isna(year) else eligible.year.eq(year)]
        subset=parts[parts[f['document_id']].isin(selected.document_id)]
        sd=len(subset);dd=subset[f['document_id']].nunique()
        for cat in categories.itertuples(index=False):
            chosen=subset[subset[f['topic_id']].isin(members[cat.view_id])];sc=len(chosen);dc=chosen[f['document_id']].nunique()
            rows.append({'group':'全部语料','group_missing':False,'group_key':ALL,'year':None if pd.isna(year) else int(year),
                         'view_id':cat.view_id,'view_name':cat.view_name,'is_outlier':bool(cat.is_outlier),
                         'segment_count':sc,'segment_denominator':sd,'segment_share':sc/sd if sd else None,
                         'document_count':dc,'document_denominator':dd,'document_mention_rate':dc/dd if dd else None})
    return pd.DataFrame(rows,columns=panel.columns)


def cells(scoped,metric,years,topics,settings,members):
    if metric not in METRICS:raise ValueError('归组年度指标无效。')
    selected=scoped[scoped.year.isin(years)&scoped.view_id.isin(topics)];rows=[]
    for row in selected.itertuples(index=False):
        num=int(row.segment_count if metric.startswith('segment_') else row.document_count)
        den=int(row.segment_denominator if metric.startswith('segment_') else row.document_denominator)
        value=(num/den if metric in ('segment_share','document_mention_rate') else num) if den else None
        rows.append({'view':'人工主题归组','group':row.group,'group_key':row.group_key,'group_missing':bool(row.group_missing),
                     'year':int(row.year),'topic_group_id':row.view_id,'topic_group_name':row.view_name,
                     'member_topic_ids':json.dumps(sorted(members[row.view_id]),ensure_ascii=False),'is_outlier':bool(row.is_outlier),
                     'metric':metric,'metric_label':METRICS[metric],'numerator':num,'denominator':den,
                     'value':value,'plot_value':value,'missing_status':'有纳入样本' if den else '无纳入样本或分母为零',
                     'segment_count':int(row.segment_count),'segment_denominator':int(row.segment_denominator),
                     'document_count':int(row.document_count),'document_denominator':int(row.document_denominator),
                     **{k:settings[k] for k in ANNUAL_KEYS},'denominator_rule':RULE})
    return pd.DataFrame(rows)


def evidence(result,docs,parts,members,scope,year,view_id):
    f=result['metadata']['generated_fields'];selected=docs[docs.selected_by_metadata & docs.year.eq(year)]
    if scope!=ALL:selected=selected[selected.group_key.eq(scope)]
    seg=parts[parts[f['document_id']].isin(selected.document_id)&parts[f['topic_id']].isin(members[view_id])].copy()
    return selected[selected.document_id.isin(seg[f['document_id']])],seg


def line_arrays(frame):
    """Insert an explicit NaN at calendar gaps, preventing interpolation across missing years."""
    ordered=frame.sort_values('year');xs=[];ys=[];previous=None
    for row in ordered.itertuples():
        year=int(row.year)
        if previous is not None and year>previous+1:xs.append(previous+1);ys.append(float('nan'))
        xs.append(year);ys.append(float(row.plot_value) if pd.notna(row.plot_value) else float('nan'));previous=year
    return xs,ys


def render_line(frame,width,height,dpi,font_path,appearance=None,fmt='png', *, translate=str):
    if frame.empty or fmt not in ('png','svg') or not font_path:raise ValueError('请选择图表范围并确认中文字体。')
    if not (4<=width<=16 and 3<=height<=12 and dpi in (150,300,600)) or width*height*dpi*dpi>36_000_000:raise ValueError('图表尺寸或分辨率超限。')
    cfg=resolve_appearance(appearance);names=display_labels(frame.drop_duplicates('topic_group_id').set_index('topic_group_id').topic_group_name.to_dict(),translate=translate)
    series=list(frame.groupby(['group_key','topic_group_id'],sort=True))
    if len(series)>12:raise ValueError('图例超过 12 条，请减少分组或主题。')
    fig=Figure(figsize=(width,height),layout='constrained',facecolor='white');FigureCanvasAgg(fig);ax=fig.subplots()
    colors=['#628ea3','#70a79e','#a98c83','#7d82a7','#8d9e6d','#ad9271'] if cfg['style']=='color' else ['#222222','#555555','#888888','#333333']
    styles=['-','--','-.',':'];markers=['o','s','^','D','v','P']
    for i,((g,topic),df) in enumerate(series):
        xs,ys=line_arrays(df);name=f"{df.group.iloc[0]} / {names[topic]}"
        ax.plot(xs,ys,color=colors[i%len(colors)],linestyle=styles[(i//len(colors))%len(styles)],marker=markers[i%len(markers)],markersize=3.5,linewidth=1,label=name)
    metric=frame.metric.iloc[0];ax.set_xlabel(translate('年份'));ax.set_ylabel(translate('人工主题归组')+' · '+translate(METRICS[metric]))
    years=sorted(int(y) for y in frame.year.unique());ax.set_xticks(years,[str(y) for y in years],rotation=45 if len(years)>8 else 0)
    ax.set_xlim(years[0]-.5,years[-1]+.5);ax.grid(axis='y',color='#e5e9e8',linewidth=.5)
    if cfg['show_title']:ax.set_title(translate('人工主题归组年度变化（历史数据）'),loc='left',fontsize=cfg['font_size']+1)
    style_axes(ax,cfg);ax.legend(fontsize=max(7,cfg['font_size']-2),loc='upper left',bbox_to_anchor=(1.01,1))
    for text in fig.findobj(match=Text):text.set_fontproperties(FontProperties(fname=font_path,size=text.get_fontsize(),weight=text.get_fontweight()))
    fig.canvas.draw();box=fig.get_tightbbox(fig.canvas.get_renderer())
    if box.x0<-.02 or box.y0<-.02 or box.x1>width+.02 or box.y1>height+.02:raise ValueError('标签可能被裁切，请增大画布或减少选择。')
    out=io.BytesIO()
    with rc_context({'svg.fonttype':'path'}):fig.savefig(out,format=fmt,dpi=dpi)
    fig.clear();return out.getvalue(),missing_glyphs(font_path,list(names.values())+frame.group.tolist()+[METRICS[metric],'人工主题归组年度变化'])
