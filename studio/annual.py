"""Auditable year/group/topic counts from saved document and segment assignments."""
import io
import json
import math
import re
import textwrap
from collections import Counter
from datetime import date, datetime

import pandas as pd
from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator

from studio.topic_names import materialize_topics
from studio.plot_style import resolve_appearance, style_axes, topic_label
from studio.visualization import missing_glyphs

METRICS = {'segment_count':'文本片段数', 'segment_share':'文本片段占比',
           'document_count':'文档提及数', 'document_mention_rate':'文档提及率'}
MAX_CELLS = 100000
RULE = ('片段分母为本组本年实际纳入的片段数；文档分母为本组本年至少有一个纳入片段的不同文档数。'
        '排除离群类时，-1 片段不计入分子及片段分母，只有离群片段的文档不计入文档分母；'
        '无建模片段的有效原文也不计入文档分母。文档可提及多个主题，各主题文档提及率之和可超过 100%。')


def parse_year(value):
    """Accept unambiguous years and validated ISO/Chinese calendar dates only."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return None if pd.isna(value) else int(value.year)
    if isinstance(value, (int, float)):
        if not math.isfinite(value) or int(value) != value or not 1000 <= int(value) <= 9999:
            return None
        return int(value)
    if not isinstance(value, str):
        return None
    s=value.strip()
    if re.fullmatch(r'\d{4}',s):
        y=int(s);return y if y >= 1000 else None
    match=re.fullmatch(r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?(?:Z|[+-]\d{2}:?\d{2})?)?',s)
    if not match:
        match=re.fullmatch(r'(\d{4})年(\d{1,2})月(\d{1,2})日',s)
    if not match:return None
    y,m,d=map(int,match.groups())
    try:
        date(y,m,d)
        if y<1000:return None
        # Validate time and offset as well; don't accept a malformed timestamp.
        if 'T' in s or (' ' in s and ':' in s):datetime.fromisoformat(s.replace('Z','+00:00').replace('/','-'))
        return y
    except ValueError:return None


def group_key(label,missing=False):
    return json.dumps([bool(missing),label],ensure_ascii=False,separators=(',',':'))


def validate_settings(result, obj):
    if obj is None:return
    required={'year_field','group_field','exclude_missing_year','exclude_missing_group','include_outliers',
              'metric','selected_groups','selected_topics','rule_version'}
    if not isinstance(obj,dict) or set(obj)!=required or obj['rule_version']!=1:
        raise ValueError('年度统计设置格式无效。')
    columns=result['metadata'].get('original_columns',[])
    if obj['year_field'] not in columns or (obj['group_field'] is not None and obj['group_field'] not in columns):
        raise ValueError('年度统计所选原始字段不存在。')
    if any(not isinstance(obj[x],bool) for x in ('exclude_missing_year','exclude_missing_group','include_outliers')):
        raise ValueError('年度统计缺失或离群设置无效。')
    if obj['metric'] not in METRICS or not isinstance(obj['selected_groups'],list) or not isinstance(obj['selected_topics'],list):
        raise ValueError('年度图表设置无效。')
    if len(obj['selected_groups'])>10000 or any(not isinstance(x,str) or len(x)>20000 for x in obj['selected_groups']) or len(set(obj['selected_groups']))!=len(obj['selected_groups']):
        raise ValueError('年度图表分组设置无效。')
    ids=set(result['topics']['Topic'].astype(int))
    if any(isinstance(x,bool) or not isinstance(x,int) or x not in ids for x in obj['selected_topics']) or len(set(obj['selected_topics']))!=len(obj['selected_topics']):
        raise ValueError('年度图表主题设置无效。')
    if not obj['include_outliers'] and -1 in obj['selected_topics']:
        raise ValueError('排除离群类时不能选择离群主题。')
    valid=set(prepare_documents(result,obj)['group_key'])
    if not set(obj['selected_groups']).issubset(valid):raise ValueError('年度图表分组不存在。')


def prepare_documents(result, settings):
    docs=result['rows'];segs=result['segments'];f=result['metadata']['generated_fields']
    required=[f['document_id'],f['original_text'],f['status'],settings['year_field']]
    if settings['group_field']:required.append(settings['group_field'])
    if any(c not in docs for c in required) or any(f[k] not in segs for k in ('document_id','segment_id','topic_id','start_char','end_char','segment_text')):
        raise ValueError('项目缺少年份字段、原始文档或片段关联，无法统计。')
    counts=Counter(segs[f['document_id']].tolist())
    included=Counter(segs.loc[segs[f['topic_id']].ne(-1) if not settings['include_outliers'] else pd.Series(True,index=segs.index),f['document_id']].tolist())
    records=[]
    for _,row in docs.iterrows():
        doc=row[f['document_id']];year=parse_year(row[settings['year_field']]);raw_group=row[settings['group_field']] if settings['group_field'] else '全部语料'
        missing=bool(settings['group_field'] and (raw_group is None or (isinstance(raw_group,float) and math.isnan(raw_group)) or not str(raw_group).strip()))
        group='（缺失分组）' if missing else str(raw_group)
        excluded_year=year is None and settings['exclude_missing_year']
        excluded_group=missing and settings['exclude_missing_group']
        selected=not excluded_year and not excluded_group
        total=counts[doc];used=included[doc] if selected else 0
        original=row[f['original_text']];valid=isinstance(original,str) and bool(original.strip())
        if excluded_year:reason='排除：年份缺失或无法解析'
        elif excluded_group:reason='排除：分组缺失'
        elif not total:reason='无建模片段（有效原文）' if valid else '无有效原文或建模片段'
        elif not used:reason='仅有离群片段，已排除'
        else:reason='纳入文档分母'
        records.append({'document_id':doc,'year':year,'year_missing':year is None,'group':group,'group_missing':missing,
                        'group_key':group_key(group,missing),'original_status':row[f['status']],
                        'valid_original_text':valid,'modeled_segments':total,'included_segments':used,
                        'included_document':bool(used),'reason':reason,'selected_by_metadata':selected})
    return pd.DataFrame(records,columns=['document_id','year','year_missing','group','group_missing','group_key',
                                        'original_status','valid_original_text','modeled_segments','included_segments',
                                        'included_document','reason','selected_by_metadata'])


def calculate(result, settings):
    docs=prepare_documents(result,settings);f=result['metadata']['generated_fields'];segs=result['segments']
    eligible=docs[docs['selected_by_metadata']]; ids=set(eligible['document_id'])
    parts=segs[segs[f['document_id']].isin(ids)].copy()
    if not settings['include_outliers']:parts=parts[parts[f['topic_id']].ne(-1)]
    groups=eligible[['group_key','group','group_missing']].drop_duplicates().sort_values(['group_missing','group'])
    years=sorted(int(y) for y in eligible['year'].dropna().unique())
    if eligible['year_missing'].any():years.append(None)
    topic_ids=sorted(int(x) for x in result['topics']['Topic'].tolist() if settings['include_outliers'] or int(x)!=-1)
    if len(groups)*len(years)*len(topic_ids)>MAX_CELLS:raise ValueError(f'面板超过 {MAX_CELLS} 行；请减少分组、年份或主题范围。')
    byid=docs.set_index('document_id'); segment_counts=Counter();doc_sets={};segment_den=Counter();document_den=Counter()
    for _,doc in eligible[eligible['included_document']].iterrows():
        y=None if doc['year_missing'] else int(doc['year']);document_den[(doc['group_key'],y)]+=1
    for _,part in parts.iterrows():
        doc=byid.loc[part[f['document_id']]];y=None if doc['year_missing'] else int(doc['year']);cell=(doc['group_key'],y)
        topic=int(part[f['topic_id']]);segment_den[cell]+=1;segment_counts[(*cell,topic)]+=1
        doc_sets.setdefault((*cell,topic),set()).add(part[f['document_id']])
    labels=materialize_topics(result).set_index('Topic')['Display_Name'].to_dict();out=[]
    for g in groups.itertuples(index=False):
        for year in years:
            cell=(g.group_key,year);sd=segment_den[cell];dd=document_den[cell]
            for topic in topic_ids:
                sc=segment_counts[(*cell,topic)];dc=len(doc_sets.get((*cell,topic),()))
                out.append({'group':g.group,'group_missing':bool(g.group_missing),'group_key':g.group_key,'year':year,
                            'topic_id':topic,'topic_label':labels[topic],'segment_count':sc,'segment_denominator':sd,
                            'segment_share':sc/sd if sd else None,'document_count':dc,'document_denominator':dd,
                            'document_mention_rate':dc/dd if dd else None})
    panel=pd.DataFrame(out,columns=['group','group_missing','group_key','year','topic_id','topic_label','segment_count',
                                    'segment_denominator','segment_share','document_count','document_denominator','document_mention_rate'])
    return docs,panel,parts


def render(panel,metric,groups,topics,width,height,dpi,font_path,appearance=None,fmt='png',include_outliers=True, *, translate=str):
    if metric not in METRICS or fmt not in ('png','svg'):raise ValueError('年度图表设置无效。')
    if not font_path:raise ValueError('未找到中文字体。')
    if not (4<=width<=16 and 3<=height<=12 and dpi in (150,300,600)) or width*height*dpi*dpi>36_000_000:raise ValueError('图表尺寸或分辨率超限。')
    selected=panel[panel['group_key'].isin(groups)&panel['topic_id'].isin(topics)&panel['year'].notna()].copy()
    if selected.empty:raise ValueError('所选范围没有可绘制的有效年份。')
    selected=selected.sort_values('year');cfg=resolve_appearance(appearance)
    fig=Figure(figsize=(width,height),layout='constrained',facecolor='white');FigureCanvasAgg(fig);ax=fig.subplots()
    colors=['#628ea3','#70a79e','#a98c83','#7d82a7','#8d9e6d','#ad9271'] if cfg['style']=='color' else ['#222222','#555555','#888888','#333333']
    styles=['-','--','-.',':'];markers=['o','s','^','D','v','P']
    lines=0
    for (group,missing,topic,label),df in selected.groupby(['group','group_missing','topic_id','topic_label'],sort=True,dropna=False):
        vals=df[metric].astype(float); name=f'{group}'+(translate('（缺失字段）') if missing else '')+f' / {topic_label(topic,label,translate=translate)}'
        ax.plot(df['year'].astype(int),vals,color=colors[lines%len(colors)],linestyle=styles[(lines//len(colors))%len(styles)],marker=markers[lines%len(markers)],markersize=3.5,linewidth=1.0,label=name);lines+=1
    if lines>12:raise ValueError('图例超过 12 条，请减少分组或主题。')
    denominator=translate('分母：本组本年纳入片段数' if metric.startswith('segment_') else '分母：本组本年至少有一个纳入片段的不同文档数')
    note=denominator+'；'+translate('纳入离群类 -1' if include_outliers else '排除离群类 -1')
    ax.set_xlabel(translate('年份')+'\n'+textwrap.fill(note,width=max(20,int(width*5))));ax.set_ylabel(translate(METRICS[metric]));ax.xaxis.set_major_locator(MaxNLocator(integer=True));ax.grid(axis='y',color='#e5e9e8',linewidth=.5)
    if cfg['show_title']:ax.set_title(translate('年度主题变化（历史数据）'),loc='left',fontsize=cfg['font_size']+1)
    style_axes(ax,cfg);ax.legend(fontsize=max(7,cfg['font_size']-2),loc='upper left',bbox_to_anchor=(1.01,1))
    from matplotlib.text import Text
    for t in fig.findobj(match=Text):t.set_fontproperties(FontProperties(fname=font_path,size=t.get_fontsize(),weight=t.get_fontweight()))
    fig.canvas.draw();box=fig.get_tightbbox(fig.canvas.get_renderer())
    if box.x0 < -.02 or box.y0 < -.02 or box.x1>width+.02 or box.y1>height+.02:raise ValueError('标签可能被裁切，请增大尺寸或减少选择。')
    out=io.BytesIO()
    with rc_context({'svg.fonttype':'path'}):fig.savefig(out,format=fmt,dpi=dpi)
    fig.clear();return out.getvalue(),missing_glyphs(font_path,selected['group'].tolist()+selected['topic_label'].tolist()+[METRICS[metric],'年度主题变化历史数据',note])
