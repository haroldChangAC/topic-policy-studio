"""Heatmaps derived from the existing annual panel, without retraining."""
import io
import math
import textwrap

import numpy as np
import pandas as pd
from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch, Rectangle
from matplotlib.text import Text

from studio.annual import METRICS, calculate
from studio.plot_style import resolve_appearance, topic_label, check_y_label_spacing
from studio.visualization import missing_glyphs

ALL='__annual_all_groups__'
MAX_YEARS=30
MAX_TOPICS=25


def data(panel, metric, group, years, topics):
    """Combine existing panel counts, preserving its filters and denominators."""
    if metric not in METRICS:raise ValueError('热力图指标无效。')
    if not years or not topics:return pd.DataFrame()
    if len(years)>MAX_YEARS or len(topics)>MAX_TOPICS:raise ValueError(f'最多选择 {MAX_YEARS} 个年份、{MAX_TOPICS} 个主题。')
    selected=panel[panel.year.isin(years)&panel.topic_id.isin(topics)].copy()
    if group!=ALL:selected=selected[selected.group_key.eq(group)]
    if selected.empty:raise ValueError('所选分组、年份或主题没有年度面板数据。')
    records=[]
    for (year,topic),frame in selected.groupby(['year','topic_id'],sort=True):
        # Groups are disjoint at document level. Sum counts and denominators, never ratios.
        sc=int(frame.segment_count.sum());sd=int(frame.segment_denominator.sum())
        dc=int(frame.document_count.sum());dd=int(frame.document_denominator.sum())
        numerator,denominator=(sc,sd) if metric.startswith('segment_') else (dc,dd)
        # A zero numerator is a real zero only when this cell has a sample.
        # Keep the raw numerator for audit, but use the same missing value in
        # the table, CSV and chart when the denominator is zero.
        value=(numerator/denominator if metric in ('segment_share','document_mention_rate') else numerator) if denominator else None
        records.append({'group':'全部语料' if group==ALL else str(frame.group.iloc[0]),
                        'group_key':group,'group_missing':False if group==ALL else bool(frame.group_missing.iloc[0]),
                        'year':int(year),'topic_id':int(topic),'topic_label':str(frame.topic_label.iloc[0]),
                        'metric':metric,'metric_label':METRICS[metric],
                        'value':value,'numerator':numerator,'denominator':denominator,
                        'plot_value':value,
                        'missing_status':'有纳入样本' if denominator else '无纳入样本或分母为零',
                        'segment_count':sc,'segment_denominator':sd,'segment_share':sc/sd if sd else None,
                        'document_count':dc,'document_denominator':dd,'document_mention_rate':dc/dd if dd else None})
    return pd.DataFrame(records)


def samples(result, docs, group, years):
    """All modeled outliers are counted before the annual outlier filter."""
    selected=docs[docs.selected_by_metadata & docs.year.isin(years)]
    if group!=ALL:selected=selected[selected.group_key.eq(group)]
    f=result['metadata']['generated_fields'];segs=result['segments'];records=[]
    group_name='全部语料' if group==ALL else str(docs.loc[docs.group_key.eq(group),'group'].iloc[0])
    missing_group=False if group==ALL else bool(docs.loc[docs.group_key.eq(group),'group_missing'].iloc[0])
    for year in years:
        frame=selected[selected.year.eq(year)];ids=set(frame.document_id)
        modeled=segs[segs[f['document_id']].isin(ids)]
        outliers=int(modeled[f['topic_id']].eq(-1).sum());total=len(modeled)
        records.append({'group':group_name,'group_missing':missing_group,'group_key':group,
                        'sample_scope':'全部纳入分组汇总' if group==ALL else '单个分组',
                        'year':int(year),'raw_valid_documents':int(frame.valid_original_text.sum()),
                        'document_denominator':int(frame.included_document.sum()),
                        'included_segments':int(frame.included_segments.sum()),
                        'outlier_segments':outliers,'modeled_segments_before_outlier_filter':total,
                        'outlier_share_of_all_modeled_segments':outliers/total if total else None})
    return pd.DataFrame(records)


def evidence(result,docs,parts,group,year,topic):
    f=result['metadata']['generated_fields'];subset=docs[docs.selected_by_metadata & docs.year.eq(year)]
    if group!=ALL:subset=subset[subset.group_key.eq(group)]
    seg=parts[parts[f['document_id']].isin(subset.document_id)&parts[f['topic_id']].eq(topic)].copy()
    return subset[subset.document_id.isin(seg[f['document_id']])],seg


def validate_heatmap(result,visual,obj):
    if obj is None:return
    keys={'schema_version','group_key','years','topics','metric'}
    if not isinstance(obj,dict) or set(obj)!=keys or obj.get('schema_version')!=1 or not visual.get('annual_panel'):
        raise ValueError('年度热力图设置无效或缺少年度面板。')
    if obj['metric'] not in METRICS or not isinstance(obj['group_key'],str) or not isinstance(obj['years'],list) or not isinstance(obj['topics'],list):raise ValueError('年度热力图选项无效。')
    if len(obj['years'])>MAX_YEARS or len(obj['topics'])>MAX_TOPICS or len(set(obj['years']))!=len(obj['years']) or len(set(obj['topics']))!=len(obj['topics']):raise ValueError('年度热力图筛选超限或重复。')
    if any(isinstance(x,bool) or not isinstance(x,int) for x in obj['years']+obj['topics']):raise ValueError('年度热力图年份或主题无效。')
    docs,panel,_=calculate(result,visual['annual_panel'])
    valid_groups=set(docs.loc[docs.selected_by_metadata,'group_key']);valid_years=set(panel.year.dropna().astype(int));valid_topics=set(panel.topic_id.astype(int))
    if obj['group_key']!=ALL and obj['group_key'] not in valid_groups:raise ValueError('年度热力图分组不存在。')
    if not set(obj['years']).issubset(valid_years) or not set(obj['topics']).issubset(valid_topics):raise ValueError('年度热力图年份或主题不存在。')


def render(frame,width,height,dpi,font_path,appearance=None,fmt='png',grouped=False, *, translate=str):
    if fmt not in ('png','svg') or not font_path:raise ValueError('不支持的格式或缺少中文字体。')
    if not(4<=width<=16 and 3<=height<=12 and dpi in (150,300,600)) or width*height*dpi*dpi>36_000_000:raise ValueError('图表尺寸或分辨率超限。')
    if frame.empty:raise ValueError('请选择年份与主题。')
    years=sorted(frame.year.unique().tolist());topics=sorted(frame.topic_id.unique().tolist())
    if len(years)>MAX_YEARS or len(topics)>MAX_TOPICS:raise ValueError('年份或主题过多，请减少选择。')
    pivot=frame.pivot(index='topic_id',columns='year',values='plot_value').reindex(index=topics,columns=years)
    arr=pivot.to_numpy(dtype=float);missing=np.isnan(arr);finite=arr[~missing]
    cfg=resolve_appearance(appearance);mono=cfg['style']=='mono'
    if width*72/len(years)<cfg['font_size']*3 or height*72/len(topics)<cfg['font_size']*1.4:
        raise ValueError('图表格子过密，请减少年份或主题、增大画布或减小字号。')
    cmap=LinearSegmentedColormap.from_list('annual_gray',['#f7f7f7','#303030'] if mono else ['#f6fafb','#6da1a5','#315e78'])
    cmap=cmap.with_extremes(bad='#dedede');metric=str(frame.metric.iloc[0]);vmax=1.0 if metric in ('segment_share','document_mention_rate') else max(1.0,float(finite.max()) if len(finite) else 0.0)
    norm=Normalize(vmin=0,vmax=vmax)
    fig=Figure(figsize=(width,height),layout='constrained',facecolor='white');FigureCanvasAgg(fig);ax=fig.subplots()
    image=ax.imshow(np.ma.masked_invalid(arr),cmap=cmap,norm=norm,aspect='auto',interpolation='nearest')
    labels=frame.drop_duplicates('topic_id').set_index('topic_id').topic_label.to_dict()
    if grouped:
        from studio.grouped_charts import display_labels
        shown=display_labels({str(t):str(labels[t]) for t in topics},translate=translate)
        names=[textwrap.fill(shown[str(t)],width=max(12,int(width*2)),break_long_words=True) for t in topics]
    else:names=[textwrap.fill(topic_label(t,labels[t]),width=max(12,int(width*2)),break_long_words=True) for t in topics]
    ax.set_xticks(range(len(years)),[str(y) for y in years],rotation=45 if len(years)>8 else 0,ha='right' if len(years)>8 else 'center');ax.set_yticks(range(len(topics)),names)
    ax.set_xlabel(translate('年份'));ax.set_ylabel(translate('人工主题归组（含未归组主题）' if grouped else '主题编号与当前名称'));ax.tick_params(labelsize=cfg['font_size']*.85,length=0)
    ax.set_xticks(np.arange(-.5,len(years),1),minor=True);ax.set_yticks(np.arange(-.5,len(topics),1),minor=True)
    ax.grid(which='minor',color='white',linewidth=.6);ax.tick_params(which='minor',bottom=False,left=False)
    for i in range(len(topics)):
        for j in range(len(years)):
            if missing[i,j]:
                ax.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor='none',edgecolor='#777777',hatch='///',linewidth=.3))
                ax.text(j,i,'—',ha='center',va='center',fontsize=max(6,cfg['font_size']*.75),color='#333333')
            else:
                val=arr[i,j];color='white' if val/vmax>.58 else '#222222'
                ax.text(j,i,(f'{val:.2f}' if metric in ('segment_share','document_mention_rate') else str(int(val))),ha='center',va='center',fontsize=max(6,cfg['font_size']*.75),color=color)
    if cfg['show_title']:ax.set_title(translate('人工主题归组年度热力图 · ' if grouped else '年度主题热力图 · ')+str(frame.group.iloc[0]),loc='left',fontsize=cfg['font_size']+1)
    bar=fig.colorbar(image,ax=ax,shrink=.85,label=translate(METRICS[metric]));bar.outline.set_linewidth(.5)
    fig.legend(handles=[Patch(facecolor='#dedede',edgecolor='#777777',hatch='///',label='无纳入样本／分母为零')],loc='outside lower center',fontsize=max(7,cfg['font_size']-2),frameon=False)
    for text in fig.findobj(match=Text):text.set_fontproperties(FontProperties(fname=font_path,size=text.get_fontsize(),weight=text.get_fontweight()))
    fig.canvas.draw();check_y_label_spacing(ax);box=fig.get_tightbbox(fig.canvas.get_renderer())
    if box.x0<-.02 or box.y0<-.02 or box.x1>width+.02 or box.y1>height+.02:raise ValueError('图形标签可能裁切，请增大尺寸或减少选择。')
    out=io.BytesIO()
    with rc_context({'svg.fonttype':'path'}):fig.savefig(out,format=fmt,dpi=dpi)
    fig.clear();return out.getvalue(),missing_glyphs(font_path,names+[METRICS[metric],str(frame.group.iloc[0]),'无纳入样本分母为零'])
