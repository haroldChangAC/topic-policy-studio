"""Publication charts for the current manual grouping, with separate saved settings."""
from studio.i18n import diagnostic, tr, localize_options
from studio.ui_layout import source_text
import copy
import json
import pandas as pd
import streamlit as st
from studio.annual import METRICS,RULE
from studio.annual_heatmap import ALL,MAX_YEARS,MAX_TOPICS,render as render_heat
from studio.grouped_charts import basis,cells,display_labels,evidence,for_scope,render_line,signature,source,validate
from studio.visualization import find_chinese_font
from studio.workflow import csv_bytes


def show(result,scope,settings):
    visual=result['visualization'];docs,panel,parts,members=source(result,settings)
    years=sorted(int(y) for y in panel.year.dropna().unique());cats=panel[['view_id','view_name']].drop_duplicates('view_id')
    names=cats.set_index('view_id')['view_name'].to_dict();labels=display_labels(names, translate=tr)
    if not years or not names:st.info(tr('没有可绘制的有效年份或主题；缺失年份保留在表格中。'));return
    groups=docs.loc[docs.selected_by_metadata,['group_key','group','group_missing']].drop_duplicates()
    group_names={r.group_key:tr('缺失分组') if r.group_missing else r.group for r in groups.itertuples()}
    group_names[ALL]=tr('全部语料（遵循缺失处理）');options=[ALL]+(groups.group_key.tolist() if settings['group_field'] is not None else [])
    b=basis(result,settings);sig=signature(b);old=visual.get('grouped_annual_charts')
    if old and old['signature']!=sig:
        st.warning(tr('主题组名称、成员或年度范围已变化；图表按当前方案重新计算，旧图表筛选已重置。'))
        old=None
    def defaults(heat=False):
        return {'scopes':[ALL],'years':years[:10],'topics':list(names)[:8 if heat else 3],'metric':'segment_share'}
    conf=copy.deepcopy(old) if old else {'schema_version':1,'basis':b,'signature':sig,'line':defaults(),'heat':defaults(True)}
    key=f'{scope}_{sig[:12]}'
    st.caption(tr('人工主题归组视图：未归组主题继续单独显示；离群类由上方开关决定，不能加入主题组。组名和成员改变后即时重算；原始主题和关键词不变。'))
    st.caption(tr('另存项目使用精简存储格式版本 5；旧版工作台无法打开版本 5，请保留原项目。'))
    st.caption(tr('全部语料从当前纳入片段及文档 ID 重新去重，不平均各组比例。缺失年份不绘入图；缺少年份或未选年份的间隔不连线。'))
    st.caption(tr('折线暂不支持直接点击回查；可在下方热力图回查区选择相同分组、年份和主题。'))
    font,_=find_chinese_font();appearance=visual['settings'];opts=list(METRICS)
    st.markdown(tr('##### 人工主题归组年度折线图'))
    saved=conf['line']
    lm=st.selectbox(tr('归组折线指标'),opts,index=opts.index(saved['metric']),format_func=localize_options(lambda x:tr(METRICS[x])),key=f'group_line_metric_{key}')
    lg=st.multiselect(tr('归组折线分组'),options,default=[x for x in saved['scopes'] if x in options],format_func=lambda x:group_names[x],key=f'group_line_scopes_{key}')
    ly=st.multiselect(tr('归组折线年份'),years,default=[x for x in saved['years'] if x in years],key=f'group_line_years_{key}')
    lt=st.multiselect(tr('归组折线主题'),list(names),default=[x for x in saved['topics'] if x in names],format_func=lambda x:labels[x],key=f'group_line_topics_{key}')
    conf['line']={'scopes':lg,'years':ly,'topics':lt,'metric':lm}
    if lg and ly and lt:
        if len(lg)*len(lt)>12:st.warning(tr('折线图最多 12 个分组与主题组合，请减少选择。'))
        else:
            lf=pd.concat([cells(for_scope(result,docs,panel,parts,members,g),lm,ly,lt,settings,members) for g in lg],ignore_index=True)
            lf['scheme_signature']=sig
            st.dataframe(lf.drop(columns=['group_key']),hide_index=True,width='stretch')
            try:
                png,missing=render_line(lf,appearance['width_inches'],appearance['height_inches'],appearance['dpi'],font,appearance.get('appearance'),'png', translate=tr)
                svg,_=render_line(lf,appearance['width_inches'],appearance['height_inches'],appearance['dpi'],font,appearance.get('appearance'),'svg', translate=tr)
                if missing:st.warning(tr('字体缺少字符：')+''.join(missing))
                st.image(png)
                st.download_button(tr('下载归组折线图 PNG'),png,'grouped_annual_line.png','image/png',key=f'group_line_png_{key}')
                st.download_button(tr('下载归组折线图 SVG'),svg,'grouped_annual_line.svg','image/svg+xml',key=f'group_line_svg_{key}')
            except ValueError as exc:st.warning(tr('归组折线图无法绘制：{0}', diagnostic(exc)))
            st.download_button(tr('下载归组折线图数据 CSV'),csv_bytes(lf.drop(columns=['group_key'])),'grouped_annual_line.csv','text/csv',key=f'group_line_csv_{key}')
    else:st.info(tr('请选择折线图的分组、年份及主题。'))
    st.markdown(tr('##### 人工主题归组年度热力图'))
    saved=conf['heat'];index=options.index(saved['scopes'][0]) if saved['scopes'] and saved['scopes'][0] in options else 0
    hg=st.selectbox(tr('归组热力图分组'),options,index=index,format_func=lambda x:group_names[x],key=f'group_heat_scope_{key}')
    hm=st.selectbox(tr('归组热力图指标'),opts,index=opts.index(saved['metric']),format_func=localize_options(lambda x:tr(METRICS[x])),key=f'group_heat_metric_{key}')
    hy=st.multiselect(tr('归组热力图年份'),years,default=[x for x in saved['years'] if x in years],key=f'group_heat_years_{key}')
    ht=st.multiselect(tr('归组热力图主题'),list(names),default=[x for x in saved['topics'] if x in names],format_func=lambda x:labels[x],key=f'group_heat_topics_{key}')
    conf['heat']={'scopes':[hg],'years':hy,'topics':ht,'metric':hm}
    if len(hy)>MAX_YEARS or len(ht)>MAX_TOPICS:st.warning(tr('热力图最多选择 {0} 个年份和 {1} 个主题。', f'{MAX_YEARS}', f'{MAX_TOPICS}'))
    elif hy and ht:
        hf=cells(for_scope(result,docs,panel,parts,members,hg),hm,hy,ht,settings,members)
        hf['scheme_signature']=sig
        st.caption(tr('灰色斜线与“—”表示无纳入样本或分母为零；有样本但未出现时为 0。图中比例四舍五入，CSV 保留实际值。'))
        st.dataframe(hf.drop(columns=['group_key']),hide_index=True,width='stretch')
        try:
            heat=hf.assign(topic_id=hf.topic_group_id,topic_label=hf.topic_group_name)
            png,missing=render_heat(heat,appearance['width_inches'],appearance['height_inches'],appearance['dpi'],font,appearance.get('appearance'),'png',grouped=True, translate=tr)
            svg,_=render_heat(heat,appearance['width_inches'],appearance['height_inches'],appearance['dpi'],font,appearance.get('appearance'),'svg',grouped=True, translate=tr)
            if missing:st.warning(tr('字体缺少字符：')+''.join(missing))
            st.image(png)
            st.download_button(tr('下载归组热力图 PNG'),png,'grouped_annual_heatmap.png','image/png',key=f'group_heat_png_{key}')
            st.download_button(tr('下载归组热力图 SVG'),svg,'grouped_annual_heatmap.svg','image/svg+xml',key=f'group_heat_svg_{key}')
        except ValueError as exc:st.warning(tr('归组热力图无法绘制：{0}', diagnostic(exc)))
        st.download_button(tr('下载归组热力图数据 CSV'),csv_bytes(hf.drop(columns=['group_key'])),'grouped_annual_heatmap.csv','text/csv',key=f'group_heat_csv_{key}')
        with st.expander(tr('回查归组图表单元格原文')):
            year=st.selectbox(tr('归组图表核查年份'),hy,key=f'group_chart_year_{key}')
            topic=st.selectbox(tr('归组图表核查主题'),ht,format_func=lambda x:labels[x],key=f'group_chart_topic_{key}')
            matched,segments=evidence(result,docs,parts,members,hg,year,topic);f=result['metadata']['generated_fields']
            st.caption(tr('原始主题 {0}；{1} 篇文档、{2} 个片段。', f'{members[topic]}', f'{len(matched)}', f'{len(segments)}'))
            st.dataframe(matched,hide_index=True,width='stretch');st.dataframe(segments,hide_index=True,width='stretch')
            if not segments.empty:
                sid=st.selectbox(tr('选择归组图表片段'),segments[f['segment_id']].tolist(),key=f'group_chart_segment_{key}')
                seg=segments.set_index(f['segment_id']).loc[sid];doc=result['rows'].set_index(f['document_id']).loc[seg[f['document_id']]]
                st.caption(tr('文档 {0}；原始主题 {1}；位置 [{2},{3})', f"{seg[f['document_id']]}", f"{seg[f['topic_id']]}", f"{int(seg[f['start_char']])}", f"{int(seg[f['end_char']])}"))
                source_text(seg[f['segment_text']]);source_text(doc[f['original_text']])
    else:st.info(tr('请选择热力图的年份和主题。'))
    conf['basis']=b;conf['signature']=sig;validate(result,visual,conf);visual['grouped_annual_charts']=conf
    st.download_button(tr('下载归组图表设置 JSON'),json.dumps({'settings':conf,'rule':RULE},ensure_ascii=False,indent=2).encode(),'grouped_annual_settings.json','application/json',key=f'group_chart_settings_{key}')
