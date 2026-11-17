"""Interactive view of the existing annual panel."""
from studio.i18n import diagnostic, tr, localize_options
from studio.ui_layout import source_text
import pandas as pd
import streamlit as st
from studio.annual import METRICS, RULE
from studio.annual_heatmap import ALL, MAX_TOPICS, MAX_YEARS, data, evidence, render, samples, validate_heatmap
from studio.visualization import find_chinese_font
from studio.workflow import csv_bytes


def show(result,scope,settings,docs,panel,parts,visual):
    st.markdown(tr('#### 年度主题热力图'))
    st.caption(tr('使用上方已计算的年度面板，不重新训练。全部语料将当前纳入的各组分子与分母分别相加后计算比例，不对组间比例取平均。'))
    eligible=docs[docs.selected_by_metadata];group_table=eligible[['group_key','group','group_missing']].drop_duplicates()
    groups={r.group_key:r.group+(' [缺失字段]' if r.group_missing else '') for r in group_table.itertuples()}
    goptions=[ALL]+(list(groups) if settings['group_field'] is not None else [])
    groups[ALL]='全部语料（遵循当前缺失处理）'
    years=sorted(int(x) for x in panel.year.dropna().unique());topics=sorted(int(x) for x in panel.topic_id.unique())
    if not years or not topics:st.info(tr('没有可绘制的有效年份或主题；缺失年份不会归入任何年份。'));return
    old=visual.get('annual_heatmap') or {'schema_version':1,'group_key':ALL,'years':years[:10],
                                        'topics':topics[:8],'metric':'segment_share'}
    group=st.selectbox(tr('热力图分组'),goptions,index=goptions.index(old['group_key']) if old['group_key'] in goptions else 0,
                       format_func=lambda x:groups[x],key=f'heat_group_{scope}')
    metric=st.selectbox(tr('热力图指标'),list(METRICS),index=list(METRICS).index(old['metric']),format_func=localize_options(lambda x:tr(METRICS[x])),key=f'heat_metric_{scope}')
    ys=st.multiselect(tr('热力图年份'),years,default=[x for x in old['years'] if x in years],key=f'heat_years_{scope}')
    ts=st.multiselect(tr('热力图主题'),topics,default=[x for x in old['topics'] if x in topics],
                      format_func=localize_options(lambda x:f'{x}'+(tr('（离群）') if x==-1 else '')),key=f'heat_topics_{scope}')
    if len(ys)>MAX_YEARS or len(ts)>MAX_TOPICS:st.warning(tr('最多选择 {0} 个年份与 {1} 个主题。', f'{MAX_YEARS}', f'{MAX_TOPICS}'));return
    obj={'schema_version':1,'group_key':group,'years':ys,'topics':ts,'metric':metric}
    validate_heatmap(result,visual,obj);visual['annual_heatmap']=obj
    st.caption(tr('指标：{0}；分组：{1}；年份缺失{2}；分组缺失{3}；离群类{4}。', f'{METRICS[metric]}', f'{groups[group]}', f"{('排除' if settings['exclude_missing_year'] else '保留在面板但不画入年份')}", f"{('排除' if settings['exclude_missing_group'] and settings['group_field'] else '保留' if settings['group_field'] else '不适用')}", f"{('纳入' if settings['include_outliers'] else '排除')}"))
    st.caption(tr(RULE)+' '+tr('离群片段数和比例以排除离群前全部已建模片段为分母；缺失年份不进入任何年份。'))
    if not ys or not ts:st.info(tr('请选择年份和主题。'));return
    frame=data(panel,metric,group,ys,ts);sample=samples(result,docs,group,ys)
    if group==ALL and settings['group_field'] is not None:
        breakdown=[samples(result,docs,key,ys) for key in group_table['group_key']]
        if breakdown:sample=pd.concat([sample,*breakdown],ignore_index=True)
    st.write(tr('各分组及年份样本量与离群情况'))
    st.dataframe(sample.drop(columns=['group_key']),hide_index=True,width='stretch',column_config={
        'group':'分组','group_missing':'分组缺失','sample_scope':'统计范围','year':'年份',
        'raw_valid_documents':'原始有效文档数','document_denominator':'纳入文档分母数',
        'included_segments':'纳入片段数','outlier_segments':'离群片段数',
        'modeled_segments_before_outlier_filter':'排除离群前建模片段数',
        'outlier_share_of_all_modeled_segments':'排除离群前的离群比例'})
    st.caption(tr('斜线与“—”为无纳入样本／分母为零；有纳入样本但该主题为零则显示 0。图片中的比例数值为显示用四舍五入，CSV 保留原值。'))
    st.dataframe(frame.drop(columns=['group_key']),hide_index=True,width='stretch',column_config={
        'group':'分组','group_missing':'分组缺失','year':'年份','topic_id':'主题编号','topic_label':'当前主题名称',
        'metric':'指标代码','metric_label':'指标','value':'实际值','numerator':'分子','denominator':'分母',
        'plot_value':'图中值','missing_status':'样本状态','segment_count':'片段数',
        'segment_denominator':'片段分母','segment_share':'片段占比','document_count':'文档提及数',
        'document_denominator':'文档分母','document_mention_rate':'文档提及率'})
    chart_settings=visual['settings'];font,_=find_chinese_font()
    try:
        png,missing=render(frame,chart_settings['width_inches'],chart_settings['height_inches'],chart_settings['dpi'],font,chart_settings.get('appearance'),'png', translate=tr)
        svg,_=render(frame,chart_settings['width_inches'],chart_settings['height_inches'],chart_settings['dpi'],font,chart_settings.get('appearance'),'svg', translate=tr)
        if missing:st.warning(tr('字体缺少字符：')+''.join(missing))
        st.image(png)
        st.download_button(tr('下载年度热力图 PNG'),png,'annual_heatmap.png','image/png',key=f'heat_png_{scope}')
        st.download_button(tr('下载年度热力图 SVG'),svg,'annual_heatmap.svg','image/svg+xml',key=f'heat_svg_{scope}')
    except ValueError as exc:st.warning(tr('热力图无法导出：{0}', diagnostic(exc)))
    exported=frame.drop(columns=['group_key']).copy()
    for k in ('year_field','group_field','exclude_missing_year','exclude_missing_group','include_outliers'):exported[k]=settings[k]
    exported['denominator_rule']=RULE
    st.download_button(tr('下载年度热力图数据 CSV'),csv_bytes(exported),'annual_heatmap.csv','text/csv',key=f'heat_csv_{scope}')
    st.download_button(tr('下载年度热力图样本量 CSV'),csv_bytes(sample),'annual_heatmap_samples.csv','text/csv',key=f'heat_samples_{scope}')
    with st.expander(tr('回查热力图单元格原文')):
        year=st.selectbox(tr('热力图核查年份'),ys,key=f'heat_check_year_{scope}')
        topic=st.selectbox(tr('热力图核查主题'),ts,format_func=localize_options(lambda x:str(x)+(tr('（离群）') if x==-1 else '')),key=f'heat_check_topic_{scope}')
        matched,segments=evidence(result,docs,parts,group,year,topic);f=result['metadata']['generated_fields']
        st.caption(tr('匹配 {0} 篇文档和 {1} 个片段。', f'{len(matched)}', f'{len(segments)}'))
        st.dataframe(matched,hide_index=True,width='stretch');st.dataframe(segments,hide_index=True,width='stretch')
        if not segments.empty:
            sid=st.selectbox(tr('选择热力图片段回查全文'),segments[f['segment_id']].tolist(),key=f'heat_check_segment_{scope}')
            segment=segments.set_index(f['segment_id']).loc[sid];document=result['rows'].set_index(f['document_id']).loc[segment[f['document_id']]]
            start,end=int(segment[f['start_char']]),int(segment[f['end_char']])
            st.caption(tr('文档 {0}；片段 {1}；主题 {2}；原文位置 [{3},{4})', f"{segment[f['document_id']]}", f'{sid}', f'{topic}', f'{start}', f'{end}'))
            source_text(segment[f['segment_text']]);source_text(document[f['original_text']])
