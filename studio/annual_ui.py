"""Year/group/topic panel for both live and restored analysis results."""
from studio.i18n import diagnostic, tr, localize_options
from studio.ui_layout import anchor, source_text
import json

import pandas as pd
import streamlit as st

from studio.annual import METRICS, RULE, calculate, prepare_documents, render, validate_settings
from studio.visualization import default_visualization, find_chinese_font
from studio.workflow import csv_bytes


def show(result, scope):
    anchor('annual')
    st.subheader(tr('年份 × 分组 × 主题'))
    st.caption(tr('依据已保存的片段主题分配统计历史数据。')+' '+tr(RULE))
    meta=result['metadata'];columns=meta.get('original_columns',[])
    if not columns:
        st.warning(tr('项目没有保存可选原始字段，无法选择年份或分组；不会猜测。'));return
    visual=result.setdefault('visualization',default_visualization());old=visual.get('annual_panel')
    year_default=old['year_field'] if old and old['year_field'] in columns else columns[0]
    year=st.selectbox(tr('选择年份或日期字段'),columns,index=columns.index(year_default),key=f'ann_year_{scope}')
    choices=[None,*columns];group_default=old['group_field'] if old and old['group_field'] in choices else None
    group_index=st.selectbox(tr('选择分组字段（可选）'),list(range(len(choices))),index=choices.index(group_default),
                             format_func=localize_options(lambda i:tr('全部语料（不分组）') if i==0 else choices[i]),key=f'ann_group_field_v2_{scope}')
    group=choices[group_index]
    a,b,c=st.columns(3)
    missing_year=a.checkbox(tr('排除年份缺失或无法解析'),value=old['exclude_missing_year'] if old else True,key=f'ann_myear_{scope}')
    missing_group=b.checkbox(tr('排除分组缺失'),value=old['exclude_missing_group'] if old else True,disabled=group is None,key=f'ann_mgroup_{scope}')
    outliers=c.checkbox(tr('统计纳入离群主题 -1'),value=old['include_outliers'] if old else True,key=f'ann_outliers_{scope}')
    config={'year_field':year,'group_field':group,'exclude_missing_year':missing_year,'exclude_missing_group':missing_group,
            'include_outliers':outliers,'metric':'segment_count','selected_groups':[],'selected_topics':[],'rule_version':1}
    try:
        preview=prepare_documents(result,config)
        st.caption(tr('预览 {0} 条原始文档：年份缺失或无法解析 {1}；分组缺失 {2}；有效原文但无建模片段 {3}。', f'{len(preview)}', f"{int(preview['year_missing'].sum())}", f"{int(preview['group_missing'].sum())}", f"{int((preview['valid_original_text'] & preview['modeled_segments'].eq(0)).sum())}"))
        summary=preview.groupby(['group','group_missing','year','year_missing','reason'],dropna=False).size().reset_index(name='document_count')
        with st.expander(tr('预览解析年份、分组与文档状态')):
            st.dataframe(summary,hide_index=True,width='stretch');st.dataframe(preview,hide_index=True,width='stretch')
        if st.button(tr('计算年度主题面板'),key=f'ann_calculate_{scope}'):
            docs,panel,_=calculate(result,config)
            config['selected_groups']=docs.loc[docs['selected_by_metadata'],'group_key'].drop_duplicates().tolist()[:2]
            topic_ids=sorted(int(x) for x in panel['topic_id'].unique()) if not panel.empty else []
            config['selected_topics']=([x for x in topic_ids if x!=-1]+([-1] if -1 in topic_ids else []))[:3]
            for widget_key in (f'ann_groups_{scope}',f'ann_topics_{scope}',f'ann_metric_{scope}',f'ann_check_group_{scope}',f'ann_check_year_{scope}',f'ann_check_topic_{scope}',f'ann_check_segment_{scope}'):
                st.session_state.pop(widget_key,None)
            visual.pop('annual_heatmap',None)
            for key in list(st.session_state):
                if key.startswith('heat_') and key.endswith(scope):st.session_state.pop(key,None)
            validate_settings(result,config);visual['annual_panel']=config;st.rerun()
    except (KeyError,ValueError,TypeError) as exc:
        st.warning(tr('无法统计：{0}', diagnostic(exc)));return
    settings=visual.get('annual_panel')
    if not settings:
        st.info(tr('选择字段后点击计算；不会修改原始数据或主题分配。'));return
    if any(config[k]!=settings[k] for k in ('year_field','group_field','exclude_missing_year','exclude_missing_group','include_outliers')):
        st.warning(tr('字段或处理方式已改变；以下仍是上次计算的设置，点击“计算年度主题面板”后更新。'))
    try: docs,panel,parts=calculate(result,settings)
    except (ValueError,KeyError,TypeError) as exc:st.warning(tr('已保存的年度设置无法计算：{0}', diagnostic(exc)));return
    st.write(tr('当前统计：年份字段 {0}；分组字段 {1}；缺失年份{2}；缺失分组{3}；离群类{4}。', f"{settings['year_field']}", settings['group_field'] or tr('全部语料'), tr('排除' if settings['exclude_missing_year'] else '保留'), tr('排除' if settings['exclude_missing_group'] else '保留'), tr('纳入' if settings['include_outliers'] else '排除')))
    st.caption(tr(RULE)+' '+tr('没有样本或分母为零时比例为空；年字段无法解析时不会猜测。'))
    if panel.empty:
        st.warning(tr('当前范围没有可统计的分组、年份或主题。'));st.dataframe(docs,hide_index=True,width='stretch');return
    view=st.radio(tr('年度统计主题视图'),['原始主题','人工归组'], format_func=localize_options(),index=1 if visual.get('annual_view')=='grouped' else 0,horizontal=True,key=f'group_annual_view_{scope}')
    visual['annual_view']='grouped' if view=='人工归组' else 'original'
    if view=='人工归组':
        from studio.grouped_annual_ui import show as show_grouped_annual
        show_grouped_annual(result,scope,settings)
        return
    st.dataframe(panel.drop(columns=['group_key']),hide_index=True,width='stretch')
    gdata=docs.loc[docs['selected_by_metadata'],['group_key','group','group_missing']].drop_duplicates()
    goptions=gdata['group_key'].tolist();gnames={r.group_key:(tr('缺失分组') if r.group_missing else tr('全部语料') if settings['group_field'] is None else r.group) for r in gdata.itertuples()}
    topics=sorted(int(x) for x in panel['topic_id'].unique())
    metric=st.selectbox(tr('年度变化指标'),list(METRICS),index=list(METRICS).index(settings['metric']),format_func=localize_options(lambda x:tr(METRICS[x])),key=f'ann_metric_{scope}')
    selected_groups=st.multiselect(tr('图表显示分组'),goptions,default=[x for x in settings['selected_groups'] if x in goptions],format_func=lambda x:gnames[x],key=f'ann_groups_{scope}')
    selected_topics=st.multiselect(tr('图表显示主题'),topics,default=[x for x in settings['selected_topics'] if x in topics],format_func=localize_options(lambda x:str(x)+(tr('（离群）') if x==-1 else '')),key=f'ann_topics_{scope}')
    settings.update(metric=metric,selected_groups=selected_groups,selected_topics=selected_topics)
    st.caption(tr('图表不绘制缺失年份；缺失年份仍可在面板与原文核对中查看。数量为整数，比例采用上方分母；线条只是历史观测之间的连接。'))
    font,_=find_chinese_font();cfg=visual['settings']
    if selected_groups and selected_topics:
        try:
            png,missing=render(panel,metric,selected_groups,selected_topics,cfg['width_inches'],cfg['height_inches'],cfg['dpi'],font,cfg.get('appearance'),'png',settings['include_outliers'], translate=tr)
            svg,_=render(panel,metric,selected_groups,selected_topics,cfg['width_inches'],cfg['height_inches'],cfg['dpi'],font,cfg.get('appearance'),'svg',settings['include_outliers'], translate=tr)
            if missing:st.warning(tr('字体缺少字符：')+''.join(missing))
            st.image(png)
            st.download_button(tr('下载年度图 PNG'),png,'annual_topics.png','image/png',key=f'ann_png_{scope}')
            st.download_button(tr('下载年度图 SVG'),svg,'annual_topics.svg','image/svg+xml',key=f'ann_svg_{scope}')
        except ValueError as exc:st.warning(tr('年度图无法生成：{0}', diagnostic(exc)))
    else:st.info(tr('请选择图表分组和主题。'))
    st.download_button(tr('下载年度主题面板 CSV'),csv_bytes(panel.drop(columns=['group_key'])),'annual_panel.csv','text/csv',key=f'ann_csv_{scope}')
    manifest={'settings':settings,'formula':RULE,'year_parsing':'仅明确四位年份及可验证的年月日；失败为缺失；不从正文或文件名推断。','empty_cells':'片段数和文档数为 0，分母为 0 时比例为空。','chart':'历史数据；不绘制缺失年份；不代表预测或主题流动。'}
    st.download_button(tr('下载年度统计设置 JSON'),json.dumps(manifest,ensure_ascii=False,indent=2).encode(),'annual_settings.json','application/json',key=f'ann_settings_{scope}')
    with st.expander(tr('按分组、年份和主题核对原文')):
        group_selected=st.selectbox(tr('核对分组'),goptions,format_func=lambda x:gnames[x],key=f'ann_check_group_{scope}')
        year_options=panel.loc[panel['group_key'].eq(group_selected),'year'].drop_duplicates().tolist()
        year_options=sorted([int(v) for v in year_options if pd.notna(v)])+([None] if any(pd.isna(v) for v in year_options) else [])
        year_selected=st.selectbox(tr('核对年份'),year_options,format_func=localize_options(lambda x:tr('缺失年份') if x is None else str(x)),key=f'ann_check_year_{scope}')
        topic_selected=st.selectbox(tr('核对主题'),topics,format_func=localize_options(lambda x:str(x)+(tr('（离群）') if x==-1 else '')),key=f'ann_check_topic_{scope}')
        mask=docs['group_key'].eq(group_selected)&(docs['year'].isna() if year_selected is None else docs['year'].eq(year_selected))&docs['selected_by_metadata']
        scope_docs=docs[mask];f=meta['generated_fields'];subset=parts[parts[f['document_id']].isin(scope_docs['document_id']) & parts[f['topic_id']].eq(topic_selected)]
        used=scope_docs[scope_docs['document_id'].isin(subset[f['document_id']])]
        st.caption(tr('当前主题涉及 {0} 篇文档、{1} 个片段；下表保留原始字段和位置。', f'{len(used)}', f'{len(subset)}'))
        st.dataframe(used,hide_index=True,width='stretch');st.dataframe(subset,hide_index=True,width='stretch')
        if not subset.empty:
            sid=st.selectbox(tr('选择片段查看全文'),subset[f['segment_id']].tolist(),key=f'ann_check_segment_{scope}')
            part=subset.set_index(f['segment_id']).loc[sid];doc=result['rows'].set_index(f['document_id']).loc[part[f['document_id']]]
            st.caption(tr('文档 {0}；片段 {1}；主题 {2}；位置 [{3},{4})', f"{part[f['document_id']]}", f'{sid}', f'{topic_selected}', f"{int(part[f['start_char']])}", f"{int(part[f['end_char']])}"))
            source_text(part[f['segment_text']]);source_text(doc[f['original_text']])

    from studio.annual_heatmap_ui import show as show_heatmap
    show_heatmap(result,scope,settings,docs,panel,parts,visual)
