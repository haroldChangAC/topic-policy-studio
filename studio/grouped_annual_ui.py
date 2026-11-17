"""Grouped annual panel; original annual and heatmap settings remain untouched."""
from studio.i18n import tr, localize_options
from studio.ui_layout import source_text
import pandas as pd
import streamlit as st
from studio.annual import RULE
from studio.topic_groups import annual,mapping
from studio.workflow import csv_bytes


def show(result,scope,settings):
    docs,panel,parts=annual(result,settings)
    st.caption(tr(RULE)+' '+tr('归组文档提及数从片段所属原始文档 ID 去重。'))
    if panel.empty:st.info(tr('没有可统计的年度数据。'));return
    displayed=panel.drop(columns=['group_key']).copy()
    displayed['segment_numerator']=displayed['segment_count'];displayed['document_numerator']=displayed['document_count']
    displayed['missing_status']=displayed['segment_denominator'].map(lambda n:'有纳入样本' if n else '无纳入样本或分母为零')
    displayed['segment_count']=displayed['segment_count'].where(displayed['segment_denominator'].gt(0))
    displayed['document_count']=displayed['document_count'].where(displayed['document_denominator'].gt(0))
    st.caption(tr('无纳入样本时计数显示为空；原始分子列仍保留 0，便于核对分母。'))
    st.dataframe(displayed,hide_index=True,width='stretch')
    exported=displayed.copy()
    for key in ('year_field','group_field','exclude_missing_year','exclude_missing_group','include_outliers'):exported[key]=settings[key]
    exported['view']='人工归组';exported['denominator_rule']=RULE
    st.download_button(tr('下载归组年度面板 CSV'),csv_bytes(exported),'grouped_annual.csv','text/csv',key=f'group_annual_csv_{scope}')
    st.download_button(tr('下载归组年度映射 CSV'),csv_bytes(mapping(result)),'grouped_annual_mapping.csv','text/csv',key=f'group_annual_map_{scope}')
    groups=docs.loc[docs.selected_by_metadata,['group_key','group','group_missing']].drop_duplicates()
    options=groups['group_key'].tolist();labels={r.group_key:r.group+(' [缺失字段]' if r.group_missing else '') for r in groups.itertuples()}
    with st.expander(tr('回查归组年度原文')):
        g=st.selectbox(tr('归组年度分组'),options,format_func=lambda x:labels[x],key=f'group_annual_group_{scope}')
        yrs=panel.loc[panel.group_key.eq(g),'year'].drop_duplicates().tolist();yrs=sorted(int(x) for x in yrs if pd.notna(x))+([None] if any(pd.isna(x) for x in yrs) else [])
        year=st.selectbox(tr('归组年度年份'),yrs,format_func=localize_options(lambda x:tr('缺失年份') if x is None else str(x)),key=f'group_annual_year_{scope}')
        names=panel[['view_id','view_name']].drop_duplicates('view_id').set_index('view_id')['view_name'].to_dict()
        vid=st.selectbox(tr('归组年度主题'),list(names),format_func=lambda x:names[x]+' · '+x,key=f'group_annual_topic_{scope}')
        mapdata=mapping(result);topics=mapdata.loc[mapdata.view_id.eq(vid),'original_topic_id'].tolist();f=result['metadata']['generated_fields']
        mask=docs.selected_by_metadata&docs.group_key.eq(g)&(docs.year.isna() if year is None else docs.year.eq(year))
        chosen=parts[parts[f['document_id']].isin(docs.loc[mask,'document_id']) & parts[f['topic_id']].isin(topics)]
        st.caption(tr('原始主题 {0}；{1} 篇文档、{2} 个片段。', f'{topics}', f"{chosen[f['document_id']].nunique()}", f'{len(chosen)}'))
        st.dataframe(chosen,hide_index=True,width='stretch')
        if not chosen.empty:
            sid=st.selectbox(tr('选择归组年度片段'),chosen[f['segment_id']].tolist(),key=f'group_annual_segment_{scope}')
            part=chosen.set_index(f['segment_id']).loc[sid];doc=result['rows'].set_index(f['document_id']).loc[part[f['document_id']]]
            st.caption(tr('文档 {0}；原始主题 {1}；位置 [{2},{3})', f"{part[f['document_id']]}", f"{part[f['topic_id']]}", f"{int(part[f['start_char']])}", f"{int(part[f['end_char']])}"))
            source_text(part[f['segment_text']]);source_text(doc[f['original_text']])
    from studio.grouped_charts_ui import show as show_charts
    show_charts(result,scope,settings)
