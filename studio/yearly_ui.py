"""Independent yearly analysis and trace-back, separate from the saved global fit."""
from studio.i18n import diagnostic, tr
from studio.ui_layout import anchor, source_text
import re

import pandas as pd
import streamlit as st
from studio.yearly import preview, run, tables


def _progress_label(text):
    match = re.fullmatch(r'正在独立分析 (\d+) 年：(\d+) 篇文档、(\d+) 个片段。', text)
    if match:
        return tr('正在独立分析 {0} 年：{1} 篇文档、{2} 个片段。', *match.groups())
    return tr(text)


def show(result, scope, encoder_resource):
    anchor('yearly')
    st.subheader(tr('按年份分别建模'))
    st.caption(tr('在已经检查并切分的片段上，每年独立运行 BERTopic。各年度主题编号只在当年有效；相同编号不代表同一主题。原有全语料模型和年度统计保持不变。'))
    st.caption(tr('不解析的年份排除；沿用原项目的空值、重复处理和切分结果。-1 是当年离群类，不自动跨年匹配。'))
    cols=result['metadata'].get('original_columns',[])
    if not cols:
        st.info(tr('项目没有保存可选择的原始年份字段，无法按年建模。'));return
    k='yearly_'+scope
    existing=result.get('yearly_analysis')
    year_field=st.selectbox(tr('按年建模的年份或日期字段'),cols,index=cols.index(existing['year_field']) if existing and existing['year_field'] in cols else 0,key=k+'_field')
    docs=preview(result,year_field)
    with st.expander(tr('预览年份解析、文档状态与片段数')):
        st.dataframe(docs,hide_index=True,width='stretch')
    missing=docs[docs.year.isna()]
    st.info(tr('年份缺失或无法明确解析的原始文档：{0} 篇，其中已有建模片段 {1} 个；这些文档不纳入任何年份。', f'{len(missing)}', f'{int(missing.segment_count.sum())}'))
    available=sorted({int(y) for y in docs.loc[docs.status.eq('modeled'),'year'].dropna()})
    if not available:
        st.warning(tr('该字段没有可解析年份及有效建模片段。'))
    else:
        initial=[x for x in (existing or {}).get('selected_years',available) if x in available]
        selected=st.multiselect(tr('选择分别建模的年份'),available,default=initial,key=k+'_years')
        st.caption(tr('本次聚类、关键词词表默认使用原项目记录的设置；历史记录没有保存的项目改用当前默认值，并在运行记录明确标注。随机种子 42；不同环境不保证完全一致。'))
        if not result['metadata'].get('clustering_settings_requested') or not result['metadata'].get('keyword_processing'):
            st.warning(tr('历史记录缺少聚类或词表配置；若运行，将明确使用当前默认值，不能视为历史配置。'))
        device=st.selectbox(tr('按年建模设备'),['cpu','mps'],key=k+'_device')
        if existing:st.warning(tr('重新运行会替换当前会话中的按年结果；如需保留旧结果，请先准备并下载项目 ZIP。'))
        if st.button(tr('开始按年份分别建模'),key=k+'_run',disabled=not selected):
            try:
                message=st.empty()
                with st.spinner(tr('依次使用本地模型分析，各年失败不会阻止其他年份……')):
                    encoder,lock=encoder_resource(device)
                    with lock:
                        candidate=run(result,year_field,selected,encoder,device,progress=lambda text: message.info(_progress_label(text)))
                    result.setdefault('visualization',{}).pop('cross_year_links',None)
                    result['yearly_analysis']=candidate
                message.empty();st.success(tr('按年建模已结束；各年结果如下，请保存新的项目 ZIP。'))
            except Exception as exc:
                st.error(tr('按年建模未完成：{0}；先前已保存的结果未覆盖。', diagnostic(exc)))
    obj=result.get('yearly_analysis')
    if not obj:return
    st.caption(tr('当前按年结果：字段 {0}；模型 {1}；随机种子 42；{2}。修改后请下载项目 ZIP 保存。', f"{obj['year_field']}", f"{obj['model_id']}", f"{obj['settings_note']}"))
    summaries=[]
    for item in obj['results']:
        n=item['segments'];out=item['outliers']
        summaries.append({tr('年份'):item['year'],tr('文档数'):item['documents'],tr('片段数'):n,tr('非离群主题数'):item['non_outlier_topics'],
                          tr('离群片段'):out,tr('离群比例'):out/n if out is not None and n else None,tr('状态'):item['status'],tr('原因'):item['error']})
    st.dataframe(pd.DataFrame(summaries),hide_index=True,width='stretch')
    complete=[x for x in obj['results'] if x['status']=='completed']
    if not complete:
        st.warning(tr('没有成功完成的年份；失败原因见上表，原项目数据未改变。'));return
    year=st.selectbox(tr('查看年度建模结果'),[x['year'] for x in complete],key=k+'_viewyear')
    item=next(x for x in complete if x['year']==year)
    absent=[x['topic_id'] for x in item['topic_vectors'] if x['values'] is None]
    zeros=[x['topic_id'] for x in item['topic_vectors'] if x['values'] is not None and not any(x['values'])]
    if absent or zeros:
        st.warning(tr('主题向量缺失：{0}；零向量：{1}。这些主题无法计算余弦相似度，不会补造向量。', f'{absent}', f'{zeros}'))
    st.dataframe(pd.DataFrame(item['topics']),hide_index=True,width='stretch')
    topic=st.selectbox(tr('查看该年度主题片段'),[t['topic_id'] for t in item['topics']],key=k+'_topic')
    _,assignments,_=tables(result)
    chosen=assignments[assignments['year'].eq(year)&assignments['topic_id'].eq(topic)]
    st.dataframe(chosen,hide_index=True,width='stretch')
    if not chosen.empty:
        sid=st.selectbox(tr('选择年度片段回查全文'),chosen['segment_id'].tolist(),key=k+'_segment')
        f=result['metadata']['generated_fields']
        seg=result['segments'].set_index(f['segment_id']).loc[sid]
        doc=result['rows'].set_index(f['document_id']).loc[seg[f['document_id']]]
        start,end=int(seg[f['start_char']]),int(seg[f['end_char']])
        with st.expander(tr('查看片段、位置与原始文档')):
            st.caption(tr('文档 ID：{0}；片段 ID：{1}；位置 [{2}, {3})', f"{seg[f['document_id']]}", f'{sid}', f'{start}', f'{end}'))
            source_text(seg[f['segment_text']]);source_text(doc[f['original_text']])
    with st.expander(tr('按年建模运行参数和处理记录')):
        st.json({key:value for key,value in obj.items() if key!='results'})
        st.json({key:value for key,value in item.items() if key not in ('assignments','topic_vectors','keyword_weights')})
    st.caption(tr('年度主题汇总、片段分配、模型参数及实际向量映射可在下方选择“年度建模数据 ZIP”按需准备。保存整个项目请选择“项目 ZIP”。'))
