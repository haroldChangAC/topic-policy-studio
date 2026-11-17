"""Researcher-controlled adjacent-year similarity graph and evidence."""
from studio.i18n import diagnostic, tr, localize_options
from studio.ui_layout import anchor, source_text
import hashlib
import pandas as pd
import streamlit as st
from studio.cross_year import (ADJACENCY, MAX_NODES, WIDTH, calculate, defaults, evidence,
                               graph, inventory, source_digest)
from studio.workflow import csv_bytes
from studio.cross_year_static import default_export, validate_export
from studio.visualization import find_chinese_font


def show(result, scope):
    anchor('cross-year')
    if not result.get('yearly_analysis'):
        st.subheader(tr('跨年主题关联'))
        st.info(tr('先完成按年份分别建模，再比较相邻年份的主题。'))
        return
    st.subheader(tr('相邻年份主题相似关联桑基图'))
    st.caption(tr('模型空间的余弦相似度仅用于人工核查；不自动表示主题延续、因果关系或文档实际流动。只比较相邻自然年份，缺失、失败或未选择的年份不会被跨过。-1 离群主题默认排除且本图不可纳入。'))
    viz=result.setdefault('visualization',{});digest=source_digest(result);saved=viz.get('cross_year_links')
    if saved and saved.get('source_sha256')!=digest:
        viz.pop('cross_year_links',None);saved=None
        st.warning(tr('按年模型或主题向量已变化，旧关联设置已失效；请重新选择。'))
    try:
        nodes=inventory(result);initial=saved or defaults(result)
    except ValueError as exc:
        st.error(tr('无法读取年度主题向量：{0}', diagnostic(exc)));return
    years=sorted({n['year'] for n in nodes});k='cross_'+scope+'_'+digest[:12]
    if len(years)<2:
        st.info(tr('至少需要两个成功建模的自然相邻年份；当前没有足够结果。'));return
    selected_years=st.multiselect(tr('关联图年份'),years,default=[y for y in initial['years'] if y in years],key=k+'_years')
    choices=[(n['year'],n['topic_id']) for n in nodes if n['year'] in selected_years and n['topic_id']!=-1]
    byid={(n['year'],n['topic_id']):n for n in nodes}
    defaults2=[tuple(x) for x in initial['topics'] if tuple(x) in choices]
    skey=k+'_topics_'+','.join(map(str,selected_years))
    selected=st.multiselect(tr('关联图主题（最多 60 个；按年份和编号区分）'),choices,default=defaults2,
        format_func=localize_options(lambda x:tr('{0} · 主题 {1} · {2}（{3} 片段）', f'{x[0]}', f'{x[1]}', f"{byid[x]['label']}", f"{byid[x]['segment_count']}")),key=skey)
    threshold=st.slider(tr('最低余弦相似度（研究者选择；不是已验证的最佳门槛）'),0.0,1.0,float(initial['threshold']),0.01,key=k+'_threshold')
    max_edges=st.slider(tr('图中最多关联数（按相似度从高到低）'),1,150,int(initial['max_edges']),key=k+'_max')
    style=st.selectbox(tr('关联图样式'),['color','mono'],index=0 if initial['style']=='color' else 1,format_func=localize_options(lambda x:tr('论文彩色') if x=='color' else tr('论文黑白')),key=k+'_style')
    if len(selected)>MAX_NODES:
        viz.pop('cross_year_links',None)
        st.warning(tr('所选主题超过 {0} 个；请减少选择，旧结果不再显示。', f'{MAX_NODES}'));return
    settings={'schema_version':1,'source_sha256':digest,'years':sorted(selected_years),
              'topics':[list(x) for x in selected], 'threshold':threshold,'max_edges':max_edges,'style':style}
    viz['cross_year_links']=settings
    chosen,frame,excluded,warnings=calculate(result,settings)
    st.caption(tr('{0}。{1}。Plotly 节点高度由连线权重总和决定，不等于主题片段数；实际片段和文档数量见表。', tr(ADJACENCY), tr(WIDTH)))
    if excluded:
        st.warning(tr('以下主题无法比较：{0}', ' · '.join(
            tr('{0} 年主题 {1}：{2}', x['year'], x['topic_id'], diagnostic(x['reason'])) for x in excluded)))
    if warnings:
        st.warning(' · '.join(diagnostic(message) for message in warnings[:20])
                   + (tr(' · 另有 {0} 项', len(warnings)-20) if len(warnings)>20 else ''))
    node_table=pd.DataFrame([{k:v for k,v in n.items() if k not in ('vector','unit')} for n in chosen])
    if not node_table.empty:st.dataframe(node_table,hide_index=True,width='stretch')
    if frame.empty:
        st.info(tr('当前没有可比较的相邻年份主题对；请检查年份、主题及向量。'));return
    drawn=frame[frame.in_graph];above=int((frame.passes_threshold & frame.similarity.gt(0)).sum())
    st.info(tr('已计算 {0} 对；达到门槛且相似度为正的候选关联 {1} 条；按上限实际绘图 {2} 条。', f'{len(frame)}', f'{above}', f'{len(drawn)}'))
    pair_summary=[]
    for year in sorted(selected_years):
        if year+1 not in selected_years:continue
        subset=frame[frame.source_year.eq(year) & frame.target_year.eq(year+1)]
        pair_summary.append({tr('年份对'):f'{year} → {year+1}',
            tr('所选起点主题数'):sum(n['year']==year for n in chosen),
            tr('所选终点主题数'):sum(n['year']==year+1 for n in chosen),
            tr('可比较主题对'):len(subset),tr('达到门槛且为正'):int((subset.passes_threshold & subset.similarity.gt(0)).sum()),
            tr('实际入图'):int(subset.in_graph.sum()),tr('门槛'):threshold,tr('全图连线上限'):max_edges})
    if pair_summary:st.dataframe(pd.DataFrame(pair_summary),hide_index=True,width='stretch')
    linked=set(zip(drawn.source_year,drawn.source_topic_id))|set(zip(drawn.target_year,drawn.target_topic_id))
    isolated=[f"{n['year']}/{n['topic_id']}" for n in chosen if (n['year'],n['topic_id']) not in linked]
    if isolated:st.caption(tr('孤立主题仅列在表格中：{0}', ', '.join(isolated)))
    st.dataframe(frame,hide_index=True,width='stretch')
    all_csv=csv_bytes(frame);edge_csv=csv_bytes(drawn)
    all_name='cross_year_similarity-'+hashlib.sha256(all_csv).hexdigest()[:24]+'.csv'
    edge_name='cross_year_edges-'+hashlib.sha256(edge_csv).hexdigest()[:24]+'.csv'
    st.download_button(tr('下载跨年相似度 CSV'),all_csv,all_name,'text/csv',key=k+'_csv')
    st.download_button(tr('下载入图连线 CSV'),edge_csv,edge_name,'text/csv',key=k+'_edges')
    fig=graph(chosen,frame,settings,translate=tr)
    if fig is None:st.info(tr('没有满足门槛且为正的关联；请调整主题或门槛。'))
    else:
        st.plotly_chart(fig,width='stretch',key=k+'_plot',config={'displayModeBar':False})
        st.caption(tr('若线条密集或标签重叠，请减少主题或连线。静态图与交互图使用相同数据，但排布由不同绘图组件计算；SVG 的中文为轮廓，不可直接编辑文字。'))
        export=viz.get('cross_year_export') or default_export()
        with st.expander(tr('静态图尺寸与字号')):
            a,b=st.columns(2)
            width=a.number_input(tr('桑基图宽度（英寸）'),min_value=6.0,max_value=16.0,value=float(export['width']),step=1.0,key=k+'_width')
            height=b.number_input(tr('桑基图高度（英寸）'),min_value=4.0,max_value=14.0,value=float(export['height']),step=1.0,key=k+'_height')
            a,b=st.columns(2)
            dpi=a.selectbox(tr('桑基图 PNG DPI'),[150,300,600],index=[150,300,600].index(export['dpi']),key=k+'_dpi')
            font_size=b.number_input(tr('桑基图字号'),min_value=7,max_value=16,value=int(export['font_size']),key=k+'_font')
            viz['cross_year_export']={'width':float(width),'height':float(height),'dpi':int(dpi),'font_size':int(font_size)}
            try:validate_export(viz['cross_year_export'])
            except ValueError as exc:st.warning(diagnostic(exc))
            font,family=find_chinese_font()
            if font:st.caption(tr('静态图字体：{0}；大图最大 3600 万像素。准备后在下方选择 PNG 或 SVG 下载。', f'{family}'))
            else:st.warning(tr('未找到支持中文的字体，无法准备静态图。'))
            if len(drawn)>80 or len(linked)>30:st.warning(tr('静态图超过 80 条连线或 30 个节点，请提高门槛或减少主题。'))
    st.caption(tr('可选择表格中的任一主题对回查；即使未入图也可检查。'))
    indices=list(frame.index)
    idx=st.selectbox(tr('选择关联回查两端原文'),indices,format_func=localize_options(lambda i:tr('{0} 主题 {1} → {2} 主题 {3}：{4}', f"{int(frame.at[i, 'source_year'])}", f"{int(frame.at[i, 'source_topic_id'])}", f"{int(frame.at[i, 'target_year'])}", f"{int(frame.at[i, 'target_topic_id'])}", f"{frame.at[i, 'similarity']:.4f}")),key=k+'_edge')
    row=frame.loc[idx]
    for side in ('source','target'):
        year,topic=int(row[f'{side}_year']),int(row[f'{side}_topic_id'])
        with st.expander(tr('{0} 年主题 {1}：关键词 {2}', f'{year}', f'{topic}', f"{row[f'{side}_keywords']}")):
            excerpts=evidence(result,year,topic);st.dataframe(excerpts,hide_index=True,width='stretch')
            f=result['metadata']['generated_fields']
            if not excerpts.empty:
                sid=st.selectbox(tr('选择片段查看完整原文'),excerpts[f['segment_id']].tolist(),key=k+'_'+side+'_segment_'+str(year)+'_'+str(topic))
                seg=result['segments'].set_index(f['segment_id']).loc[sid];doc=result['rows'].set_index(f['document_id']).loc[seg[f['document_id']]]
                st.caption(tr('文档 ID：{0}；片段 ID：{1}；位置 [{2}, {3})', f"{seg[f['document_id']]}", f'{sid}', f"{int(seg[f['start_char']])}", f"{int(seg[f['end_char']])}"))
                source_text(seg[f['segment_text']]);source_text(doc[f['original_text']])
    st.caption(tr('项目 ZIP 保存当前筛选与计算口径；更改年度建模结果后会清除旧关联设置并重新计算。'))
