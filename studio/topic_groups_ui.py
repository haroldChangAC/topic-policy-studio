"""Manual grouping controls and non-destructive exports."""
from studio.i18n import diagnostic, tr, localize_options
from studio.ui_layout import source_text
import json
import streamlit as st
from studio.topic_groups import create,delete,mapping,summary,update
from studio.workflow import csv_bytes


def show(result,scope):
    st.subheader(tr('研究者手动归组主题'))
    st.caption(tr('人工归组仅改变统计视图；不训练模型，不改变主题编号、片段分配或已有名称。-1 为离群类，不可归组。每组至少两个原始主题。同名组按 ID 区分。新保存项目使用版本 5，旧版工作台无法打开，请保留原项目。'))
    table=mapping(result);f=result['metadata']['generated_fields'];segments=result['segments']
    counts=segments[f['topic_id']].value_counts();table['segment_count']=table.original_topic_id.map(counts).fillna(0).astype(int)
    st.caption(tr('原始主题关键词，未重新计算；不合并或平均关键词权重。'))
    st.dataframe(table[['original_topic_id','original_topic_name','original_keywords','segment_count','group_name','is_outlier','keyword_note']],hide_index=True,width='stretch')
    topic=st.selectbox(tr('归组前查看原始主题'),table.original_topic_id.tolist(),format_func=localize_options(lambda x:str(x)+(tr('（离群）') if x==-1 else '')),key=f'group_inspect_topic_{scope}')
    subset=segments[segments[f['topic_id']].eq(topic)];st.dataframe(subset,hide_index=True,width='stretch')
    if not subset.empty:
        sid=st.selectbox(tr('归组前选择原文片段'),subset[f['segment_id']].tolist(),key=f'group_inspect_segment_{scope}')
        item=subset.set_index(f['segment_id']).loc[sid];doc=result['rows'].set_index(f['document_id']).loc[item[f['document_id']]]
        with st.expander(tr('查看该片段与完整原文')):
            st.caption(tr('文档 {0}；片段 {1}；主题 {2}；位置 [{3},{4})', f"{item[f['document_id']]}", f'{sid}', f'{topic}', f"{int(item[f['start_char']])}", f"{int(item[f['end_char']])}"))
            source_text(item[f['segment_text']]);source_text(doc[f['original_text']])
    groups=(result.get('visualization') or {}).get('topic_groups',{}).get('groups',[])
    available=table.loc[(~table.is_outlier)&table.group_id.eq(''),'original_topic_id'].tolist()
    names={int(r.original_topic_id):str(r.original_topic_name) for r in table.itertuples()}
    with st.form(f'group_create_form_{scope}'):
        name=st.text_input(tr('新上级主题组名称'),max_chars=100,key=f'group_create_name_{scope}')
        members=st.multiselect(tr('选择至少两个未归组主题'),available,format_func=lambda x:f'{x} · {names[x]}',key=f'group_create_members_{scope}')
        clicked=st.form_submit_button(tr('创建主题组'), key=f'group_create_{scope}')
    if clicked:
        try:create(result,name,[int(x) for x in members]);st.rerun()
        except ValueError as exc:st.error(diagnostic(exc))
    if groups:
        groups_by_id={g['group_id']:g for g in groups}
        gid=st.selectbox(tr('选择已有主题组'),list(groups_by_id),format_func=lambda x:groups_by_id[x]['name']+' · '+x,key=f'group_edit_select_{scope}')
        group=groups_by_id[gid];options=sorted(set(available+group['members']))
        with st.form(f'group_edit_form_{scope}_{gid}'):
            name=st.text_input(tr('修改主题组名称'),value=group['name'],max_chars=100,key=f'group_edit_name_{scope}_{gid}')
            members=st.multiselect(tr('调整主题组成员'),options,default=group['members'],format_func=lambda x:f'{x} · {names[x]}',key=f'group_edit_members_{scope}_{gid}')
            a,b=st.columns(2);save=a.form_submit_button(tr('保存主题组修改'), key=f'group_update_{scope}_{gid}');remove=b.form_submit_button(tr('解除该主题组'), key=f'group_dissolve_{scope}_{gid}')
        if save or remove:
            try:
                if remove:delete(result,gid);st.session_state.pop(f'group_edit_select_{scope}',None)
                else:
                    changed=update(result,gid,name,[int(x) for x in members])
                    if not changed:st.info(tr('组名和成员均未变化，未新增记录。'));return
                for key in list(st.session_state):
                    if key.startswith(('group_create_name_','group_create_members_','group_edit_name_','group_edit_members_')) and scope in key:st.session_state.pop(key,None)
                st.rerun()
            except ValueError as exc:st.error(diagnostic(exc))
    view=st.radio(tr('主题统计视图'),['原始主题','人工归组'], format_func=localize_options(),horizontal=True,key=f'group_view_{scope}')
    if view=='人工归组':
        stats=summary(result);st.caption(tr('片段数按归组累加；每组文档提及数按原始文档 ID 去重。未归组主题和离群类单独显示。'))
        st.dataframe(stats,hide_index=True,width='stretch')
    else:st.caption(tr('当前是原始主题视图；上方每行保持原有主题编号和关键词。'))
    obj=(result.get('visualization') or {}).get('topic_groups')
    if obj:
        with st.expander(tr('归组操作记录')):
            st.json(obj['history'])
        st.download_button(tr('下载主题归组映射 CSV'),csv_bytes(mapping(result)),'topic_group_mapping.csv','text/csv',key=f'group_mapping_{scope}')
        st.download_button(tr('下载主题归组统计 CSV'),csv_bytes(summary(result)),'topic_group_summary.csv','text/csv',key=f'group_summary_{scope}')
        st.caption(tr('完整归组片段对应 CSV 可在下方“准备导出”中按需生成。'))
        st.download_button(tr('下载主题归组记录 JSON'),json.dumps(obj,ensure_ascii=False,indent=2).encode(),'topic_group_history.json','application/json',key=f'group_history_{scope}')
        st.caption(tr('归组记录和原文随新项目 ZIP 保存；请下载新文件以保留当前修改。'))
