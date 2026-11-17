"""Whole-document transfer controls shown inside an opened theme project."""
from studio.i18n import diagnostic, tr, localize_options
import hashlib
import uuid

import pandas as pd
import streamlit as st

from studio.policy_project import open_policy_project
from studio.policy_transfer import apply_transfer, documents, plan_transfer


def _choose(document_id, key):
    selected = st.session_state['theme_transfer_selection']
    if st.session_state[key]: selected.add(document_id)
    else: selected.discard(document_id)


def _transfer(result, digest, ids, mapping, append, duplicate, text_column):
    try:
        target = st.session_state.get('pmc_project') if append else None
        if append and not target:
            raise ValueError('当前没有可追加的政策项目。')
        plan = plan_transfer(result, digest, ids, mapping, target, duplicate, text_column)
        project = apply_transfer(plan, target)
        st.session_state['pmc_project'] = project
        st.session_state['pmc_loaded_digest'] = None
        st.session_state['pmc_download_nonce'] = uuid.uuid4().hex[:12]
        st.session_state['pmc_append_message'] = f"已转入 {len(plan['accepted'])} 篇原始文档；主题关联仅作参考，确认指标后新政策各项均待编码。请下载新政策项目 ZIP。"
        st.session_state['pmc_mode'] = '导入政策'
        st.session_state['work_mode'] = '基础政策评价'
    except ValueError as exc:
        st.session_state['theme_transfer_error'] = str(exc)


def render(result, digest):
    with st.expander(tr('转入基础政策评价')):
        st.caption(tr('主题聚类单位可能是片段；转入和 PMC 评价单位为整篇原始文档。每篇文档只列一次，其多个片段与主题作为来源快照保留，不自动编码；项目之间不会双向同步。'))
        error = st.session_state.pop('theme_transfer_error', None)
        if error: st.error(error)
        if st.session_state.get('theme_transfer_digest') != digest:
            st.session_state['theme_transfer_digest'] = digest
            st.session_state['theme_transfer_selection'] = set()
        existing = st.session_state.get('pmc_project')
        uploaded = st.file_uploader(tr('可选：打开已有政策项目作为追加目标'), type=['zip'], max_upload_size=50, key='theme_transfer_policy_upload')
        if uploaded:
            st.warning(tr('加载政策项目会替换当前会话中的政策项目；请先保存需要保留的结果。'))
            if st.button(tr('加载追加目标'), key='theme_transfer_load_policy'):
                try:
                    existing = open_policy_project(uploaded.getvalue())
                    st.session_state['pmc_project'] = existing
                    st.session_state['pmc_loaded_digest'] = hashlib.sha256(uploaded.getvalue()).hexdigest()
                    st.success(tr('追加目标已载入；原始上传文件不变。'))
                except ValueError as exc:
                    st.error(tr('无法打开政策项目：{0}', diagnostic(exc)))
        targets = ['新建政策评价项目'] + (['追加到当前政策项目'] if existing else [])
        target_label = st.radio(tr('转入目标'), targets, format_func=localize_options(), key='theme_transfer_target')
        append = target_label == '追加到当前政策项目'
        target = existing if append else None
        if target:
            st.caption(tr('目标项目：{0}；已有 {1} 份政策。', f"{target['project_id']}", f"{len(target['policies'])}"))
        elif existing:
            st.warning(tr('新建会替换当前会话中的政策项目；请先下载需要保留的结果。'))
        topic_options = ['全部原始文档', *sorted(result['topics']['Topic'].astype(int).tolist())]
        chosen_topic = st.selectbox(tr('按主题筛选相关片段所属文档'), topic_options,
                                    format_func=localize_options(lambda x: tr(x) if isinstance(x,str) else tr('主题 {0}', f'{x}')), key='theme_transfer_topic')
        all_docs = documents(result)
        shown = all_docs if isinstance(chosen_topic,str) else documents(result, chosen_topic)
        st.caption(tr('筛选到 {0} 篇原始文档；勾选可跨主题筛选保留，每篇文档只会转入一次。', f'{len(shown)}'))
        pages = max(1,(len(shown)+19)//20)
        page = st.number_input(tr('文档页码（每页 20 篇）'),min_value=1,max_value=pages,value=1,step=1,
                               key=f'theme_transfer_page_{digest[:12]}_{chosen_topic}')
        selection = st.session_state['theme_transfer_selection']
        columns=result['metadata']['original_columns']; options=[None,*columns]
        text_column=result['metadata'].get('text_column')
        if text_column not in columns:
            st.warning(tr('历史项目未记录正文列；请明确选择与原始文档完整原文一致的字段。'))
            text_column=st.selectbox(tr('历史项目正文列'),options,format_func=localize_options(lambda x:tr('请选择') if x is None else x),
                                     key=f'theme_transfer_text_{digest[:12]}')
        for document in shown[(page-1)*20:page*20]:
            doc_id=document['document_id']; key='theme_transfer_doc_'+hashlib.sha256((digest+doc_id).encode()).hexdigest()
            title=next((document['original_fields'].get(x,'') for x in ('标题','title','政策名称','名称') if document['original_fields'].get(x,'')), '')
            st.checkbox(tr('{0} · {1} · {2} 个片段', title or tr('未命名文档'), f'{doc_id}', f"{len(document['segments'])}"), value=doc_id in selection,
                        key=key, on_change=_choose, args=(doc_id,key))
        st.caption(tr('已勾选 {0} 篇；可在不同主题或页面继续勾选。', f'{len(selection)}'))
        labels={'title':('标题字段',['标题','title','政策名称','名称']),
                'date':('发布时间或年份字段',['发布时间','日期','年份','date','year']),
                'agency':('发布机构字段',['发布机构','机构','agency']),
                'region':('地区字段',['地区','region']), 'source':('来源字段',['来源','source'])}
        mapping={}
        for key,(label,candidates) in labels.items():
            default=next((options.index(x) for x in candidates if x in options),0)
            mapping[key]=st.selectbox(tr('{0}（可选）', tr(label)),options,index=default,format_func=localize_options(lambda x:tr('不选择') if x is None else x),
                                      key=f'theme_transfer_field_{digest[:12]}_{key}')
        duplicate_label=st.radio(tr('原文重复时'),['保留所有有效政策（默认）','排除相同原文'], format_func=localize_options(),key='theme_transfer_duplicates')
        duplicate='keep' if duplicate_label.startswith('保留') else 'exclude'
        if not selection:
            st.info(tr('请选择要转入的原始文档。')); return
        try:
            ids=[d['document_id'] for d in all_docs if d['document_id'] in selection]
            plan=plan_transfer(result,digest,ids,mapping,target,duplicate,text_column)
            st.caption(tr('目标：{0}；所选 {1} 篇，正文有效 {2} 篇，实际将转入 {3} 篇。', tr(target_label), f'{len(ids)}', f"{plan['valid_count']}", f"{len(plan['accepted'])}"))
            previews=[]
            for row,doc in zip(plan['rows'],plan['documents']):
                previews.append({**row, **{'原字段::'+k:v for k,v in doc['original_fields'].items() if k!=text_column}})
            st.dataframe(pd.DataFrame(previews),hide_index=True,width='stretch')
            if plan['empty_count']:st.warning(tr('空正文不会转入。'))
            if plan['duplicate_count']:st.warning(tr('发现 {0} 篇原文重复；按当前选择保留或排除。', f"{plan['duplicate_count']}"))
            if plan['system_id_collisions'] or plan['theme_id_conflicts'] or plan['document_id_conflicts']:
                st.warning(tr('已转入相同主题文档 {0} 篇；原文档 ID 与既有 ID 冲突 {1} 篇；系统 ID 冲突 {2} 篇。系统冲突使用唯一后缀，原文不会覆盖。', f"{plan['theme_id_conflicts']}", f"{plan['document_id_conflicts']}", f"{plan['system_id_collisions']}"))
            st.button(tr('确认转入并进入政策评价'), disabled=not plan['accepted'], key='theme_transfer_confirm', type='primary',
                      on_click=_transfer, args=(result,digest,ids,mapping,append,duplicate,text_column))
        except ValueError as exc:
            st.error(tr('无法预览转入：{0}', diagnostic(exc)))
