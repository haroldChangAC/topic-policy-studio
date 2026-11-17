"""Streamlit workflow for manual policy evaluation."""
from studio.i18n import diagnostic, tr, localize_options
from studio.ui_layout import anchor, source_text
import copy
import hashlib
import json
import uuid

import pandas as pd
import streamlit as st

from studio.corpus import read_csv, read_txt, read_xlsx, xlsx_sheets
from studio.policy import (RULE, active_version, add_indicator, append_policies, coding_matrix, confirm_indicators,
                           create_project, delete_indicator, edit_indicator, plan_append, record_encoding, revision, revise_policy, score)
from studio.policy_project import excel_policy, open_policy_project, save_policy_project, tables
from studio.segmentation import TokenBudget, split_text
from studio.workflow import csv_bytes
from studio.policy_charts import (MAX_POLICIES, METHOD, SURFACE_METHOD, chart_settings, drawing_data,
                                  render_radar, render_surface, surface_data, validate_chart_settings)
from studio.visualization import find_chinese_font

LABELS = {'yes':'1：符合','no':'0：不符合','pending':'待核查','not_applicable':'不适用'}


def _index(options, candidates):
    for candidate in candidates:
        if candidate in options:
            return options.index(candidate)
    return 0


def workspace(encoder_resource):
    st.subheader(tr('基础政策评价'))
    st.info(tr('本模块由研究者制定指标并审核编码；评价单位是整份政策。可选本机模型只提出建议，不自动赋值；不计算 PMC-AE。'))
    mode = st.radio(tr('政策评价工作方式'), ['导入政策', '打开政策评价项目'], format_func=localize_options(), horizontal=True, key='pmc_mode')
    if mode == '导入政策':
        uploaded = st.file_uploader(tr('上传政策 CSV、XLSX 或 TXT'), type=['csv','xlsx','txt'], max_upload_size=10, key='pmc_upload')
        if uploaded:
            try:
                raw=uploaded.getvalue(); extension=uploaded.name.rsplit('.',1)[-1].lower()
                if extension=='xlsx':
                    names=xlsx_sheets(raw)
                    if not names: raise ValueError('XLSX 中没有工作表。')
                    sheet=st.selectbox(tr('政策工作表'),names,key='pmc_sheet'); corpus=read_xlsx(raw,sheet)
                    st.caption(tr('Excel 公式读取已保存的缓存；正文公式无缓存时不能创建项目。不会执行公式。'))
                    if any(not x['cached_available'] for x in corpus.source_details.get('formula_cells', [])):
                        st.warning(tr('部分公式没有缓存；非正文列会保留公式表达式作为原始字段，请核对。'))
                else:
                    encoding=st.selectbox(tr('政策文件编码'),['utf-8-sig','gb18030','utf-16'],key='pmc_encoding')
                    corpus=read_txt(raw,encoding) if extension=='txt' else read_csv(raw,encoding)
                columns=corpus.frame.columns.tolist()
                text_col=st.selectbox(tr('政策正文列'),columns,index=_index(columns,['正文','text','文本','内容']),key='pmc_text_column')
                field_options=[None,*columns]
                labels={'title':('政策标题列（可选）',['标题','title','政策名称']),
                        'date':('发布时间列（可选）',['发布时间','日期','date']),
                        'agency':('发布机构列（可选）',['发布机构','机构','agency']),
                        'region':('地区列（可选）',['地区','region']),
                        'source':('来源列（可选）',['来源','source'])}
                mapping={}
                for key,(label,candidates) in labels.items():
                    mapping[key]=st.selectbox(tr(label),field_options,index=_index(field_options,candidates),
                                              format_func=localize_options(lambda x:tr('不选择') if x is None else x),key='pmc_field_'+key)
                st.dataframe(corpus.frame.head(100), hide_index=True, width='stretch')
                st.caption(tr('原始记录 {0} 条；每行作为一份政策，重复政策不会自动删除。原始字段完整保留。', f'{len(corpus.frame)}'))
                if st.session_state.get('pmc_project'):
                    st.warning(tr('创建新项目会替换当前会话中的政策项目；请先下载需要保留的结果。'))
                if st.button(tr('创建政策评价项目'),key='pmc_create',type='primary'):
                    st.session_state['pmc_project']=create_project(corpus,text_col,mapping,uploaded.name)
                    st.session_state['pmc_loaded_digest']=None
                    st.success(tr('已创建独立的政策评价项目；如有原有会话结果，请事先下载保存。'))
            except ValueError as exc:
                st.error(tr('政策材料无法导入：{0}', diagnostic(exc)))
    else:
        if st.session_state.get('pmc_project'):
            st.caption(tr('上传另一项目会替换当前会话中的政策项目；请先下载需要保留的结果。'))
        uploaded=st.file_uploader(tr('上传政策评价项目 ZIP'),type=['zip'],max_upload_size=50,key='pmc_project_upload')
        if uploaded:
            digest=hashlib.sha256(uploaded.getvalue()).hexdigest()
            if st.session_state.get('pmc_loaded_digest')!=digest:
                try:
                    st.session_state['pmc_project']=open_policy_project(uploaded.getvalue())
                    st.session_state['pmc_loaded_digest']=digest
                    st.success(tr('政策评价项目完整性与证据位置检查通过。'))
                except ValueError as exc:
                    st.error(tr('无法打开政策评价项目：{0}', diagnostic(exc)))
                    return
    project=st.session_state.get('pmc_project')
    if not project:
        st.info(tr('请先导入政策材料，或打开已有政策评价项目。'))
        return
    st.caption(tr('政策项目 {0}；{1} 份原始记录。项目包含原文和编码记录，请自行选择保存位置。', f"{project['project_id']}", f"{len(project['policies'])}"))
    message=st.session_state.pop('pmc_append_message',None)
    if message: st.success(message)
    _append(project)
    anchor('policy-source')
    _policies(project,encoder_resource)
    anchor('policy-indicators')
    _indicators(project)
    anchor('policy-coding')
    _coding(project)
    from studio.policy_assist_ui import render as render_assist
    render_assist(project)
    from studio.policy_long_ui import render as render_long
    render_long(project)
    anchor('policy-results')
    _results(project)
    _charts(project)
    anchor('policy-export')
    _exports(project)


def _append(project):
    with st.expander(tr('追加政策文件')):
        st.caption(tr('CSV/XLSX 可包含多份政策，TXT 每次一份。追加不会覆盖已有政策、指标版本、编码或得分；请下载新项目 ZIP 保存。'))
        if not active_version(project):
            st.info(tr('当前尚未确认指标体系；仍可追加政策，确认指标后才能编码。'))
        uploaded=st.file_uploader(tr('选择要追加的 CSV、XLSX 或 TXT'),type=['csv','xlsx','txt'],max_upload_size=10,key='pmc_append_upload')
        if not uploaded:
            return
        try:
            raw=uploaded.getvalue(); extension=uploaded.name.rsplit('.',1)[-1].lower()
            digest=hashlib.sha256(raw).hexdigest()
            if extension=='xlsx':
                names=xlsx_sheets(raw)
                if not names: raise ValueError('XLSX 中没有工作表。')
                sheet=st.selectbox(tr('追加文件工作表'),names,key='pmc_append_sheet')
                corpus=read_xlsx(raw,sheet)
                st.caption(tr('Excel 公式读取已保存的缓存；正文公式无缓存时不能追加。不会执行公式。'))
                if any(not x['cached_available'] for x in corpus.source_details.get('formula_cells', [])):
                    st.warning(tr('部分公式没有缓存；非正文列会保留公式表达式，请核对。'))
            else:
                encoding=st.selectbox(tr('追加文件编码'),['utf-8-sig','gb18030','utf-16'],key='pmc_append_encoding')
                corpus=read_txt(raw,encoding) if extension=='txt' else read_csv(raw,encoding)
            context=(project['project_id'],digest,corpus.source_format,corpus.source_details.get('sheet'))
            if st.session_state.get('pmc_append_context') != context:
                for key in ('pmc_append_text','pmc_append_id','pmc_existing_id','pmc_append_title','pmc_append_date','pmc_append_agency','pmc_append_region','pmc_append_source','pmc_append_duplicates'):
                    st.session_state.pop(key,None)
                st.session_state['pmc_append_context']=context
            columns=corpus.frame.columns.tolist()
            if not columns: raise ValueError('追加文件没有可选择的字段。')
            text_col=st.selectbox(tr('追加文件正文列'),columns,index=_index(columns,['正文','text','文本','内容']),key='pmc_append_text')
            fields=[None,*columns]
            id_column=st.selectbox(tr('追加文件原始政策 ID 列（可选）'),fields,
                                   index=_index(fields,['政策ID','政策编号','policy_id','id','编号','ID']),
                                   format_func=localize_options(lambda x:tr('不选择') if x is None else x),key='pmc_append_id')
            existing_columns=sorted({key for p in project['policies'] for key in p['original_fields']})
            existing_options=[None,*existing_columns]
            existing_id=st.selectbox(tr('原项目用于核对的原始 ID 列（可选）'),existing_options,
                                     index=_index(existing_options,[id_column,'政策ID','政策编号','policy_id','id','编号','ID']),
                                     format_func=localize_options(lambda x:tr('不选择') if x is None else x),key='pmc_existing_id')
            labels={'title':('追加文件标题列（可选）',['标题','title','政策名称']),
                    'date':('追加文件发布时间列（可选）',['发布时间','日期','date']),
                    'agency':('追加文件发布机构列（可选）',['发布机构','机构','agency']),
                    'region':('追加文件地区列（可选）',['地区','region']),
                    'source':('追加文件来源列（可选）',['来源','source'])}
            mapping={}
            for key,(label,candidates) in labels.items():
                mapping[key]=st.selectbox(tr(label),fields,index=_index(fields,candidates),
                                           format_func=localize_options(lambda x:tr('不选择') if x is None else x),key='pmc_append_'+key)
            duplicate=st.radio(tr('相同原文的处理方式'),['保留所有有效政策（默认）','排除相同原文'], format_func=localize_options(),key='pmc_append_duplicates')
            policy='keep' if duplicate=='保留所有有效政策（默认）' else 'exclude'
            plan=plan_append(project,corpus,text_col,mapping,uploaded.name,id_column,policy,existing_id)
            st.caption(tr('输入 {0} 条；正文有效 {1} 条；空正文 {2} 条；按当前选择将追加 {3} 条。', f"{len(plan['rows'])}", f"{plan['valid_count']}", f"{plan['empty_count']}", f"{len(plan['accepted'])}"))
            st.dataframe(corpus.frame.head(100),hide_index=True,width='stretch')
            st.dataframe(plan['rows'],hide_index=True,width='stretch')
            if plan['empty_count']:
                st.warning(tr('空白正文不会追加；请检查预览中的排除原因。'))
            if plan['duplicate_count']:
                st.warning(tr('发现 {0} 条与已有政策或本批次前面政策原文完全相同；按上方选择保留或排除，不覆盖已有政策。', f"{plan['duplicate_count']}"))
            if plan['external_id_conflicts'] or plan['system_id_collisions']:
                st.warning(tr('原始 ID 冲突 {0} 条；系统记录 ID 冲突 {1} 条。原始 ID 原样保留，系统 ID 在冲突时加唯一后缀；详情见预览。', f"{plan['external_id_conflicts']}", f"{plan['system_id_collisions']}"))
            st.caption(tr('重复按导入原文逐字比较；原始 ID 与所选列、常见 ID 列及系统 ID 比较并忽略两端空白。系统 ID 按文件哈希、工作表与行号生成，冲突时加后缀。'))
            if st.button(tr('确认追加政策'),disabled=not plan['accepted'],key='pmc_append_confirm'):
                applied=append_policies(project,corpus,text_col,mapping,uploaded.name,id_column,policy,existing_id)
                st.session_state['pmc_download_nonce']=uuid.uuid4().hex[:12]
                st.session_state['pmc_append_message']=f"已追加 {len(applied['accepted'])} 份政策；新政策各指标为待编码，请下载新的项目 ZIP。"
                st.rerun()
        except ValueError as exc:
            st.error(tr('追加文件无法使用：{0}', diagnostic(exc)))


def _policies(project,encoder_resource):
    st.markdown(tr('#### 1 · 政策与原文'))
    ids=[p['policy_id'] for p in project['policies']]
    lookup={p['policy_id']:p for p in project['policies']}
    selected=st.selectbox(tr('选择政策查看原文'),ids,format_func=lambda x:f"{lookup[x]['title']} · {x[-12:]}",key='pmc_policy_view')
    policy=lookup[selected]; current=revision(policy)
    if not current['text'].strip():
        st.warning(tr('此政策正文为空；原始行保留，但须先补充文本修订后才能编码。'))
    st.caption(tr('政策 ID：{0}；当前文本版本：{1}。原始字段和导入原文保留。', f'{selected}', f"{current['revision_id']}"))
    origin=policy.get('theme_origin')
    if origin:
        st.caption(tr('主题关联是转入时的研究参考，不是政策编码；之后源主题项目的修改不会自动同步。'))
        st.write(tr('来源主题运行：{0}；原始文档 ID：{1}；转入时间：{2}', f"{origin['project_run_id']}", f"{origin['document_id']}", f"{origin['transferred_at']}"))
        if origin['segments']:
            st.dataframe(origin['segments'],hide_index=True,width='stretch')
        else:
            st.info(tr('该原始文档没有已保存的建模片段。'))
    with st.expander(tr('查看完整政策原文及字段')):
        st.json(policy['original_fields'])
        st.write(tr('当前文本：')); source_text(current['text'])
        if len(policy['revisions'])>1:
            st.write(tr('导入原文：')); source_text(policy['original_text'])
        st.dataframe([{'revision_id':v['revision_id'],'created_at':v['created_at'],'note':v['note']} for v in policy['revisions']],hide_index=True,width='stretch')
    with st.expander(tr('修订政策文本（保留导入原文）')):
        st.caption(tr('保存为新文本版本；原文和先前证据保留，旧文本版本的编码须重新确认才能计入总分。'))
        key=f"pmc_text_{selected}_{current['revision_id']}"
        with st.form('pmc_revision_form'):
            edited=st.text_area(tr('修订后的政策文本'),value=current['text'],key=key,height=160)
            note=st.text_input(tr('修订说明（可留空）'),key='pmc_revision_note')
            submit=st.form_submit_button(tr('保存政策文本修订'), key='pmc_save_revision')
        if submit:
            try:
                if revise_policy(project,selected,edited,note): st.rerun()
                st.info(tr('正文未变化。'))
            except ValueError as exc: st.error(diagnostic(exc))
    with st.expander(tr('按现有模型长度切分以便查看')):
        st.caption(tr('切分仅供查看，保留原文位置；评分始终以整份政策为单位。需要本地模型 tokenizer，不重新训练。'))
        if st.button(tr('生成政策查看片段'),key='pmc_split'):
            try:
                encoder,lock=encoder_resource('cpu')
                with lock:
                    budget=TokenBudget(encoder); spans=split_text(current['text'],budget)
                    rows=[{'order':i,'start':a,'end':b,'text':current['text'][a:b],'tokens':budget.count(current['text'][a:b])}
                          for i,(a,b) in enumerate(spans,1)]
                st.session_state['pmc_split_result']={'revision_id':current['revision_id'],'rows':rows,'limit':budget.limit}
            except Exception as exc: st.error(tr('切分未完成：{0}', diagnostic(exc)))
        view=st.session_state.get('pmc_split_result')
        if view and view['revision_id']==current['revision_id']:
            st.caption(tr('token 上限 {0}（含特殊 token）；位置从 0 开始，终点不包含。', f"{view['limit']}"))
            st.dataframe(view['rows'],hide_index=True,width='stretch')


def _indicators(project):
    st.markdown(tr('#### 2 · 制定并确认指标体系'))
    st.caption(tr('示例仅演示操作，不预设适用于任何政策。一级或二级指标的名称和说明均须由研究者决定。'))
    active=active_version(project)
    if active and active['indicators']!=project['draft']:
        st.warning(tr('草稿与已确认版本不同。确认后将建立新版本，旧版本及其编码完整保留；新版本须重新编码。'))
        if st.button(tr('撤销草稿修改并恢复当前确认版本'),key='pmc_discard_draft'):
            project['draft']=copy.deepcopy(active['indicators']); st.rerun()
    if st.button(tr('载入可编辑的 A/B/C 演示指标'),disabled=bool(project['draft']),key='pmc_example'):
        for name in ('A','B','C'):
            parent=add_indicator(project,1,'示例 '+name,'演示用一级指标，请自行修改。')
            for n in (1,2): add_indicator(project,2,f'示例 {name}{n}','演示用二级指标，请自行定义。',parent)
        st.rerun()
    with st.form('pmc_add_primary'):
        name=st.text_input(tr('新增一级指标名称'),key='pmc_new_primary_name')
        definition=st.text_area(tr('一级指标定义或编码说明'),key='pmc_new_primary_definition')
        submit=st.form_submit_button(tr('新增一级指标'), key='pmc_add_primary_submit')
    if submit:
        try: add_indicator(project,1,name,definition); st.rerun()
        except ValueError as exc: st.error(diagnostic(exc))
    parents=[i for i in project['draft'] if i['level']==1]; byid={i['id']:i for i in project['draft']}
    if parents:
        pids=[i['id'] for i in parents]
        with st.form('pmc_add_secondary'):
            parent=st.selectbox(tr('所属一级指标'),pids,format_func=lambda x:byid[x]['name'],key='pmc_new_secondary_parent')
            name=st.text_input(tr('新增二级指标名称'),key='pmc_new_secondary_name')
            definition=st.text_area(tr('二级指标定义或编码说明'),key='pmc_new_secondary_definition')
            submit=st.form_submit_button(tr('新增二级指标'), key='pmc_add_secondary_submit')
        if submit:
            try: add_indicator(project,2,name,definition,parent); st.rerun()
            except ValueError as exc: st.error(diagnostic(exc))
    if project['draft']:
        st.dataframe(project['draft'],hide_index=True,width='stretch')
        choices=[i['id'] for i in project['draft']]
        choice=st.selectbox(tr('选择修改或删除的指标'),choices,format_func=localize_options(lambda x:tr('一级' if byid[x]['level']==1 else '二级') + f" · {byid[x]['name']} · {x[-8:]}"),key='pmc_edit_indicator')
        item=byid[choice]
        with st.form('pmc_edit_form'):
            name=st.text_input(tr('修改指标名称'),value=item['name'],key='pmc_edit_name_'+choice)
            definition=st.text_area(tr('修改指标定义或说明'),value=item['definition'],key='pmc_edit_def_'+choice)
            parent=(st.selectbox(tr('调整所属一级指标'),[i['id'] for i in parents],
                                 index=[i['id'] for i in parents].index(item['parent_id']),format_func=lambda x:byid[x]['name'],
                                 key='pmc_edit_parent_'+choice) if item['level']==2 else None)
            save=st.form_submit_button(tr('保存指标修改'), key='pmc_save_indicator'); delete=st.form_submit_button(tr('删除指标'), key='pmc_delete_indicator')
        if save or delete:
            try:
                if delete: delete_indicator(project,choice)
                else: edit_indicator(project,choice,name,definition,parent)
                st.rerun()
            except ValueError as exc: st.error(diagnostic(exc))
        st.caption(tr('删除一级指标会同时从草稿删除其二级指标；已确认版本与旧编码保留。'))
    if st.button(tr('确认当前指标体系'),key='pmc_confirm'):
        try:
            if confirm_indicators(project): st.success(tr('指标体系已确认；此前版本和编码均保留。'))
            else: st.info(tr('指标未变化，沿用当前版本。'))
        except ValueError as exc: st.error(diagnostic(exc))
    active=active_version(project)
    st.caption(tr('当前确认版本：{0}', active['version_id'] if active else tr('尚未确认')))
    with st.expander(tr('已确认指标版本历史')):
        for version in project['versions']:
            st.write(f"{version['version_id']} · {version['confirmed_at']}")
            st.dataframe(version['indicators'],hide_index=True,width='stretch')


def _coding(project):
    st.markdown(tr('#### 3 · 人工编码与证据'))
    version=active_version(project)
    if not version:
        st.info(tr('请先确认一套完整指标体系，再进行人工编码。'))
        return
    dirty=version['indicators']!=project['draft']
    if dirty: st.warning(tr('草稿尚未确认；请确认新版本或恢复草稿后继续编码。已确认版本的现有结果仍可查看。'))
    policies={p['policy_id']:p for p in project['policies']}
    indicators={i['id']:i for i in version['indicators'] if i['level']==2}
    if not indicators:return
    pid=st.selectbox(tr('编码政策'),list(policies),format_func=lambda x:policies[x]['title']+' · '+x[-10:],key='pmc_code_policy')
    iid=st.selectbox(tr('编码二级指标'),list(indicators),format_func=lambda x:indicators[x]['name']+' · '+x[-8:],key='pmc_code_indicator')
    policy=policies[pid];text=revision(policy)['text']
    state_rows=[row for row in coding_matrix(project) if row['policy_id']==pid]
    st.caption(tr('当前指标逐项状态；“待编码”表示尚无人工记录，不是 0 或“待核查”。'))
    st.dataframe(pd.DataFrame(state_rows)[['indicator_name','status','binary_value','indicator_id','policy_id','version_id','event_id']],hide_index=True,width='stretch')
    context=(project['project_id'],version['version_id'],pid,iid,policy['current_revision'])
    if st.session_state.get('pmc_code_context') != context:
        for key in ('pmc_value','pmc_evidence','pmc_precise','pmc_start','pmc_end','pmc_rationale'):
            st.session_state.pop(key,None)
        st.session_state['pmc_code_context']=context
    st.caption(tr('当前指标说明：{0}；编码人可留空。', f"{indicators[iid]['definition']}"))
    st.caption(tr('“不符合”是人工判断，不会因软件未找到证据而自动赋 0。证据可不定位；勾选精确位置时必须与当前正文完全一致。'))
    with st.expander(tr('查看当前政策全文')):
        source_text(text)
    with st.form('pmc_encoding_form'):
        state=st.selectbox(tr('编码状态'),['请选择','1：符合','0：不符合','待核查','不适用'], format_func=localize_options(),key='pmc_value')
        evidence=st.text_area(tr('证据原文（可留空）'),key='pmc_evidence')
        precise=st.checkbox(tr('校验精确原文位置'),key='pmc_precise')
        a,b=st.columns(2)
        start=a.number_input(tr('证据起点（从 0 开始）'),min_value=0,value=0,step=1,key='pmc_start')
        end=b.number_input(tr('证据终点（不包含）'),min_value=0,value=0,step=1,key='pmc_end')
        rationale=st.text_area(tr('判断依据或备注（可留空）'),key='pmc_rationale')
        coder=st.text_input(tr('编码人（可留空）'),key='pmc_coder')
        save=st.form_submit_button(tr('保存人工编码'), key='pmc_save_manual',disabled=dirty)
    if save:
        try:
            inverse={v:k for k,v in LABELS.items()}
            if state not in inverse: raise ValueError('请选择编码状态。')
            changed=record_encoding(project,pid,iid,inverse[state],evidence,int(start) if precise else None,
                                    int(end) if precise else None,rationale,coder)
            st.success(tr('已保存编码记录。' if changed else '编码没有变化，未新增记录。'))
        except ValueError as exc: st.error(diagnostic(exc))
    with st.expander(tr('编码与证据修改历史（含历史指标版本）')):
        st.dataframe(tables(project)['encodings'],hide_index=True,width='stretch')
        st.caption(tr('表中的待编码状态行仅用于显示尚无人工记录，不会创建赋值、编码人或修改时间。'))
        st.caption(tr('精确证据的位置对应记录中的文本版本；未提供位置的证据为人工填写，未作精确定位校验。'))
        if project['encodings']:
            events={event['event_id']:event for event in project['encodings']}
            selected=st.selectbox(tr('选择历史编码回查对应文本版本'),list(events),
                                  format_func=lambda x:f"{events[x]['changed_at']} · {events[x]['value']} · {x[-8:]}",
                                  key='pmc_history_event')
            event=events[selected]; historic=policies[event['policy_id']]
            st.caption(tr('政策 ID：{0}；指标 ID：{1}；文本版本：{2}；位置：{3}–{4}', f"{event['policy_id']}", f"{event['indicator_id']}", f"{event['revision_id']}", f"{event['start']}", f"{event['end']}"))
            st.write(tr('记录的证据：')); source_text(event['evidence'])
            st.write(tr('该编码对应的完整政策文本：')); source_text(revision(historic,event['revision_id'])['text'])


def _results(project):
    st.markdown(tr('#### 4 · 基础 PMC 计算'))
    st.caption(tr('一级得分 = 二级 0/1 赋值之和 ÷ 二级指标数量；原始 PMC = 一级得分之和；10 分制 = 原始 PMC ÷ 一级指标数 × 10。'))
    st.caption(tr('缺失、待核查、不适用或文本修订后未重新确认的编码均阻止完整总分；不调整分母、不自动排名，也不判断实际实施效果。'))
    if st.button(tr('计算政策评价'),key='pmc_calculate'):
        st.session_state['pmc_show_scores']=True
    if st.session_state.get('pmc_show_scores'):
        primary,policies=score(project)
        st.write(tr('一级指标计算（分子为目前已明确的符合项数，未完成时不得当作完整分数）：'))
        st.dataframe(pd.DataFrame(primary),hide_index=True,width='stretch')
        st.write(tr('政策得分（理论满分为一级指标数）：'))
        st.dataframe(tables(project)['policy_scores'],hide_index=True,width='stretch')
        if any(x['raw_pmc'] is None for x in policies):
            st.warning(tr('部分政策尚不能计算完整总分；具体原因见得分表。'))
    with st.expander(tr('查看历史确认版本的计算结果')):
        for version in project['versions']:
            st.write(version['version_id'])
            primary,policies=score(project,version['version_id'])
            st.dataframe(pd.DataFrame(policies),hide_index=True,width='stretch')


def _chart_output(frame, settings, font_path, kind, scope):
    if not font_path:
        st.warning(tr('未找到支持中文的本机字体；数据表仍可下载。'))
        return
    draw=render_radar if kind=='radar' else render_surface
    try:
        png,missing=draw(frame,settings,font_path,translate=tr)
        preview=png if settings['dpi']==150 else draw(frame,{**settings,'dpi':150},font_path,translate=tr)[0]
        if missing: st.warning(tr('当前字体缺少部分字符：')+''.join(missing[:30]))
        st.image(preview,width='stretch')
        st.download_button(tr('下载雷达图 PNG' if kind=='radar' else '下载曲面图 PNG'),png,
                           f'pmc_{kind}.png','image/png',key='pmc_'+kind+'_png_'+scope)
        svg,_=draw(frame,settings,font_path,'svg',translate=tr)
        st.download_button(tr('下载雷达图 SVG' if kind=='radar' else '下载曲面图 SVG'),svg,
                           f'pmc_{kind}.svg','image/svg+xml',key='pmc_'+kind+'_svg_'+scope)
    except (ValueError,RuntimeError) as exc:
        st.warning(tr('图表无法完整生成：{0}', diagnostic(exc)))


def _charts(project):
    st.markdown(tr('#### 政策评价图表'))
    version=active_version(project)
    if not version:
        st.info(tr('请先确认指标体系并人工编码；不会自动赋值。'))
        return
    settings=chart_settings(project)
    scope=(st.session_state.get('pmc_loaded_digest') or project['project_id'])[:16]+'_'+version['version_id']
    st.caption(tr('仅比较当前已确认指标版本 {0}，不混合历史版本；图表每次从当前编码和文本版本重新计算，不沿用旧图数值。', f"{version['version_id']}"))
    st.caption(tr('图表设置与排列随新下载项目保存；长政策分段任务使用版本 6，其他项目按内容使用版本 1–5。原文件不受影响。SVG 文字为轮廓，不能直接作为文字编辑。'))
    if version['indicators']!=project['draft']:
        st.warning(tr('指标草稿已修改；请确认新版本或撤销草稿修改后绘图，旧图不会作为当前结果。'))
        return
    a,b,c,d=st.columns(4)
    style=a.selectbox(tr('政策图表样式'),['论文彩色','论文黑白'], format_func=localize_options(),index=0 if settings['style']=='color' else 1,key='pmc_chart_style_'+scope)
    width=b.number_input(tr('政策图表宽度（英寸）'),min_value=4.0,max_value=16.0,step=.5,value=float(settings['width_inches']),key='pmc_chart_width_'+scope)
    height=c.number_input(tr('政策图表高度（英寸）'),min_value=3.0,max_value=12.0,step=.5,value=float(settings['height_inches']),key='pmc_chart_height_'+scope)
    dpi=d.selectbox(tr('政策图表 PNG 分辨率（DPI）'),[150,300,600],index=[150,300,600].index(settings['dpi']),key='pmc_chart_dpi_'+scope)
    a,b=st.columns(2)
    font_size=a.number_input(tr('政策图表字号（磅）'),min_value=8,max_value=18,value=int(settings['font_size']),key='pmc_chart_font_'+scope)
    show_title=b.checkbox(tr('显示政策图表标题'),value=settings['show_title'],key='pmc_chart_title_'+scope)
    if width*height*dpi*dpi>36000000:
        st.warning(tr('图表超过 3600 万像素，请减小尺寸或分辨率；未保存超限设置。'))
        return
    settings.update({'style':'color' if style=='论文彩色' else 'mono','width_inches':float(width),'height_inches':float(height),
                     'dpi':int(dpi),'font_size':int(font_size),'show_title':show_title})
    font_path,family=find_chinese_font()
    if font_path:st.caption(tr('本机绘图字体：{0}；PNG {1} × {2} 像素，{3} DPI。', f'{family}', f'{int(width * dpi)}', f'{int(height * dpi)}', f'{dpi}'))
    policies={p['policy_id']:p for p in project['policies']};ids=list(policies)
    label=lambda pid:policies[pid]['title']+' · '+pid[-10:]
    primary=[i for i in version['indicators'] if i['level']==1]; primary_ids=[i['id'] for i in primary]
    st.markdown(tr('##### PMC 雷达图'))
    st.caption(tr('一级指标轴固定 0–1；最多比较 8 份政策。不同版本、未完成编码或文本修订后未确认的政策不能参与完整得分图。'))
    selected=st.multiselect(tr('选择雷达图政策'),ids,default=settings['radar_policy_ids'],format_func=label,
                            max_selections=MAX_POLICIES,key='pmc_radar_policies_'+scope)
    settings['radar_policy_ids']=list(selected)
    if selected:
        frame,blocked=drawing_data(project,list(selected),version['version_id'])
        st.dataframe(frame,hide_index=True,width='stretch')
        st.download_button(tr('下载雷达图数据 CSV'),csv_bytes(frame),'pmc_radar_data.csv','text/csv',key='pmc_radar_csv_'+scope)
        if blocked:
            for item in blocked:st.warning(tr('{0} 不能绘图：{1}', f"{item['policy_name']}", f"{item['reason']}"))
        elif len(primary)<3:
            st.info(tr('雷达图至少需要三个一级指标；数据表仍可查看与下载。'))
        else:
            if len(primary)>12:st.warning(tr('一级指标较多，标签可能拥挤；请增大画布并检查导出结果。'))
            _chart_output(frame,settings,font_path,'radar',scope)
    else:
        st.info(tr('请选择至少一份政策查看雷达图数据。'))
    st.markdown(tr('##### 3×3 基础 PMC 曲面图'))
    st.caption(tr('九个节点是九项实际得分。{0}', tr(SURFACE_METHOD)))
    surface=st.selectbox(tr('选择曲面图政策'),[None,*ids],
                         index=([None,*ids].index(settings['surface_policy_id']) if settings['surface_policy_id'] in [None,*ids] else 0),
                         format_func=localize_options(lambda x:tr('请选择') if x is None else label(x)),key='pmc_surface_policy_'+scope)
    settings['surface_policy_id']=surface
    if len(primary_ids)!=9:
        st.info(tr('当前有 {0} 个一级指标；3×3 曲面必须恰好有九个，不补零、不删减或复制。', f'{len(primary_ids)}'))
        if surface:
            frame,blocked=drawing_data(project,[surface],version['version_id'])
            st.dataframe(frame,hide_index=True,width='stretch')
            for item in blocked:st.warning(tr('{0} 尚未完成：{1}', f"{item['policy_name']}", f"{item['reason']}"))
        return
    names={i['id']:i['name'] for i in primary}
    st.caption(tr('矩阵从左到右、从上到下排列；比较不同政策时请保持相同排列。调整时允许暂时重复，只有九项互不重复才保存并绘图。'))
    order=[]
    for row in range(3):
        cols=st.columns(3)
        for col in range(3):
            i=row*3+col;old=settings['surface_order'][i]
            order.append(cols[col].selectbox(tr('第 {0} 行第 {1} 列', row+1, col+1),primary_ids,index=primary_ids.index(old),
                                              format_func=lambda x:names[x]+' · '+x[-8:],key=f'pmc_matrix_{scope}_{i}'))
    if len(set(order))!=9:
        st.warning(tr('矩阵排列有重复指标；请选择九个不同指标。当前不绘图，也不会保存不完整排列。'))
        return
    settings['surface_order']=order
    validate_chart_settings(project,project['chart_settings'])
    if not surface:
        st.info(tr('请选择一份政策查看矩阵数据。'))
        return
    frame,blocked=drawing_data(project,[surface],version['version_id'])
    matrix=surface_data(frame,order,require_complete=False)
    st.dataframe(matrix,hide_index=True,width='stretch')
    st.download_button(tr('下载曲面矩阵映射 CSV'),csv_bytes(matrix),'pmc_surface_matrix.csv','text/csv',key='pmc_surface_csv_'+scope)
    if blocked:
        for item in blocked:st.warning(tr('{0} 不能绘图：{1}', f"{item['policy_name']}", f"{item['reason']}"))
        return
    _chart_output(matrix,settings,font_path,'surface',scope)


def _exports(project):
    st.markdown(tr('#### 5 · 保存与导出'))
    st.warning(tr('项目 ZIP 含政策原文、修订、指标和人工编码证据，请自行选择安全保存位置。重新下载为新文件，不覆盖原始上传文件。'))
    try:
        data=save_policy_project(project)
        if st.session_state.get('pmc_download_project_id')!=project['project_id']:
            st.session_state['pmc_download_project_id']=project['project_id'];st.session_state['pmc_download_nonce']=uuid.uuid4().hex[:12]
        st.download_button(tr('下载政策评价项目 ZIP'),data,file_name=f"{project['project_id']}-save-{st.session_state['pmc_download_nonce']}.pmc.zip",mime='application/zip',key='pmc_download_project')
    except ValueError as exc:
        st.error(tr('项目无法保存：{0}', diagnostic(exc)))
    for name,frame in tables(project).items():
        labels={'policies':'政策信息','indicators':'指标体系','encodings':'编码与证据','coding_status':'逐项编码状态','primary_scores':'一级指标计算',
                'policy_scores':'政策得分','text_revisions':'政策文本版本','import_sources':'导入来源',
                'ai_suggestions':'模型建议记录','ai_reviews':'模型建议审核记录',
                'long_jobs':'长政策任务','long_segments':'长政策分段','long_evidence':'长政策候选证据','long_unresolved':'长政策未定位证据'}
        st.download_button(tr('下载{0} CSV', tr(labels[name])),csv_bytes(frame),file_name=name+'.csv',mime='text/csv',key='pmc_csv_'+name)
    try:
        st.download_button(tr('下载政策评价 Excel'),excel_policy(project),'policy_evaluation.xlsx',
                           'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',key='pmc_excel')
    except ValueError as exc:
        st.warning(tr('Excel 无法无损导出：{0}', diagnostic(exc)))
    st.download_button(tr('下载 PMC 计算规则 JSON'),json.dumps(RULE,ensure_ascii=False,indent=2).encode(),'pmc_rule.json','application/json',key='pmc_rule')
