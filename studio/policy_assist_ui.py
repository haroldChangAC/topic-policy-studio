"""Opt-in model suggestions and explicit human review for policy coding."""
from studio.i18n import diagnostic, tr, localize_options
from studio.ui_layout import source_text
import pandas as pd
import streamlit as st

from studio.policy import active_version, get_policy, revision
from studio.policy_assist import MAX_SENT_CHARS, generate, latest_encoding, review, review_batch, status

LABELS={'yes':'符合','no':'不符合','pending':'待核查','not_applicable':'不适用'}


def render(project):
    st.markdown(tr('#### 模型辅助编码（须人工审核）'))
    st.caption(tr('仅主动点击时向本机 Ollama 兼容服务发送所选原文和指标说明；不下载模型，不接入外部 API。请确认本机服务及所选模型不会转发至云端。'))
    st.caption(tr('模型输出是待审核建议，不参与 PMC 得分。即使发送整篇，也无法验证模型是否完整处理；模型给出的“不符合”先转为待核查，研究者可独立修改判断。'))
    version=active_version(project)
    if not version:
        st.info(tr('请先确认指标体系，再生成建议。'));return
    if version['indicators']!=project['draft']:
        st.warning(tr('指标草稿已修改，请先确认新版本或撤销修改；历史建议仍可查看。'))
    policies={p['policy_id']:p for p in project['policies']};indicators={i['id']:i for i in version['indicators'] if i['level']==2}
    if not indicators:return
    pid=st.selectbox(tr('模型建议的政策'),list(policies),format_func=lambda x:policies[x]['title']+' · '+x[-10:],key='ai_policy')
    iid=st.selectbox(tr('模型建议的二级指标'),list(indicators),format_func=lambda x:indicators[x]['name']+' · '+x[-8:],key='ai_indicator')
    text=revision(policies[pid])['text']
    st.write(tr('指标说明：{0}', indicators[iid]['definition']))
    service=st.text_input(tr('本机 Ollama 服务地址'),value='http://127.0.0.1:11434',key='ai_service')
    model=st.text_input(tr('本机已有模型名称'),key='ai_model',help=tr('只使用用户已在本机配置的模型，不自动下载。'))
    mode=st.radio(tr('发送范围'),['整篇政策','手动选择原文范围'], format_func=localize_options(),horizontal=True,key='ai_scope')
    if mode=='手动选择原文范围' and text:
        left,right=st.columns(2)
        start=int(left.number_input(tr('发送起点（包含）'),min_value=0,max_value=len(text)-1,value=0,key='ai_start'))
        end=int(right.number_input(tr('发送终点（不包含）'),min_value=1,max_value=len(text),value=min(len(text),MAX_SENT_CHARS),key='ai_end'))
    else: start,end=0,len(text)
    valid=bool(text.strip()) and start<end and end-start<=MAX_SENT_CHARS
    st.caption(tr('将发送政策字符范围 [{0}, {1})，共 {2} / {3} 字符，另含指标名称和编码说明；位置按 Unicode 字符计。', f'{start}', f'{end}', f'{max(0, end - start)}', f'{len(text)}'))
    if not valid:st.warning(tr('正文为空、范围无效或超过 {0} 字符；请手动选择有效范围。', f'{MAX_SENT_CHARS}'))
    elif start!=0 or end!=len(text):st.warning(tr('只发送部分政策；未找到证据不能解释为整篇政策不符合。'))
    with st.expander(tr('预览将发送的政策文本和指标')):
        source_text(text[start:end] if valid else '')
        st.write(tr('指标：{0}', indicators[iid]['name']));source_text(indicators[iid]['definition'])
    if st.button(tr('向本机模型生成建议'),disabled=not valid or not model or version['indicators']!=project['draft'],key='ai_generate'):
        try:
            result=generate(project,pid,iid,model,start,end,service)
            st.session_state['ai_selected_suggestion']=result['suggestion_id']
            st.rerun()
        except ValueError as exc:st.error(diagnostic(exc))
    items=project.get('ai_suggestions',[])
    if not items:
        st.info(tr('尚无模型建议。手工编码功能仍可独立使用。'));return
    st.markdown(tr('##### 建议与审核'))
    lookup={s['suggestion_id']:s for s in items}
    if st.session_state.get('ai_selected_suggestion') not in lookup:
        st.session_state.pop('ai_selected_suggestion',None)
    suggestion=st.selectbox(tr('选择模型建议'),list(lookup),format_func=lambda x:f"{lookup[x]['created_at']} · {lookup[x]['model']} · {x[-8:]}",key='ai_selected_suggestion')
    item=lookup[suggestion];policy=get_policy(project,item['policy_id'])
    old_version=next(v for v in project['versions'] if v['version_id']==item['version_id'])
    indicator=next(i for i in old_version['indicators'] if i['id']==item['indicator_id'])
    historic=revision(policy,item['revision_id'])['text']
    st.caption(tr('政策：{0}（{1}）；指标：{2}（{3}）；版本：{4}；文本版本：{5}', f"{policy['title']}", f"{item['policy_id']}", f"{indicator['name']}", f"{item['indicator_id']}", f"{item['version_id']}", f"{item['revision_id']}"))
    st.write(tr('指标说明：{0}', indicator['definition']))
    st.caption(tr('请求模型：{0}；服务报告模型：{1}；模板：{2}；发送范围 [{3}, {4})；{5}，服务实际处理完整性无法验证。', f"{item['model']}", item['model_reported'] or tr('未报告（模拟或失败）'), f"{item['template_version']}", f"{item['start']}", f"{item['end']}", tr('已发送整篇' if item['full_text_sent'] else '仅发送部分')))
    previous=latest_encoding(project,item)
    detail = tr('{0} · 证据 {1} · 编码时间 {2}', tr(LABELS[previous['value']]), previous['evidence'], previous['changed_at']) if previous else tr('尚无')
    st.write(tr('当前正式编码：{0}', detail))
    current=status(project,item);st.write(tr('状态：{0}', tr(current)))
    if item['result']=='failed':st.error(tr('生成失败：{0}', diagnostic(item['error'])))
    else:
        st.write(tr('模型原始建议：{0}；待审核建议：{1}', tr(LABELS[item['model_status']]), tr(LABELS[item['suggested_status']])))
        st.write(tr('模型依据：{0}', item['rationale']));source_text(tr('模型证据：{0}', item['evidence']))
        st.write(tr('证据定位：{0}', tr({'none':'无证据','missing':'未找到','unique':'唯一匹配','multiple':'多处匹配','too_many':'匹配过多，仅显示前 100 处'}[item['match_status']])))
        if item['warning']:st.warning(diagnostic(item['warning']))
        for match in item['matches'][:10]:
            a,b=match['start'],match['end'];source_text(f"[{a}, {b}) {'发送范围内' if item['start']<=a<b<=item['end'] else '发送范围外'}：…{historic[max(0,a-35):min(len(historic),b+35)]}…")
    with st.expander(tr('查看模型原始响应和当时的完整政策文本')):
        source_text(item['raw_response']);source_text(historic)
    prior_review=next((r for r in project.get('ai_reviews',[]) if r['suggestion_id']==suggestion),None)
    if prior_review:st.json(prior_review)
    if current=='待审核':
        action=st.radio(tr('审核操作'),['接受建议','修改后写入正式编码','拒绝建议'], format_func=localize_options(),horizontal=True,key='ai_review_action_'+suggestion)
        st.caption(tr('选择“接受建议”时使用模型建议值和依据；需要修改时请选择“修改后写入正式编码”。'))
        matches=item['matches'];chosen=None
        if action=='接受建议' and item['evidence'] and matches:
            choices=matches if item['match_status']=='unique' else [None,*matches]
            chosen=st.selectbox(tr('选择证据匹配位置'),choices,
                                format_func=localize_options(lambda x:tr('请选择具体位置') if x is None else f"[{x['start']}, {x['end']})"),
                                key='ai_match_'+suggestion)
        with st.form('ai_review_form_'+suggestion):
            value=st.selectbox(tr('最终编码状态'),list(LABELS),index=list(LABELS).index(item['suggested_status']),format_func=localize_options(lambda x:tr(LABELS[x])),key='ai_final_value_'+suggestion)
            evidence=st.text_area(tr('人工核对后的证据'),value=item['evidence'],key='ai_final_evidence_'+suggestion)
            default_start=matches[0]['start'] if len(matches)==1 else 0
            default_end=matches[0]['end'] if len(matches)==1 else 0
            a,b=st.columns(2)
            start=int(a.number_input(tr('审核证据起点'),min_value=0,value=default_start,step=1,key='ai_final_start_'+suggestion))
            end=int(b.number_input(tr('审核证据终点（不包含）'),min_value=0,value=default_end,step=1,key='ai_final_end_'+suggestion))
            rationale=st.text_area(tr('人工判断依据'),value=item['rationale'],key='ai_final_rationale_'+suggestion)
            coder=st.text_input(tr('审核人（可留空）'),key='ai_reviewer_'+suggestion)
            replace=st.checkbox(tr('明确允许替换已有正式编码'),key='ai_replace_'+suggestion)
            submit=st.form_submit_button(tr('提交审核'), key='ai_review_submit_'+suggestion)
        if submit:
            try:
                op={'接受建议':'accept','修改后写入正式编码':'modify','拒绝建议':'reject'}[action]
                if op=='accept':a,b=(chosen['start'],chosen['end']) if chosen else (None,None)
                else:a,b=(start,end) if evidence else (None,None)
                review(project,suggestion,op,value,evidence,a,b,rationale,coder,replace)
                st.rerun()
            except ValueError as exc:st.error(diagnostic(exc))
    eligible=[s for s in items if status(project,s)=='待审核']
    with st.expander(tr('批量审核所选建议')):
        st.caption(tr('仅接受所选建议；同一政策和指标只能选一条。证据必须唯一且在发送范围内；已有正式编码须明确允许替换。任一条失败时整批不写入。'))
        selected=st.multiselect(tr('选择批量接受的建议'),[s['suggestion_id'] for s in eligible],
             format_func=localize_options(lambda x:f"{get_policy(project,lookup[x]['policy_id'])['title']} · {lookup[x]['indicator_id'][-8:]} · {tr(LABELS[lookup[x]['suggested_status']])} · {x[-8:]}"),key='ai_batch')
        if selected:
            st.dataframe(pd.DataFrame([{'suggestion_id':s,'policy_id':lookup[s]['policy_id'],'indicator_id':lookup[s]['indicator_id'],
                'suggested_status':lookup[s]['suggested_status'],'evidence':lookup[s]['evidence'],
                'match_status':lookup[s]['match_status'],'existing_encoding':(latest_encoding(project,lookup[s]) or {}).get('value','无')} for s in selected]),hide_index=True,width='stretch')
        batch_coder=st.text_input(tr('批量审核人（可留空）'),key='ai_batch_coder')
        batch_replace=st.checkbox(tr('批量明确允许替换已有正式编码'),key='ai_batch_replace')
        if st.button(tr('批量接受所选建议'),disabled=not selected,key='ai_batch_accept'):
            try:review_batch(project,selected,batch_replace,batch_coder);st.rerun()
            except ValueError as exc:st.error(diagnostic(exc))
