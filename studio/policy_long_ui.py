"""Interactive, resumable long-policy inspection and evidence review."""
from studio.i18n import diagnostic, tr, localize_options
from studio.ui_layout import source_text
import json

import pandas as pd
import streamlit as st

from studio.policy import active_version, get_policy, revision
from studio.policy_assist import latest_encoding, review, status
from studio.policy_long import cancel_remaining, candidates, coverage, create_job, is_current, process_segment, split_ranges


def render(project):
    with st.expander(tr('长政策分段辅助检查')):
        st.caption(tr('原文保持不变。每段保留完整文档位置和重叠范围；精确匹配只说明文字存在，不代表支持指标。模型建议不自动写入正式编码。'))
        st.warning(tr('请求 4096 token 上下文，预留 512 输出 token 和 512 服务模板估算；使用提示及正文 UTF-8 字节数估算，不是 Qwen tokenizer 实测，服务实际处理是否完整无法确认。'))
        version=active_version(project)
        if not version:
            st.info(tr('请先确认指标体系。'));return
        policies={p['policy_id']:p for p in project['policies']};inds={i['id']:i for i in version['indicators'] if i['level']==2}
        if not inds:return
        pid=st.selectbox(tr('长文本检查政策'),list(policies),format_func=lambda x:policies[x]['title']+' · '+x[-10:],key='long_policy')
        iid=st.selectbox(tr('长文本检查指标'),list(inds),format_func=lambda x:inds[x]['name']+' · '+x[-8:],key='long_indicator')
        model=st.text_input(tr('长文本本机模型名称'),value=st.session_state.get('ai_model',''),key='long_model')
        address=st.text_input(tr('长文本本机服务地址'),value='http://127.0.0.1:11434',key='long_address')
        overlap=int(st.number_input(tr('分段重叠字符数'),min_value=0,max_value=128,value=40,step=1,key='long_overlap'))
        text=revision(policies[pid])['text'];dirty=version['indicators']!=project['draft']
        try:
            spans,cfg=split_ranges(text,inds[iid],overlap)
            st.caption(tr('预览 {0} 段；正文估算预算 {1} UTF-8 字节／段，重叠最多 {2} 字符。位置从 0 开始、终点不含。', f'{len(spans)}', f"{cfg['body_utf8_byte_budget']}", f'{overlap}'))
            st.dataframe(pd.DataFrame(spans)[['order','start','end','overlap_before']],hide_index=True,width='stretch')
            if st.button(tr('创建长文本检查任务'),disabled=not model or dirty,key='long_create'):
                job=create_job(project,pid,iid,model,overlap);st.session_state['long_selected_job']=job['job_id'];st.rerun()
        except ValueError as exc:st.warning(diagnostic(exc))
        jobs=project.get('long_jobs',[])
        if not jobs:st.info(tr('尚无长文本检查任务；创建任务不会发送任何原文。'));return
        lookup={j['job_id']:j for j in jobs}
        if st.session_state.get('long_selected_job') not in lookup:st.session_state.pop('long_selected_job',None)
        selected=st.selectbox(tr('查看长文本任务'),list(lookup),format_func=lambda x:f"{lookup[x]['created_at']} · {lookup[x]['model']} · {x[-8:]}",key='long_selected_job')
        job=lookup[selected];current=is_current(project,job);cov=coverage(job)
        st.caption(tr('政策 ID：{0}；指标 ID：{1}；指标版本：{2}；文本版本：{3}。', f"{job['policy_id']}", f"{job['indicator_id']}", f"{job['version_id']}", f"{job['revision_id']}"))
        if not current:st.warning(tr('政策文本、指标版本或草稿已变化；旧任务仅供查看，不能继续处理或审核。'))
        st.write(tr('已有返回覆盖 {0} / {1} 个原文字符；返回范围：{2}', f"{cov['returned_characters']}", f"{cov['total_characters']}", f"{cov['returned_ranges']}"))
        if cov['all_returned']:st.info(tr('全部分段已发送并返回；这不证明模型理解或完整处理了全部内容，证据是否支持指标仍待研究者判断。'))
        else:st.warning(tr('总体状态：待核查。部分分段未返回；不能声称完整检查或审核分段建议，也不能因未发现证据自动赋 0。'))
        st.dataframe(pd.DataFrame(job['segments']),hide_index=True,width='stretch')
        st.caption(tr('可逐段处理或每批处理最多五段；批次内不可中断，关闭页面前请下载项目。失败或取消的段可单独重试，已返回的段不会重复生成。'))
        pending=[s for s in job['segments'] if s['status']=='pending']
        amount=int(st.number_input(tr('本批最多处理段数'),min_value=1,max_value=5,value=1,key='long_batch_amount'))
        if st.button(tr('处理下一批待处理分段'),disabled=not current or not pending,key='long_batch'):
            try:
                for segment in pending[:amount]:
                    item=process_segment(project,job,segment['order'],address)
                    if item['result']=='failed':break
                st.rerun()
            except ValueError as exc:st.error(diagnostic(exc))
        order=st.selectbox(tr('选择处理或重试的分段'),[s['order'] for s in job['segments']],
                           format_func=localize_options(lambda x:tr('第 {0} 段', f'{x}')),key='long_segment_select')
        selected_segment=job['segments'][order-1]
        with st.expander(tr('查看所选分段的原文和每次返回')):
            historical_text=revision(get_policy(project,job['policy_id']),job['revision_id'])['text']
            st.caption(tr('第 {0} 段：[{1}, {2})；状态：{3}。', f'{order}', f"{selected_segment['start']}", f"{selected_segment['end']}", f"{selected_segment['status']}"))
            source_text(historical_text[selected_segment['start']:selected_segment['end']])
            attempt_lookup={item['suggestion_id']:item for item in project.get('ai_suggestions',[])}
            if not selected_segment['attempt_ids']:st.info(tr('此分段尚无模型请求记录。'))
            for attempt_id in selected_segment['attempt_ids']:
                attempt=attempt_lookup[attempt_id]
                st.json(attempt,expanded=False)
        if st.button(tr('处理或重试所选分段'),disabled=not current or selected_segment['status']=='returned',key='long_process_selected'):
            try:process_segment(project,job,order,address);st.rerun()
            except ValueError as exc:st.error(diagnostic(exc))
        if st.button(tr('取消尚未完成的分段'),disabled=not current,key='long_cancel'):
            try:cancel_remaining(project,job);st.rerun()
            except ValueError as exc:st.error(diagnostic(exc))
        st.json(job['settings'])
        found,unresolved=candidates(project,job)
        st.write(tr('候选证据按完整原文位置去重；相同文字在不同位置分别保留。'))
        if unresolved:st.warning(tr('部分分段没有可定位证据或达到匹配上限，请查看下表。'));st.dataframe(unresolved,hide_index=True,width='stretch')
        if not found:st.info(tr('总体状态：待核查。尚无可定位的候选证据；不能推断政策不符合。'));return
        st.dataframe(pd.DataFrame(found),hide_index=True,width='stretch')
        idx=st.selectbox(tr('选择候选证据'),list(range(len(found))),format_func=lambda x:f"[{found[x]['start']}, {found[x]['end']}) · {found[x]['evidence'][:60]}",key='long_candidate')
        candidate=found[idx];historic=revision(get_policy(project,job['policy_id']),job['revision_id'])['text']
        source_text(historic[max(0,candidate['start']-60):min(len(historic),candidate['end']+60)])
        suggestions={s['suggestion_id']:s for s in project['ai_suggestions']}
        source_key=f"long_candidate_source_{job['job_id']}_{candidate['start']}_{candidate['end']}"
        sid=st.selectbox(tr('选择产生该证据的建议'),candidate['suggestion_ids'],format_func=lambda x:x[-8:],key=source_key)
        item=suggestions[sid];existing=latest_encoding(project,item)
        st.write(tr('当前正式编码：{0}', f"{existing['value']} · {existing['evidence']}" if existing else tr('尚无')))
        st.caption(tr('该建议模型原始状态：{0}；待审核状态：{1}；来源段：{2}。', f"{item['model_status']}", f"{item['suggested_status']}", f"{candidate['segment_orders']}"))
        if not current or status(project,item)!='待审核':return
        scope=f"{job['job_id']}_{candidate['start']}_{candidate['end']}_{sid}"
        action=st.radio(tr('候选证据审核方式'),['接受建议','修改后写入','拒绝建议'], format_func=localize_options(),horizontal=True,key='long_review_action_'+scope)
        with st.form('long_review_form_'+scope):
            values=['yes','no','pending','not_applicable'];labels={'yes':'符合','no':'不符合','pending':'待核查','not_applicable':'不适用'}
            value=st.selectbox(tr('人工最终状态'),values,index=values.index(item['suggested_status']),format_func=localize_options(lambda x:tr(labels[x])),key='long_final_value_'+scope)
            evidence=st.text_area(tr('人工核对证据'),value=candidate['evidence'],key='long_final_evidence_'+scope)
            a,b=st.columns(2)
            start=int(a.number_input(tr('完整原文证据起点'),min_value=0,value=candidate['start'],step=1,key='long_final_start_'+scope))
            end=int(b.number_input(tr('完整原文证据终点'),min_value=0,value=candidate['end'],step=1,key='long_final_end_'+scope))
            rationale=st.text_area(tr('人工审核依据'),value=item['rationale'],key='long_final_reason_'+scope)
            coder=st.text_input(tr('长文本审核人（可留空）'),key='long_reviewer_'+scope)
            replace=st.checkbox(tr('明确允许替换已有正式编码'),key='long_replace_'+scope)
            submit=st.form_submit_button(tr('提交候选证据审核'), key='long_review_submit_'+scope)
        if submit:
            try:
                op={'接受建议':'accept','修改后写入':'modify','拒绝建议':'reject'}[action]
                review(project,sid,op,value,evidence,start if evidence else None,end if evidence else None,rationale,coder,replace)
                st.rerun()
            except ValueError as exc:st.error(diagnostic(exc))
