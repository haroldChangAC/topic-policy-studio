"""Resumable, position-preserving long-policy model inspection jobs."""
import copy
import hashlib

from studio.policy import active_version, get_policy, new_id, now, revision
from studio.policy_assist import _target, _valid_model, generate, make_prompt
from studio.segmentation import BOUNDARIES

CONTEXT=4096
OUTPUT_RESERVE=512
TEMPLATE_RESERVE=512
PROMPT_BYTE_BUDGET=CONTEXT-OUTPUT_RESERVE-TEMPLATE_RESERVE
PARAMETERS={'temperature':0,'num_ctx':CONTEXT,'num_predict':OUTPUT_RESERVE}
RULE='paragraph-sentence-utf8-overlap-v1'
MAX_SEGMENTS=500


def settings_for(indicator,text,overlap):
    if isinstance(overlap,bool) or not isinstance(overlap,int) or not 0<=overlap<=128:
        raise ValueError('重叠字符数应为 0–128。')
    # Positions with the greatest decimal width conservatively bound the template.
    overhead=len(make_prompt(indicator,text,len(text),len(text)).encode('utf-8'))
    body=PROMPT_BYTE_BUDGET-overhead
    if body<128 or overlap*4>body//2:
        raise ValueError('指标或提示过长，无法保留安全估算余量；请缩短指标说明或重叠。')
    return {'rule':RULE,'context_requested':CONTEXT,'output_tokens_reserved':OUTPUT_RESERVE,
            'template_tokens_reserved_estimate':TEMPLATE_RESERVE,'prompt_utf8_byte_budget':PROMPT_BYTE_BUDGET,
            'prompt_fixed_utf8_bytes':overhead,'body_utf8_byte_budget':body,'overlap_characters':overlap,
            'estimation':'UTF-8 字节数作为保守估算，非 Qwen tokenizer 实测；服务端模板和实际 token 未验证',
            'request_parameters':copy.deepcopy(PARAMETERS)}


def split_ranges(text,indicator,overlap=40):
    if not isinstance(text,str) or not text.strip(): raise ValueError('政策正文为空。')
    cfg=settings_for(indicator,text,overlap);limit=cfg['body_utf8_byte_budget'];out=[];start=0;previous=0
    while start<len(text):
        lo=start;hi=min(len(text),start+min(limit,12000))
        while lo<hi:
            mid=(lo+hi+1)//2
            if len(text[start:mid].encode('utf-8'))<=limit:lo=mid
            else:hi=mid-1
        hard=lo
        if hard<=previous:raise ValueError('重叠过大，分段无法前进。')
        end=hard
        if hard<len(text):
            threshold=max(previous+1,start+(hard-start)//2)
            for pattern in BOUNDARIES:
                boundaries=[m.end() for m in pattern.finditer(text,start,hard) if m.end()>=threshold]
                if boundaries:
                    end=boundaries[-1];break
        if end<=previous:raise ValueError('分段无法向后推进。')
        out.append({'order':len(out)+1,'start':start,'end':end,'overlap_before':max(0,previous-start),
                    'status':'pending','attempt_ids':[]})
        if len(out)>MAX_SEGMENTS:raise ValueError(f'分段超过 {MAX_SEGMENTS} 段，请缩短文本或指标说明。')
        previous=end
        if end==len(text):break
        start=max(start+1,end-overlap)
    return out,cfg


def create_job(project,policy_id,indicator_id,model,overlap=40):
    policy,version,indicator,text=_target(project,policy_id,indicator_id);_valid_model(model)
    ranges,cfg=split_ranges(text,indicator,overlap)
    job={'job_id':new_id('longjob'),'policy_id':policy_id,'indicator_id':indicator_id,
         'version_id':version['version_id'],'revision_id':policy['current_revision'],
         'model':model,'created_at':now(),'text_sha256':hashlib.sha256(text.encode('utf-8')).hexdigest(),
         'settings':cfg,'segments':ranges}
    project.setdefault('ai_suggestions',[]);project.setdefault('ai_reviews',[])
    project.setdefault('long_jobs',[]).append(job)
    return job


def is_current(project,job):
    policy=get_policy(project,job['policy_id']);version=active_version(project)
    return bool(policy and version and version['version_id']==job['version_id']
                and policy['current_revision']==job['revision_id'] and version['indicators']==project['draft'])


def process_segment(project,job,order,base_url='http://127.0.0.1:11434',requester=None):
    if not is_current(project,job):raise ValueError('政策文本或指标版本已改变；旧任务不能继续，请创建新任务。')
    if job not in project.get('long_jobs',[]):raise ValueError('分段任务不存在。')
    segment=next((s for s in job['segments'] if s['order']==order),None)
    if segment is None:raise ValueError('分段不存在。')
    if segment['status']=='returned':raise ValueError('该段已有结果，不重复生成；需要新分析时请新建任务。')
    item=generate(project,job['policy_id'],job['indicator_id'],job['model'],segment['start'],segment['end'],
                  base_url,requester,generation_parameters=PARAMETERS)
    segment['attempt_ids'].append(item['suggestion_id'])
    segment['status']='returned' if item['result']=='ready' else 'failed'
    return item


def cancel_remaining(project,job):
    if job not in project.get('long_jobs',[]) or not is_current(project,job):raise ValueError('旧任务不能继续操作。')
    count=0
    for segment in job['segments']:
        if segment['status'] in ('pending','failed'):
            segment['status']='cancelled';count+=1
    return count


def coverage(job):
    ranges=sorted((s['start'],s['end']) for s in job['segments'] if s['status']=='returned')
    merged=[]
    for start,end in ranges:
        if merged and start<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],end)
        else:merged.append([start,end])
    total=job['segments'][-1]['end']
    return {'returned_ranges':merged,'returned_characters':sum(b-a for a,b in merged),
            'total_characters':total,'all_returned':all(s['status']=='returned' for s in job['segments'])}


def candidates(project,job):
    """Deduplicate identical original spans; retain distinct occurrences and source segments."""
    policy=get_policy(project,job['policy_id']);text=revision(policy,job['revision_id'])['text']
    suggestions={s['suggestion_id']:s for s in project.get('ai_suggestions',[])};found={};unresolved=[]
    for segment in job['segments']:
        if segment['status']!='returned':continue
        item=suggestions[segment['attempt_ids'][-1]];evidence=item['evidence']
        if not evidence:
            unresolved.append({'order':segment['order'],'suggestion_id':item['suggestion_id'],'reason':'模型未提供证据'});continue
        pos=segment['start'];count=0
        while True:
            pos=text.find(evidence,pos,segment['end'])
            if pos<0:break
            end=pos+len(evidence);key=(pos,end)
            if key not in found:
                found[key]={'start':pos,'end':end,'evidence':evidence,'segment_orders':[],
                            'suggestion_ids':[],'model_statuses':[],'match_statuses':[],'exact_match':'原文逐字匹配'}
            if segment['order'] not in found[key]['segment_orders']:found[key]['segment_orders'].append(segment['order'])
            if item['suggestion_id'] not in found[key]['suggestion_ids']:
                found[key]['suggestion_ids'].append(item['suggestion_id']);found[key]['model_statuses'].append(item['model_status']);found[key]['match_statuses'].append(item['match_status'])
            count+=1;pos+=1
            if count>=100:
                unresolved.append({'order':segment['order'],'suggestion_id':item['suggestion_id'],'reason':'本段匹配达到 100 处上限，结果可能不全'});break
        if not count:unresolved.append({'order':segment['order'],'suggestion_id':item['suggestion_id'],'reason':'证据在本次发送范围内未精确匹配'})
    return sorted(found.values(),key=lambda x:(x['start'],x['end'])),unresolved


def validate_long_jobs(project):
    jobs=project.get('long_jobs',[])
    if not isinstance(jobs,list):raise ValueError('长政策分段任务格式无效。')
    ids=set();used=set();versions={v['version_id']:v for v in project['versions']}
    suggestions={s['suggestion_id']:s for s in project.get('ai_suggestions',[])}
    for job in jobs:
        if (not isinstance(job,dict) or set(job)!={'job_id','policy_id','indicator_id','version_id','revision_id','model','created_at','text_sha256','settings','segments'}
            or any(not isinstance(job[k],str) or not job[k] for k in ('job_id','policy_id','indicator_id','version_id','revision_id','model','created_at','text_sha256'))
            or job['job_id'] in ids):raise ValueError('长政策任务标识或字段无效。')
        ids.add(job['job_id']);_valid_model(job['model']);policy=get_policy(project,job['policy_id']);version=versions.get(job['version_id'])
        indicator=next((i for i in version['indicators'] if i['id']==job['indicator_id'] and i['level']==2),None) if version else None
        if not policy or not indicator:raise ValueError('分段任务的政策或指标不存在。')
        text=revision(policy,job['revision_id'])['text']
        if job['text_sha256']!=hashlib.sha256(text.encode('utf-8')).hexdigest() or not isinstance(job['settings'],dict):raise ValueError('分段任务原文或设置无效。')
        expected,cfg=split_ranges(text,indicator,job['settings'].get('overlap_characters'))
        if job['settings']!=cfg or not isinstance(job['segments'],list) or len(job['segments'])!=len(expected):raise ValueError('分段规则或数量无效。')
        for segment,plan in zip(job['segments'],expected):
            if (not isinstance(segment,dict) or set(segment)!=set(plan) or any(segment[k]!=plan[k] for k in ('order','start','end','overlap_before'))
                or segment['status'] not in ('pending','returned','failed','cancelled') or not isinstance(segment['attempt_ids'],list)):
                raise ValueError('分段范围或状态无效。')
            attempts=[]
            for sid in segment['attempt_ids']:
                if not isinstance(sid,str) or sid in used or sid not in suggestions:raise ValueError('分段尝试记录无效。')
                used.add(sid);item=suggestions[sid];attempts.append(item)
                if (any(item[k]!=job[k] for k in ('policy_id','indicator_id','version_id','revision_id','model'))
                    or item['start']!=segment['start'] or item['end']!=segment['end'] or item['parameters']!=PARAMETERS):
                    raise ValueError('分段与模型建议来源不一致。')
            if (segment['status']=='pending' and attempts) or (segment['status']=='returned' and (not attempts or attempts[-1]['result']!='ready')) or (segment['status']=='failed' and not attempts):
                raise ValueError('分段状态与尝试记录不一致。')
            if any(x['result']!='failed' for x in (attempts[:-1] if segment['status']=='returned' else attempts)):
                raise ValueError('已返回的分段不能重复处理。')
        if not coverage(job)['all_returned']:
            job_attempts={sid for segment in job['segments'] for sid in segment['attempt_ids']}
            if any(r['suggestion_id'] in job_attempts for r in project.get('ai_reviews',[])):
                raise ValueError('未完成的分段任务不能写入审核编码。')
