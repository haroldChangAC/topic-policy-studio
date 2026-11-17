"""Researcher annotations for non-destructive grouping of original topics."""
import copy
import re
import unicodedata
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from studio.topic_names import materialize_topics


def empty():return {'schema_version':1,'groups':[],'history':[]}

def _valid_name(value):
    if not isinstance(value,str) or any(unicodedata.category(c).startswith('C') for c in value):raise ValueError('组名不能包含换行或控制字符。')
    value=value.strip()
    if not value or len(value)>100:raise ValueError('组名不能为空，最多 100 个字符。')
    return value


def _apply(result,obj,operation,group_id,name=None,members=None,timestamp=None):
    existing=next((x for x in obj['groups'] if x['group_id']==group_id),None)
    if operation=='create' and existing:raise ValueError('组编号重复。')
    if operation in ('update','delete') and not existing:raise ValueError('主题组不存在。')
    if operation not in ('create','update','delete') or not isinstance(group_id,str) or not re.fullmatch(r'group-[0-9a-f]{32}',group_id):raise ValueError('主题组操作或编号无效。')
    before=copy.deepcopy(existing)
    after=None
    if operation!='delete':
        name=_valid_name(name)
        if not isinstance(members,list) or any(isinstance(x,bool) or not isinstance(x,int) for x in members):raise ValueError('主题成员无效。')
        members=sorted(members)
        if len(members)<2 or len(set(members))!=len(members):raise ValueError('一个主题组须包含至少两个不同的原始主题。')
        if -1 in members:raise ValueError('离群类 -1 不参与主题归组。')
        valid=set(result['topics']['Topic'].astype(int))
        if not set(members).issubset(valid):raise ValueError('主题组包含不存在的原始主题。')
        for item in obj['groups']:
            if item['group_id']!=group_id and set(item['members']) & set(members):raise ValueError('原始主题已属于其他组，不能重复归组。')
        after={'group_id':group_id,'name':name,'members':members}
        if before==after:return False
    if existing:obj['groups'].remove(existing)
    if after:obj['groups'].append(after)
    obj['groups'].sort(key=lambda x:x['group_id'])
    obj['history'].append({'operation':operation,'group_id':group_id,'before':before,'after':copy.deepcopy(after),
                           'timestamp':timestamp or datetime.now(ZoneInfo('Asia/Seoul')).isoformat()})
    return True


def create(result,name,members):
    obj=result.setdefault('visualization',{}).setdefault('topic_groups',empty())
    gid='group-'+uuid.uuid4().hex;_apply(result,obj,'create',gid,name,members);return gid


def update(result,gid,name,members):return _apply(result,result['visualization']['topic_groups'],'update',gid,name,members)

def delete(result,gid):return _apply(result,result['visualization']['topic_groups'],'delete',gid)


def validate(result,obj):
    if obj is None:return
    if not isinstance(obj,dict) or set(obj)!={'schema_version','groups','history'} or obj['schema_version']!=1 or not isinstance(obj['groups'],list) or not isinstance(obj['history'],list):raise ValueError('主题归组记录格式无效。')
    replay=empty()
    for event in obj['history']:
        if not isinstance(event,dict) or set(event)!={'operation','group_id','before','after','timestamp'} or not isinstance(event['timestamp'],str):raise ValueError('主题归组历史记录格式无效。')
        try:datetime.fromisoformat(event['timestamp'])
        except ValueError as exc:raise ValueError('主题归组时间无效。') from exc
        after=event['after']
        if after is not None and (not isinstance(after,dict) or set(after)!={'group_id','name','members'} or after['group_id']!=event['group_id']):raise ValueError('主题归组变更格式无效。')
        try:
            changed=_apply(result,replay,event['operation'],event['group_id'],after['name'] if after else None,after['members'] if after else None,event['timestamp'])
        except (KeyError,TypeError) as exc:raise ValueError('主题归组历史记录无效。') from exc
        if not changed or replay['history'][-1]!=event:raise ValueError('主题归组历史前后不一致。')
    if replay['groups']!=obj['groups']:raise ValueError('主题归组方案与历史不一致。')


def mapping(result):
    topics=materialize_topics(result);obj=(result.get('visualization') or {}).get('topic_groups') or empty()
    owner={topic:group for group in obj['groups'] for topic in group['members']};records=[]
    for row in topics.itertuples():
        topic=int(row.Topic);group=owner.get(topic)
        records.append({'original_topic_id':topic,'original_topic_name':row.Display_Name,'original_keywords':row.Keywords,
                        'group_id':group['group_id'] if group else '', 'group_name':group['name'] if group else '',
                        'view_id':group['group_id'] if group else f'topic:{topic}',
                        'view_name':group['name'] if group else row.Display_Name,
                        'is_outlier':topic==-1,'keyword_note':'原始主题关键词，未重新计算'})
    return pd.DataFrame(records)


def trace(result):
    f=result['metadata']['generated_fields'];segments=result['segments'];m=mapping(result).set_index('original_topic_id');records=[]
    for _,s in segments.iterrows():
        topic=int(s[f['topic_id']]);mrow=m.loc[topic]
        record={'view_id':mrow['view_id'],'view_name':mrow['view_name'],'group_id':mrow['group_id'],
                'original_topic_id':topic,'original_topic_name':mrow['original_topic_name'],
                'document_id':s[f['document_id']],'segment_id':s[f['segment_id']],
                'segment_order':s[f['segment_order']],'start_char':s[f['start_char']],
                'end_char':s[f['end_char']],'segment_text':s[f['segment_text']]}
        for col in result['metadata']['original_columns']:record['raw::'+col]=s[col]
        records.append(record)
    return pd.DataFrame(records)


def summary(result):
    f=result['metadata']['generated_fields'];segments=result['segments'];m=mapping(result).set_index('original_topic_id')
    assoc=segments[[f['document_id'],f['topic_id']]].copy()
    assoc['view_id']=assoc[f['topic_id']].map(m['view_id'])
    total=len(assoc);rows=[]
    for view_id,frame in assoc.groupby('view_id',sort=True):
        topic_ids=sorted(int(x) for x in frame[f['topic_id']].unique());entry=m.loc[topic_ids[0]]
        rows.append({'view_id':view_id,'view_name':entry['view_name'],'original_topic_ids':','.join(map(str,topic_ids)),
                     'is_group':bool(entry['group_id']),'is_outlier':topic_ids==[-1],
                     'segment_count':len(frame),'segment_denominator':total,'segment_share':len(frame)/total if total else None,
                     'document_count':frame[f['document_id']].nunique(),'keyword_note':'原始主题关键词，未重新计算'})
    return pd.DataFrame(rows)


def annual(result,settings):
    """Use original annual filters/denominators; count documents from source segments, never sum topic document counts."""
    from studio.annual import calculate,MAX_CELLS
    docs,panel,parts=calculate(result,settings);f=result['metadata']['generated_fields'];mapping_frame=mapping(result)
    if not settings['include_outliers']:mapping_frame=mapping_frame[~mapping_frame.is_outlier]
    categories=mapping_frame[['view_id','view_name','is_outlier']].drop_duplicates('view_id')
    base=panel[['group_key','group','group_missing','year','segment_denominator','document_denominator']].drop_duplicates()
    if len(base)*len(categories)>MAX_CELLS:raise ValueError('归组年度面板超过行数上限。')
    topic_map=mapping_frame.set_index('original_topic_id')['view_id'].to_dict();byid=docs.set_index('document_id');counts={};docsets={}
    for _,segment in parts.iterrows():
        doc=byid.loc[segment[f['document_id']]];year=None if doc['year_missing'] else int(doc['year']);vid=topic_map[int(segment[f['topic_id']])];key=(doc['group_key'],year,vid)
        counts[key]=counts.get(key,0)+1;docsets.setdefault(key,set()).add(segment[f['document_id']])
    rows=[]
    for _,cell in base.iterrows():
        year=None if pd.isna(cell['year']) else int(cell['year']);sd=int(cell['segment_denominator']);dd=int(cell['document_denominator'])
        for cat in categories.itertuples(index=False):
            key=(cell['group_key'],year,cat.view_id);sc=counts.get(key,0);dc=len(docsets.get(key,set()))
            rows.append({'group':cell['group'],'group_missing':bool(cell['group_missing']),'group_key':cell['group_key'],'year':year,
                         'view_id':cat.view_id,'view_name':cat.view_name,'is_outlier':bool(cat.is_outlier),
                         'segment_count':sc,'segment_denominator':sd,'segment_share':sc/sd if sd else None,
                         'document_count':dc,'document_denominator':dd,'document_mention_rate':dc/dd if dd else None})
    return docs,pd.DataFrame(rows),parts
