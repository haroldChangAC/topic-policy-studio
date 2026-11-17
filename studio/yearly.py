"""Independent per-year topic fits against already checked source segments."""
import importlib.metadata
import json
import math
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from studio.annual import parse_year
from studio.modeling import default_clustering_settings, fit_topics, model_config
from studio.similarity import VECTOR_TYPE


def preview(result, field):
    if field not in result['metadata'].get('original_columns', []):
        raise ValueError('所选年份字段不是已保存的原始字段。')
    f = result['metadata']['generated_fields']
    counts = Counter(result['segments'][f['document_id']].tolist())
    return pd.DataFrame([{'document_id': row[f['document_id']], 'year': parse_year(row[field]),
                          'source_value': str(row[field]), 'status': row[f['status']],
                          'segment_count': counts[row[f['document_id']]]}
                         for _, row in result['rows'].iterrows()])


def run(result, field, years, encoder, device='cpu', progress=lambda text: None, fit=None):
    fit = fit or fit_topics
    if not years or any(isinstance(y, bool) or not isinstance(y, int) for y in years) or len(set(years)) != len(years):
        raise ValueError('请选择不重复的有效年份。')
    docs = preview(result, field); f = result['metadata']['generated_fields']; meta = result['metadata']
    eligible = docs[docs['status'].eq('modeled')]
    available = {int(x) for x in eligible['year'].dropna()}
    if not set(years).issubset(available):
        raise ValueError('所选年份没有可用的已建模片段。')
    clustering = meta.get('clustering_settings_requested') or default_clustering_settings()
    keyword = meta.get('keyword_processing') or {'mode':'legacy','stopwords_enabled':False,'stopwords':[],'user_dictionary':[]}
    config = model_config()
    versions = {n:importlib.metadata.version(n) for n in ('bertopic','torch','sentence-transformers','umap-learn','hdbscan','scikit-learn','numpy','jieba','transformers','tokenizers')}
    obj = {'schema_version':1, 'source_run_id':meta['run_id'], 'year_field':field,
           'selected_years':sorted(years), 'missing_year_rule':'年份无法明确解析时排除；不推测年份',
           'missing_document_ids':docs.loc[docs.year.isna(), 'document_id'].tolist(),
           'source_segment_rule':'复用原分析中有效且已建模的无重叠原文片段；原分析的空值和重复处理保留',
           'segmentation':meta.get('segmentation'), 'device':device, 'model_id':config['model_id'],
           'model_revision':config['revision'], 'clustering_requested':clustering, 'keyword_processing':keyword,
           'settings_note':'历史未保存的参数明确使用当前默认设置' if not meta.get('clustering_settings_requested') or not meta.get('keyword_processing') else '使用原项目已保存的聚类和关键词设置',
           'versions':versions, 'created_at':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
           'vector_type':VECTOR_TYPE, 'results':[]}
    for year in sorted(years):
        ids=set(eligible.loc[eligible.year.eq(year),'document_id'])
        segs=result['segments'][result['segments'][f['document_id']].isin(ids)]
        texts=segs[f['segment_text']].tolist()
        item={'year':year,'documents':len(ids),'segments':len(segs),'status':'failed','error':None,
              'topics':[],'assignments':[],'topic_vectors':[],'keyword_weights':[], 'actual_parameters':None,
              'timings_seconds':None, 'outliers':None, 'non_outlier_topics':None,
              'vector_dimension':None, 'model_token_limit':None, 'max_input_tokens':None}
        try:
            progress(f'正在独立分析 {year} 年：{len(ids)} 篇文档、{len(segs)} 个片段。')
            output=fit(texts,encoder,keyword_settings=keyword,clustering_settings=clustering)
            if len(output['topics']) != len(segs):raise ValueError('主题分配与片段数不一致。')
            topics=[int(t) for t in output['topics']]
            strengths=output['strengths']
            item.update(status='completed',topics=[{'topic_id':int(x.Topic),'count':int(x.Count),'label':str(x.Name),'keywords':str(x.Keywords)} for x in output['topic_info'].itertuples()],
                assignments=[{'segment_id':sid,'topic_id':t,'membership_strength':None if strengths is None else float(strengths[i])} for i,(sid,t) in enumerate(zip(segs[f['segment_id']],topics))],
                topic_vectors=output['topic_vectors'],keyword_weights=output['keyword_weights'],actual_parameters=output['parameters'],
                timings_seconds=output['timings_seconds'],outliers=topics.count(-1),non_outlier_topics=len(set(topics)-{-1}),
                vector_dimension=output['topic_vector_dimension'],model_token_limit=output['model_token_limit'],max_input_tokens=output['max_input_tokens'])
        except Exception as exc:
            item['error']=f'{type(exc).__name__}: {exc}'
        obj['results'].append(item)
    validate(result,obj)
    return obj


def validate(result, obj):
    if not isinstance(obj,dict) or obj.get('schema_version') != 1 or obj.get('source_run_id') != result['metadata'].get('run_id'):
        raise ValueError('年度独立建模记录的版本或来源无效。')
    if obj.get('vector_type') != VECTOR_TYPE or obj.get('year_field') not in result['metadata']['original_columns']:
        raise ValueError('年度模型向量口径或年份字段无效。')
    years=obj.get('selected_years');runs=obj.get('results')
    if not isinstance(years,list) or not years or any(isinstance(x,bool) or not isinstance(x,int) for x in years) or years!=sorted(set(years)) or not isinstance(runs,list) or [x.get('year') for x in runs]!=years:
        raise ValueError('年度建模年份记录无效。')
    docs=preview(result,obj['year_field']);f=result['metadata']['generated_fields'];segs=result['segments']
    if obj.get('missing_document_ids')!=docs.loc[docs.year.isna(),'document_id'].tolist():raise ValueError('年度缺失年份文档记录不一致。')
    for run in runs:
        ids=set(docs.loc[docs['year'].eq(run['year']) & docs.status.eq('modeled'),'document_id'])
        relevant=segs[segs[f['document_id']].isin(ids)];valid=set(relevant[f['segment_id']])
        if run.get('documents')!=len(ids) or run.get('segments')!=len(valid) or run.get('status') not in ('completed','failed'):
            raise ValueError('年度样本数量或状态无效。')
        assignments=run.get('assignments');topics=run.get('topics');vectors=run.get('topic_vectors')
        if not isinstance(assignments,list) or not isinstance(topics,list) or not isinstance(vectors,list):raise ValueError('年度结果结构无效。')
        if run['status']=='failed':
            if assignments or topics or vectors or not isinstance(run.get('error'),str):raise ValueError('失败年份不能包含主题结果。')
            continue
        if len(assignments)!=len(valid) or {a.get('segment_id') for a in assignments}!=valid:raise ValueError('年度片段引用缺失或重复。')
        ids2=[t.get('topic_id') for t in topics]
        if any(isinstance(x,bool) or not isinstance(x,int) for x in ids2) or len(ids2)!=len(set(ids2)):raise ValueError('年度主题编号无效。')
        counts=Counter(a.get('topic_id') for a in assignments)
        if set(counts)!=set(ids2) or any(t.get('count')!=counts[t['topic_id']] for t in topics):raise ValueError('年度主题统计与分配不一致。')
        if run.get('outliers')!=counts.get(-1,0) or run.get('non_outlier_topics')!=len(set(ids2)-{-1}):raise ValueError('年度离群统计不一致。')
        dim=run.get('vector_dimension');seen=set()
        if isinstance(dim,bool) or not isinstance(dim,int) or not 1<=dim<=10000:raise ValueError('年度主题向量维度无效。')
        for v in vectors:
            tid=v.get('topic_id');values=v.get('values')
            if tid not in counts or tid in seen:raise ValueError('年度向量编号映射无效。')
            seen.add(tid)
            if values is not None:
                if not isinstance(values,list) or not values or len(values)>10000 or any(isinstance(x,bool) or not isinstance(x,(float,int)) or not math.isfinite(x) for x in values):raise ValueError('年度向量无效。')
                if len(values)!=dim:raise ValueError('年度向量维度不一致。')
        if seen!=set(ids2):raise ValueError('年度向量编号映射不完整。')
        for a in assignments:
            val=a.get('membership_strength')
            if val is not None and (isinstance(val,bool) or not isinstance(val,(int,float)) or not math.isfinite(val)):raise ValueError('年度片段归属强度无效。')


def tables(result):
    obj=result['yearly_analysis'];f=result['metadata']['generated_fields'];summary=[];assignments=[];vectors=[]
    for run in obj['results']:
        for t in run['topics']:
            summary.append({'year':run['year'],**t,'is_outlier':t['topic_id']==-1})
        for a in run['assignments']:assignments.append({'year':run['year'],**a})
        for v in run['topic_vectors']:vectors.append({'year':run['year'],'topic_id':v['topic_id'],'vector_type':obj['vector_type'],'values_json':json.dumps(v['values'],ensure_ascii=False)})
    ass=pd.DataFrame(assignments,columns=['year','segment_id','topic_id','membership_strength'])
    cols=[f[k] for k in ('segment_id','document_id','segment_order','start_char','end_char','segment_text')]
    if not ass.empty:
        ass=ass.merge(result['segments'][cols],left_on='segment_id',right_on=f['segment_id'],validate='one_to_one',how='left')
    return pd.DataFrame(summary,columns=['year','topic_id','count','label','keywords','is_outlier']),ass,pd.DataFrame(vectors,columns=['year','topic_id','vector_type','values_json'])
