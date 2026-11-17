"""Traceable adjacent-natural-year topic cosine relations; no topic matching."""
import hashlib
import json
import math
from collections import defaultdict

import numpy as np
import pandas as pd
from studio.similarity import VECTOR_TYPE
from studio.yearly import validate as validate_yearly

METHOD = '余弦相似度：主题向量点积除以两向量长度之积'
ADJACENCY = '仅相邻自然年份 y 与 y+1；缺年、失败或未选择时不跨越'
WIDTH = '相似度权重 = 正的余弦相似度；不是文档数、片段数或实际转移量'
MAX_NODES = 60
MAX_EDGES = 150


def source_digest(result):
    obj=result.get('yearly_analysis')
    if obj is None:raise ValueError('项目未保存按年份分别建模结果。')
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def inventory(result):
    obj=result.get('yearly_analysis');validate_yearly(result,obj)
    f=result['metadata']['generated_fields']
    lookup=result['segments'].set_index(f['segment_id'])[f['document_id']].to_dict()
    nodes=[]
    for run in obj['results']:
        if run['status']!='completed':continue
        docs=defaultdict(set)
        for a in run['assignments']:docs[a['topic_id']].add(lookup[a['segment_id']])
        vectors={v['topic_id']:v['values'] for v in run['topic_vectors']}
        for t in run['topics']:
            nodes.append({'year':run['year'],'topic_id':t['topic_id'],'label':t['label'],
                          'keywords':t['keywords'],'segment_count':t['count'],
                          'document_count':len(docs[t['topic_id']]),'vector':vectors.get(t['topic_id']),
                          'dimension':run['vector_dimension'],
                          'model_id':run.get('model_id',obj.get('model_id')),
                          'model_revision':run.get('model_revision',obj.get('model_revision')),
                          'vector_type':run.get('vector_type',obj.get('vector_type'))})
    return nodes


def defaults(result):
    nodes=inventory(result);years=sorted({n['year'] for n in nodes})
    chosen=[]
    for year in years:
        group=sorted((n for n in nodes if n['year']==year and n['topic_id']!=-1),key=lambda n:(-n['segment_count'],n['topic_id']))[:5]
        chosen.extend([[year,n['topic_id']] for n in group])
    return {'schema_version':1,'source_sha256':source_digest(result),'years':years,'topics':chosen[:MAX_NODES],
            'threshold':0.7,'max_edges':50,'style':'color'}


def validate_settings(result, settings, allow_stale=False):
    if not isinstance(settings,dict) or set(settings)!={'schema_version','source_sha256','years','topics','threshold','max_edges','style'} or settings['schema_version']!=1:
        raise ValueError('跨年关联设置格式无效。')
    if settings['source_sha256']!=source_digest(result) and not allow_stale:
        raise ValueError('按年建模数据已变化，跨年关联设置已过期。')
    years=settings['years'];topics=settings['topics'];threshold=settings['threshold'];max_edges=settings['max_edges']
    nodes=inventory(result);valid={(n['year'],n['topic_id']) for n in nodes if n['topic_id']!=-1};available={n['year'] for n in nodes}
    if not isinstance(years,list) or any(isinstance(x,bool) or not isinstance(x,int) for x in years) or len(set(years))!=len(years) or not set(years)<=available:
        raise ValueError('跨年关联年份无效。')
    if not isinstance(topics,list) or len(topics)>MAX_NODES or any(not isinstance(x,list) or len(x)!=2 or any(isinstance(v,bool) or not isinstance(v,int) for v in x) or tuple(x) not in valid or x[0] not in years for x in topics) or len({tuple(x) for x in topics})!=len(topics):
        raise ValueError('跨年关联所选主题无效或超过 60 个；离群类不可选。')
    if isinstance(threshold,bool) or not isinstance(threshold,(float,int)) or not math.isfinite(threshold) or not 0<=threshold<=1:
        raise ValueError('相似度门槛须在 0 到 1 之间。')
    if isinstance(max_edges,bool) or not isinstance(max_edges,int) or not 1<=max_edges<=MAX_EDGES or settings['style'] not in ('color','mono'):
        raise ValueError('跨年关联显示设置无效。')


def calculate(result, settings):
    validate_settings(result,settings)
    nodes=inventory(result);chosen={tuple(x) for x in settings['topics']};nodes=[n for n in nodes if (n['year'],n['topic_id']) in chosen]
    exclusions=[];valid=[]
    for n in nodes:
        vec=n['vector'];reason=None
        if not isinstance(n['model_id'],str) or not n['model_id'].strip() or not isinstance(n['model_revision'],str) or not n['model_revision'].strip():reason='缺少模型名称或修订版本'
        elif n['vector_type']!=VECTOR_TYPE:reason='主题向量来源不同或不明'
        elif vec is None:reason='缺少主题向量'
        elif len(vec)!=n['dimension']:reason='主题向量维度与记录不一致'
        else:
            arr=np.asarray(vec,dtype=float);scale=float(np.max(np.abs(arr)))
            if not np.isfinite(scale) or scale==0:reason='零向量或无效向量'
            else:
                scaled=arr/scale;norm=np.linalg.norm(scaled)
                if not np.isfinite(norm) or norm==0:reason='向量无法归一化'
                else:n={**n,'unit':scaled/norm};valid.append(n)
        if reason:exclusions.append({'year':n['year'],'topic_id':n['topic_id'],'reason':reason})
    pairs=[];warnings=[];years=sorted(settings['years'])
    for y in years:
        if y+1 not in years:
            if any(z>y+1 for z in years):warnings.append(f'{y} 年与后续所选年份不相邻；未跨年连接。')
            continue
        left=[n for n in valid if n['year']==y];right=[n for n in valid if n['year']==y+1]
        for a in left:
            for b in right:
                if (a['model_id'],a['model_revision'],a['vector_type'],a['dimension'])!=(b['model_id'],b['model_revision'],b['vector_type'],b['dimension']):
                    warnings.append(f"{y} 主题 {a['topic_id']} 与 {y+1} 主题 {b['topic_id']} 的模型、向量口径或维度不一致，已排除。")
                    continue
                sim=float(np.clip(np.dot(a['unit'],b['unit']),-1,1))
                pairs.append({'source_year':y,'source_topic_id':a['topic_id'],'source_label':a['label'],
                 'source_keywords':a['keywords'],'source_segments':a['segment_count'],'source_documents':a['document_count'],
                 'target_year':y+1,'target_topic_id':b['topic_id'],'target_label':b['label'],
                 'target_keywords':b['keywords'],'target_segments':b['segment_count'],'target_documents':b['document_count'],
                 'similarity':sim,'threshold':float(settings['threshold']),'max_edges':settings['max_edges'],
                 'passes_threshold':sim>=settings['threshold'],'in_graph':False,'line_weight':None,
                 'vector_type':a['vector_type'],'model_id':a['model_id'],'model_revision':a['model_revision'],
                 'method':METHOD,'year_rule':ADJACENCY,'line_rule':WIDTH})
    cols=['source_year','source_topic_id','source_label','source_keywords','source_segments','source_documents','target_year','target_topic_id','target_label','target_keywords','target_segments','target_documents','similarity','threshold','max_edges','passes_threshold','in_graph','line_weight','vector_type','model_id','model_revision','method','year_rule','line_rule']
    frame=pd.DataFrame(pairs,columns=cols)
    if not frame.empty:
        eligible=frame[(frame.similarity>=settings['threshold']) & (frame.similarity>0)].sort_values(['similarity','source_year','source_topic_id','target_topic_id'],ascending=[False,True,True,True]).head(settings['max_edges'])
        frame.loc[eligible.index,'in_graph']=True;frame.loc[eligible.index,'line_weight']=frame.loc[eligible.index,'similarity']
    return nodes,frame,exclusions,list(dict.fromkeys(warnings))


def evidence(result,year,topic):
    f=result['metadata']['generated_fields'];run=next(x for x in result['yearly_analysis']['results'] if x['year']==year and x['status']=='completed')
    ids={a['segment_id'] for a in run['assignments'] if a['topic_id']==topic}
    cols=[f[x] for x in ('document_id','segment_id','segment_order','start_char','end_char','segment_text')]
    return result['segments'].loc[result['segments'][f['segment_id']].isin(ids),cols]


def graph(nodes, frame, settings, *, translate=lambda text: text):
    import html
    import plotly.graph_objects as go
    drawn=frame[frame.in_graph].copy()
    if drawn.empty:return None
    keys=set(zip(drawn.source_year,drawn.source_topic_id))|set(zip(drawn.target_year,drawn.target_topic_id))
    active=sorted((n for n in nodes if (n['year'],n['topic_id']) in keys),key=lambda n:(n['year'],n['topic_id']))
    mapping={(n['year'],n['topic_id']):i for i,n in enumerate(active)}
    node_label = translate('{0} · 主题 {1} · {2} ({3} 片段)')
    labels = [html.escape(node_label.format(n['year'], n['topic_id'], n['label'], n['segment_count'])) for n in active]
    colors=('#a7b5b9','rgba(90,100,105,0.28)') if settings['style']=='mono' else ('#8cb8b8','rgba(99,151,164,0.30)')
    fig=go.Figure(go.Sankey(arrangement='snap',node={'label':labels,'pad':16,'thickness':12,'color':colors[0],
                       'line':{'color':'#6c797b','width':0.4}},
        link={'source':[mapping[(r.source_year,r.source_topic_id)] for r in drawn.itertuples()],
              'target':[mapping[(r.target_year,r.target_topic_id)] for r in drawn.itertuples()],
              'value':drawn.line_weight.astype(float).tolist(),'color':colors[1],
              'hovertemplate':translate('相似度权重：')+'%{value:.5f}<extra></extra>'}))
    fig.update_layout(title=translate('相邻自然年份主题相似关联（非转移）'),font={'size':11,'family':'Arial, sans-serif'},
                      paper_bgcolor='white',plot_bgcolor='white',height=max(500,min(1200,45*max(sum(n['year']==y for n in active) for y in settings['years']))),
                      margin={'l':20,'r':20,'t':65,'b':30})
    return fig
