"""Build synthetic projects and sample process RSS for independent stages.

This is a technical benchmark, not a real research corpus. psutil samples RSS
at 5 ms; short spikes can be missed. Logical dataframe deep size double-counts
shared Python string objects and is reported separately from process RSS.
"""
import argparse,gc,hashlib,json,os,platform,random,threading,time
from pathlib import Path
import psutil
import pandas as pd
from studio.project import open_project,save_project
from studio.visualization import default_visualization
from studio.annual import calculate
from studio.grouped_charts import source,evidence,for_scope,cells
from studio.annual_heatmap import ALL
from studio.topic_groups import create
from studio.workflow import csv_bytes,excel_bytes
from studio.policy_transfer import plan_transfer,apply_transfer

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'outputs/memory_validation';OUT.mkdir(parents=True,exist_ok=True)

def build(doc_count,text_chars,part_chars,metadata_chars):
    f={k:'studio_'+k for k in ('record_id','document_id','original_text','status','segment_id','segment_order','start_char','end_char','segment_text','topic_id')}
    docs=[];segments=[];counts={0:0,1:0,2:0,-1:0};alphabet='政策治理文化教育发展社会城市农村社区工作政府项目合作环境保障数据科技产业人口'
    for i in range(doc_count):
        rng=random.Random(i+4821);text=''.join(rng.choices(alphabet,k=text_chars-12))+f'{i:012d}'
        metadata='附加字段'+''.join(rng.choices(alphabet,k=metadata_chars));did=f'D{i:04}'
        row={'文档':did,'正文':text,'年份':str(2020+i%3),'地区':['甲','乙'][i%2],'备注':metadata,
             f['record_id']:did,f['document_id']:did,f['original_text']:text,f['status']:'modeled'}
        docs.append(row)
        for order,start in enumerate(range(0,len(text),part_chars),1):
            end=min(start+part_chars,len(text));topic=[0,1,2,-1][(i+order)%4];counts[topic]+=1
            segments.append({**row,f['segment_id']:f'{did}-S{order:05}',f['segment_order']:order,
                             f['start_char']:start,f['end_char']:end,f['segment_text']:text[start:end],f['topic_id']:topic})
    topics=pd.DataFrame([{'Topic':k,'Count':v,'Name':f'{k}_合成主题','Keywords':'合成、测试'} for k,v in counts.items() if v])
    result={'rows':pd.DataFrame(docs),'segments':pd.DataFrame(segments),'topics':topics,
            'metadata':{'generated_fields':f,'original_columns':['文档','正文','年份','地区','备注'],'counts':{'input':len(docs),'segments':len(segments)},
                        'schema_version':2,'run_id':f'synthetic-{doc_count}-{text_chars}','model_id':'synthetic-no-model','model_revision':'synthetic','versions':{'bertopic':'0.17.4'},'text_column':'正文','source_sha256':'0'*64},
            'visualization':default_visualization()}
    settings={'year_field':'年份','group_field':'地区','exclude_missing_year':True,'exclude_missing_group':True,'include_outliers':True,'metric':'segment_share','selected_groups':[],'selected_topics':[],'rule_version':1}
    result['visualization']['annual_panel']=settings;create(result,'合成归组',[0,1]);return result


def sample(stage,fn,stats):
    gc.collect();proc=psutil.Process();begin=proc.memory_info().rss;peak=[begin];stop=threading.Event()
    def poll():
        while not stop.wait(.005):peak[0]=max(peak[0],proc.memory_info().rss)
    thread=threading.Thread(target=poll,daemon=True);thread.start();t=time.perf_counter()
    try: value=fn()
    finally:
        elapsed=time.perf_counter()-t;stop.set();thread.join();end=proc.memory_info().rss;peak[0]=max(peak[0],end)
    stats[stage]={'seconds':round(elapsed,4),'rss_before_bytes':begin,'rss_after_bytes':end,'rss_sampled_peak_bytes':peak[0],'sample_interval_ms':5}
    return value


def benchmark(path,kind):
    stats={};raw=path.read_bytes()
    r=sample('open',lambda:open_project(raw),stats);f=r['metadata']['generated_fields'];settings=r['visualization']['annual_panel']
    layout={'rows_logical_bytes':int(r['rows'].memory_usage(index=True,deep=True).sum()),
            'segments_logical_bytes':int(r['segments'].memory_usage(index=True,deep=True).sum()),
            'segment_rows':len(r['segments'])}
    docs,panel,parts=sample('annual',lambda:calculate(r,settings),stats)
    gd,gpanel,gparts,members=sample('grouped_annual',lambda:source(r,settings),stats)
    valid=gd[gd.year.notna() & gd.selected_by_metadata];year=int(valid.iloc[0].year);topic=next(x for x in members if x.startswith('group-'))
    def trace_one():
        _,selected=evidence(r,gd,gparts,members,ALL,year,topic)
        part=selected.iloc[0];parent=r['rows'].set_index(f['document_id']).loc[part[f['document_id']]]
        text=parent[f['original_text']];start=int(part[f['start_char']]);end=int(part[f['end_char']])
        assert text[start:end]==part[f['segment_text']]
        return len(text)
    sample('trace',trace_one,stats)
    csv=sample('csv',lambda:csv_bytes(r['segments']),stats);csv_len=len(csv);del csv;gc.collect()
    try:
        xlsx=sample('excel',lambda:excel_bytes(r),stats);xlsx_len=len(xlsx);del xlsx;gc.collect();excel_error=None
    except ValueError as exc:xlsx_len=None;excel_error=str(exc)
    # Verify policy transfer uses the same saved document and segment links.
    did=str(r['rows'].iloc[0][f['document_id']]);mapping={'title':r['metadata']['original_columns'][0]}
    def transfer():
        plan=plan_transfer(r,hashlib.sha256(raw).hexdigest(),[did],mapping)
        p=apply_transfer(plan);assert p['policies'][0]['original_text']==r['rows'].iloc[0][f['original_text']]
    sample('policy_transfer',transfer,stats)
    # Arrow materializes Python strings in tolist(); count identities only after
    # timing/RSS stages so this diagnostic does not inflate their measurements.
    texts=r['segments'][f['original_text']].tolist()
    layout['original_text_unique_python_objects']=len({id(x) for x in texts})
    return {'kind':kind,'path':str(path.relative_to(ROOT)),'archive_bytes':len(raw),'documents':len(r['rows']),'segments':len(r['segments']),
            'text_characters_per_document':[len(x) for x in r['rows'][f['original_text']]],'layout':layout,'stats':stats,
            'csv_bytes':csv_len,'excel_bytes':xlsx_len,'excel_error':excel_error,
            'settings':settings,'platform':platform.platform(),'cpu_logical':psutil.cpu_count(),'ram_bytes':psutil.virtual_memory().total}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['build','benchmark']);ap.add_argument('--kind',choices=['real','small','large']);ap.add_argument('--label',default='baseline');a=ap.parse_args()
    if a.mode=='build':
        for name,dimensions in [('small',(8,6000,120,500)),('large',(20,12000,120,1000))]:
            p=OUT/f'synthetic_{name}.hpts.zip';p.write_bytes(save_project(build(*dimensions)));print(name,p.stat().st_size,dimensions,flush=True)
    else:
        path=ROOT/'outputs/project_compact_validation/real_policy_grouped_v5.hpts.zip' if a.kind=='real' else OUT/f'synthetic_{a.kind}.hpts.zip'
        data=benchmark(path,a.kind);(OUT/f'{a.label}_{a.kind}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');print(json.dumps(data,ensure_ascii=False),flush=True)
