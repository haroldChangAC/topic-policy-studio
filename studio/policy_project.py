"""Policy-evaluation ZIP, CSV and Excel; separate from theme projects."""
import hashlib
import io
import json
import re
import stat
import zipfile
import zlib

import pandas as pd
from openpyxl import Workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

from studio.policy import RULE, VALUES, active_version, coding_matrix, get_policy, now, revision, score, validate_indicators
from studio.project import MAX_ARCHIVE, MAX_ENTRY, MAX_TOTAL

FORMAT = 'humanities-policy-evaluation'
VERSION = 6
FILES = {'project.json','policies.json','indicators.json','encodings.json','results.json'}
AI_FILE = 'ai_assist.json'
LONG_FILE = 'long_assist.json'


def validate(project):
    if not isinstance(project, dict) or project.get('schema_version') != 1 or project.get('rule') != RULE:
        raise ValueError('政策评价项目格式或计算规则无效。')
    source = project['source']; source_map = {}
    sources = project.get('sources', [])
    if not isinstance(sources, list):
        raise ValueError('政策追加来源记录无效。')
    for item in sources:
        if not isinstance(item, dict) or not isinstance(item.get('source_id'), str) or not item['source_id'] or item['source_id'] in source_map:
            raise ValueError('政策追加来源标识无效或重复。')
        source_map[item['source_id']] = item
    for item in [source, *sources]:
        columns = item['original_columns']; text_col = item['text_column']
        if not isinstance(columns, list) or len(columns) != len(set(columns)) or text_col not in columns:
            raise ValueError('政策原始字段映射无效。')
        if item.get('original_id_column') is not None and item['original_id_column'] not in columns:
            raise ValueError('原始政策 ID 列不存在。')
        if item.get('format') == 'theme_project':
            details = item.get('details', {})
            if (not re.fullmatch(r'[0-9a-f]{64}', item.get('sha256', ''))
                or not isinstance(details, dict) or not isinstance(details.get('project_run_id'), str)
                or not details['project_run_id'] or details.get('project_sha256') != item['sha256']):
                raise ValueError('主题材料来源记录无效。')
    policies = project['policies']; ids = set(); revision_ids = set()
    if not isinstance(policies, list) or not policies:
        raise ValueError('项目没有政策记录。')
    for p in policies:
        if not isinstance(p, dict) or not isinstance(p.get('policy_id'), str) or not p['policy_id'] or p['policy_id'] in ids:
            raise ValueError('政策 ID 无效或重复。')
        ids.add(p['policy_id'])
        if any(not isinstance(p.get(k), str) for k in ('title','date','agency','region','source','original_text')):
            raise ValueError('政策字段无效。')
        source_id = p.get('source_id')
        if source_id is not None and source_id not in source_map:
            raise ValueError('政策指向不存在的追加来源。')
        document_source = source_map[source_id] if source_id is not None else source
        columns, text_col = document_source['original_columns'], document_source['text_column']
        if not isinstance(p.get('original_fields'), dict) or set(p['original_fields']) != set(columns) or any(not isinstance(v,str) for v in p['original_fields'].values()) or p['original_fields'][text_col] != p['original_text']:
            raise ValueError('政策原始字段或原文不一致。')
        if document_source.get('format') == 'theme_project':
            origin = p.get('theme_origin'); details = document_source['details']
            keys = {'project_run_id','project_sha256','document_id','transferred_at','segments'}
            if (not isinstance(origin, dict) or set(origin) != keys
                or origin['project_run_id'] != details['project_run_id'] or origin['project_sha256'] != document_source['sha256']
                or not isinstance(origin['document_id'], str) or not origin['document_id']
                or not isinstance(origin['transferred_at'], str) or not origin['transferred_at']
                or not isinstance(origin['segments'], list)):
                raise ValueError('主题材料文档来源无效。')
            base_id=p.get('source_record_id','')
            if (not isinstance(base_id,str) or not re.fullmatch(re.escape(document_source['sha256'])+r':\d{8,}',base_id)
                or not re.fullmatch(re.escape(base_id)+r'(?::append-\d{6,})?',p['policy_id'])):
                raise ValueError('主题材料系统 ID 与来源记录不一致。')
            segment_ids = set(); cursor = 0
            for order,part in enumerate(origin['segments'],1):
                if (not isinstance(part, dict) or set(part) != {'segment_id','segment_order','start','end','topic_id','topic_name'}
                    or not isinstance(part['segment_id'], str) or not part['segment_id'] or part['segment_id'] in segment_ids
                    or any(isinstance(part[k],bool) or not isinstance(part[k],int) for k in ('segment_order','start','end','topic_id'))
                    or part['segment_order'] != order or part['start'] != cursor or not 0 <= part['start'] < part['end'] <= len(p['original_text'])
                    or not isinstance(part['topic_name'],str)):
                    raise ValueError('主题材料片段来源或位置无效。')
                segment_ids.add(part['segment_id']); cursor=part['end']
            if origin['segments'] and cursor != len(p['original_text']):
                raise ValueError('主题材料片段位置未覆盖原始全文。')
        elif 'theme_origin' in p:
            raise ValueError('主题材料来源与政策记录不一致。')
        if source_id is not None:
            id_col = document_source.get('original_id_column')
            if (any(not isinstance(p.get(k),str) for k in ('source_record_id','source_original_id','source_id_conflict_with','duplicate_of'))
                or not p['source_record_id'] or p['source_original_id'] != (p['original_fields'].get(id_col,'') if id_col else '')):
                raise ValueError('追加政策的来源或原始 ID 不一致。')
            fmt = document_source.get('format'); sha = document_source.get('sha256')
            if not isinstance(sha,str) or not re.fullmatch(r'[0-9a-f]{64}',sha) or fmt not in ('csv','xlsx','txt','theme_project'):
                raise ValueError('追加来源格式或哈希无效。')
            namespace = (':sheet-' + hashlib.sha256(document_source['details']['sheet'].encode('utf-8')).hexdigest()[:16]
                         if fmt == 'xlsx' else ':txt' if fmt == 'txt' else '')
            if not re.fullmatch(re.escape(sha+namespace)+r':\d{8,}',p['source_record_id']):
                raise ValueError('追加政策记录 ID 与来源不一致。')
            if not re.fullmatch(re.escape(p['source_record_id'])+r'(?::append-\d{6,})?',p['policy_id']):
                raise ValueError('追加政策系统 ID 与来源记录不一致。')
        versions = p.get('revisions')
        if not isinstance(versions, list) or not versions or versions[0].get('text') != p['original_text']:
            raise ValueError('政策原文版本缺失或不一致。')
        for item in versions:
            if (not isinstance(item, dict) or set(item) != {'revision_id','text','created_at','note'}
                or any(not isinstance(item[k], str) for k in item) or not item['revision_id'] or item['revision_id'] in revision_ids):
                raise ValueError('政策文本版本无效。')
            revision_ids.add(item['revision_id'])
        revision(p)
    for p in policies:
        for key in ('source_id_conflict_with','duplicate_of'):
            if p.get(key) and p[key] not in ids:
                raise ValueError('追加政策的冲突或重复引用不存在。')
    validate_indicators(project['draft'], require_complete=False)
    versions = project['versions']; version_map = {}
    if not isinstance(versions, list):
        raise ValueError('指标版本无效。')
    for v in versions:
        if not isinstance(v, dict) or set(v) != {'version_id','confirmed_at','indicators'} or not isinstance(v['version_id'], str) or not v['version_id'] or v['version_id'] in version_map or not isinstance(v['confirmed_at'], str):
            raise ValueError('指标版本无效或重复。')
        validate_indicators(v['indicators']); version_map[v['version_id']] = v
    if project['active_version'] is not None and project['active_version'] not in version_map:
        raise ValueError('当前指标版本不存在。')
    if versions and project['active_version'] is None:
        raise ValueError('已确认指标但没有当前版本。')
    events = project['encodings']; event_ids = set()
    if not isinstance(events, list):
        raise ValueError('编码记录无效。')
    for e in events:
        fields = {'event_id','version_id','policy_id','indicator_id','value','evidence','start','end','rationale','coder','revision_id','changed_at'}
        if not isinstance(e, dict) or set(e) != fields or e['event_id'] in event_ids or not all(isinstance(e[k],str) for k in fields-{'start','end'}) or e['value'] not in VALUES:
            raise ValueError('编码记录无效。')
        event_ids.add(e['event_id']); p = get_policy(project, e['policy_id']); v = version_map.get(e['version_id'])
        if p is None or v is None or not any(i['id']==e['indicator_id'] and i['level']==2 for i in v['indicators']):
            raise ValueError('编码指向不存在的政策或指标。')
        text = revision(p, e['revision_id'])['text']; start, end = e['start'], e['end']
        if not text.strip() or (start is None) != (end is None):
            raise ValueError('编码正文或证据位置无效。')
        if start is not None and (isinstance(start,bool) or not isinstance(start,int) or isinstance(end,bool) or not isinstance(end,int)
                                  or not 0 <= start < end <= len(text) or not e['evidence'] or text[start:end] != e['evidence']):
            raise ValueError('证据与原文位置不一致。')
    from studio.policy_charts import validate_chart_settings
    validate_chart_settings(project,project.get('chart_settings'))
    from studio.policy_assist import validate_assist
    validate_assist(project)
    from studio.policy_long import validate_long_jobs
    validate_long_jobs(project)


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8')


def save_policy_project(project):
    validate(project)
    primary, policies = score(project)
    theme = any(s.get('format') == 'theme_project' for s in [project['source'], *project.get('sources', [])])
    has_ai=bool(project.get('ai_suggestions') or project.get('ai_reviews'))
    has_long=bool(project.get('long_jobs'))
    version = 6 if has_long else 5 if has_ai else 4 if theme else 3 if 'chart_settings' in project else 2 if project.get('sources') else 1
    meta = {'format':FORMAT,'version':version,'saved_at':now(),'project_id':project['project_id'],
            'created_at':project['created_at'],'source':project['source'],'rule':project['rule']}
    if project.get('sources'):
        meta['sources'] = project['sources']
    if 'chart_settings' in project:
        meta['chart_settings'] = project['chart_settings']
    data = {'project.json':_json(meta), 'policies.json':_json(project['policies']),
            'indicators.json':_json({'draft':project['draft'],'versions':project['versions'],'active_version':project['active_version']}),
            'encodings.json':_json(project['encodings']), 'results.json':_json({'primary':primary,'policies':policies})}
    if has_ai or has_long:
        data[AI_FILE]=_json({'suggestions':project.get('ai_suggestions',[]),'reviews':project.get('ai_reviews',[])})
    if has_long:
        data[LONG_FILE]=_json({'jobs':project['long_jobs']})
    if any(len(x)>MAX_ENTRY for x in data.values()) or sum(map(len,data.values()))>MAX_TOTAL:
        raise ValueError('政策评价项目内容超过大小上限。')
    manifest = {'algorithm':'sha256','files':{k:{'size':len(v),'sha256':hashlib.sha256(v).hexdigest()} for k,v in data.items()}}
    data['manifest.json'] = _json(manifest); output=io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for name, content in data.items(): z.writestr(name,content)
    if output.tell()>MAX_ARCHIVE:
        raise ValueError('政策评价 ZIP 超过 50 MiB。')
    return output.getvalue()


def open_policy_project(raw):
    if not raw or len(raw)>MAX_ARCHIVE:
        raise ValueError('政策评价项目为空或超过 50 MiB。')
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            entries=z.infolist(); names=[x.filename for x in entries]
            if len(names)!=len(set(names)) or any('/' in n or '\\' in n or n in ('.','..') for n in names):
                raise ValueError('项目包含重复文件或异常路径。')
            expected=FILES|{'manifest.json'}|({AI_FILE} if AI_FILE in names else set())|({LONG_FILE} if LONG_FILE in names else set())
            if set(names)!=expected:
                raise ValueError('政策评价项目文件缺失或包含不支持的文件；主题项目请使用“打开已有项目”。')
            if any(i.is_dir() or stat.S_ISLNK(i.external_attr>>16) or i.flag_bits&1 or i.file_size>MAX_ENTRY for i in entries) or sum(i.file_size for i in entries)>MAX_TOTAL:
                raise ValueError('政策评价项目包含异常文件或内容过大。')
            files={n:z.read(n) for n in names}
        manifest=json.loads(files['manifest.json'])
        expected_files=FILES|({AI_FILE} if AI_FILE in files else set())|({LONG_FILE} if LONG_FILE in files else set())
        if manifest['algorithm']!='sha256' or set(manifest['files'])!=expected_files:
            raise ValueError('政策评价完整性清单无效。')
        for name in expected_files:
            item=manifest['files'][name]
            if item['size']!=len(files[name]) or item['sha256']!=hashlib.sha256(files[name]).hexdigest():
                raise ValueError(f'政策评价完整性校验失败：{name}')
        info=json.loads(files['project.json'])
        if info.get('format')!=FORMAT or info.get('version') not in (1, 2, 3, 4, 5, VERSION):
            raise ValueError('政策评价项目格式或版本不兼容。')
        if (info['version'] in (5,6)) != (AI_FILE in files) or (info['version']==6)!=(LONG_FILE in files):
            raise ValueError('政策项目版本与模型建议文件不一致。')
        indicators=json.loads(files['indicators.json'])
        project={'schema_version':1,'project_id':info['project_id'],'created_at':info['created_at'],
                 'source':info['source'],'rule':info['rule'],'policies':json.loads(files['policies.json']),
                 'draft':indicators['draft'],'versions':indicators['versions'],'active_version':indicators['active_version'],
                 'encodings':json.loads(files['encodings.json'])}
        if info['version'] == 2 or 'sources' in info:
            if not isinstance(info.get('sources'), list) or not info['sources']:
                raise ValueError('政策评价项目缺少追加来源记录。')
            project['sources'] = info['sources']
        if info['version'] == 3 and 'chart_settings' not in info:
            raise ValueError('政策评价项目缺少图表设置。')
        if 'chart_settings' in info:
            project['chart_settings'] = info['chart_settings']
        theme = any(s.get('format') == 'theme_project' for s in [project['source'], *project.get('sources', [])])
        if (info['version'] == 4 and not theme) or (info['version'] < 4 and theme):
            raise ValueError('政策评价项目版本与主题材料来源不一致。')
        if info['version'] in (5,6):
            assist=json.loads(files[AI_FILE])
            if not isinstance(assist,dict) or set(assist)!={'suggestions','reviews'}:
                raise ValueError('模型建议文件格式无效。')
            project['ai_suggestions']=assist['suggestions'];project['ai_reviews']=assist['reviews']
        if info['version']==6:
            long_data=json.loads(files[LONG_FILE])
            if not isinstance(long_data,dict) or set(long_data)!={'jobs'} or not long_data['jobs']:
                raise ValueError('长政策分段任务文件无效。')
            project['long_jobs']=long_data['jobs']
        validate(project)
        primary, policies=score(project)
        if json.loads(files['results.json']) != {'primary':primary,'policies':policies}:
            raise ValueError('保存的得分与编码重新计算结果不一致。')
        return project
    except json.JSONDecodeError as exc:
        raise ValueError('政策评价项目 JSON 内容无法读取。') from exc
    except ValueError:
        raise
    except (zipfile.BadZipFile, OSError, EOFError, RuntimeError, KeyError, TypeError, UnicodeError, OverflowError, AttributeError, IndexError, zlib.error) as exc:
        raise ValueError('政策评价项目损坏或无法读取。') from exc


def tables(project):
    p_rows=[]; revision_rows=[]
    source_map={item['source_id']:item for item in project.get('sources', [])}
    for p in project['policies']:
        source=source_map.get(p.get('source_id'),project['source'])
        origin=p.get('theme_origin', {})
        p_rows.append({'policy_id':p['policy_id'],'title':p['title'],'date':p['date'],'agency':p['agency'],
                       'region':p['region'],'source':p['source'],'original_text':p['original_text'],
                       'current_text':revision(p)['text'],'current_revision':p['current_revision'],
                       'import_source_id':p.get('source_id','首次导入'), 'import_format':source['format'],
                       'import_filename':source['filename'], 'import_sha256':source['sha256'],
                       'import_sheet':source['details'].get('sheet',''),
                       'source_record_id':p.get('source_record_id',p['policy_id']),
                       'source_original_id':p.get('source_original_id',''),
                       'source_id_conflict_with':p.get('source_id_conflict_with',''),
                       'duplicate_of':p.get('duplicate_of',''),
                       'theme_project_run_id':origin.get('project_run_id',''),
                       'theme_project_sha256':origin.get('project_sha256',''),
                       'theme_document_id':origin.get('document_id',''),
                       'theme_transferred_at':origin.get('transferred_at',''),
                       'theme_segments_json':json.dumps(origin.get('segments',[]),ensure_ascii=False),
                       **{'raw::'+k:v for k,v in p['original_fields'].items()}})
        for v in p['revisions']:
            revision_rows.append({'policy_id':p['policy_id'], **v})
    ind=[]
    for v in project['versions']:
        ind.extend({'version_id':v['version_id'],'confirmed_at':v['confirmed_at'],**item} for item in v['indicators'])
    ind.extend({'version_id':'draft','confirmed_at':'',**item} for item in project['draft'])
    primary, policies=score(project)
    matrix=coding_matrix(project)
    by_policy={p['policy_id']:[] for p in project['policies']}
    for item in matrix:
        by_policy[item['policy_id']].append(item)
    for row in policies:
        items=by_policy[row['policy_id']]
        row['coding_status']='指标未确认' if not project['active_version'] else '已完成' if row['raw_pmc'] is not None else '未完成'
        row['binary_coded_count']=sum(x['status'] in ('符合','不符合') for x in items)
        row['required_count']=len(items)
        row['uncoded_count']=sum(x['status']=='待编码' for x in items)
    last = {(e['version_id'],e['policy_id'],e['indicator_id']):e['event_id'] for e in project['encodings']}
    encoding_rows=[]
    for event in project['encodings']:
        policy=get_policy(project,event['policy_id'])
        encoding_rows.append({**event,'record_type':'人工编码记录','binary_value':1 if event['value']=='yes' else 0 if event['value']=='no' else None,
                              'status_label':{'yes':'符合','no':'不符合','pending':'待核查','not_applicable':'不适用'}[event['value']],
                              'is_latest':last[(event['version_id'],event['policy_id'],event['indicator_id'])]==event['event_id'],
                              'is_current_version':event['version_id']==project['active_version'],
                              'is_current_text':event['revision_id']==policy['current_revision']})
    for item in matrix:
        if item['status']=='待编码':
            encoding_rows.append({'event_id':'','version_id':item['version_id'],'policy_id':item['policy_id'],
                                  'indicator_id':item['indicator_id'],'value':'uncoded','record_type':'待编码状态（非人工记录）',
                                  'binary_value':None,'status_label':'待编码','evidence':'','start':None,'end':None,
                                  'rationale':'','coder':'','revision_id':'','changed_at':'',
                                  'is_latest':False,'is_current_version':True,'is_current_text':None})
    import_rows=[]
    for source in [project['source'], *project.get('sources', [])]:
        import_rows.append({'source_id':source.get('source_id','首次导入'), 'filename':source['filename'],
                            'format':source['format'], 'sha256':source['sha256'], 'encoding':source['encoding'],
                            'sheet':source['details'].get('sheet',''), 'text_column':source['text_column'],
                            'original_id_column':source.get('original_id_column'),
                            'existing_id_column':source.get('existing_id_column'),
                            'field_mapping':json.dumps(source['field_mapping'],ensure_ascii=False),
                            'original_columns':json.dumps(source['original_columns'],ensure_ascii=False),
                            'details':json.dumps(source['details'],ensure_ascii=False),
                            'imported_at':source.get('imported_at',''), 'duplicate_policy':source.get('duplicate_policy','首次导入保留'),
                            'valid_count':source.get('valid_count'), 'empty_count':source.get('empty_count'),
                            'duplicate_count':source.get('duplicate_count'), 'accepted_count':source.get('accepted_count')})
    from studio.policy_assist import status as suggestion_status
    from studio.policy_long import candidates, coverage, is_current
    ai_rows=[{**s,'current_status':suggestion_status(project,s),
              'parameters':json.dumps(s['parameters'],ensure_ascii=False),
              'matches':json.dumps(s['matches'],ensure_ascii=False)} for s in project.get('ai_suggestions',[])]
    long_rows=[];long_segments=[];long_evidence=[];long_unresolved=[]
    for job in project.get('long_jobs',[]):
        long_rows.append({'job_id':job['job_id'],'policy_id':job['policy_id'],'indicator_id':job['indicator_id'],
                          'version_id':job['version_id'],'revision_id':job['revision_id'],'model':job['model'],
                          'created_at':job['created_at'],'is_current':is_current(project,job),
                          'settings':json.dumps(job['settings'],ensure_ascii=False),'coverage':json.dumps(coverage(job),ensure_ascii=False)})
        long_segments.extend({'job_id':job['job_id'],**s,'attempt_ids':json.dumps(s['attempt_ids'],ensure_ascii=False)} for s in job['segments'])
        found,unresolved=candidates(project,job)
        long_evidence.extend({'job_id':job['job_id'],**c,'segment_orders':json.dumps(c['segment_orders']),
                              'suggestion_ids':json.dumps(c['suggestion_ids']),'model_statuses':json.dumps(c['model_statuses']),
                              'match_statuses':json.dumps(c['match_statuses'])} for c in found)
        long_unresolved.extend({'job_id':job['job_id'],**u} for u in unresolved)
    return {'policies':pd.DataFrame(p_rows),
            'indicators':pd.DataFrame(ind,columns=['version_id','confirmed_at','id','level','parent_id','name','definition']),
            'encodings':pd.DataFrame(encoding_rows,columns=['event_id','version_id','policy_id','indicator_id','value','record_type','binary_value','status_label','evidence','start','end','rationale','coder','revision_id','changed_at','is_latest','is_current_version','is_current_text']),
            'coding_status':pd.DataFrame(matrix,columns=['title','indicator_name','status','binary_value','policy_id','version_id','indicator_id','event_id']),
            'primary_scores':pd.DataFrame(primary,columns=['policy_id','title','version_id','primary_id','primary_name','numerator','denominator','score','reason']),
            'policy_scores':pd.DataFrame(policies,columns=['title','coding_status','binary_coded_count','required_count','uncoded_count','raw_pmc','theoretical_max','normalized_10','reason','policy_id','version_id','primary_count']),
            'text_revisions':pd.DataFrame(revision_rows,columns=['policy_id','revision_id','text','created_at','note']),
            'import_sources':pd.DataFrame(import_rows),
            'ai_suggestions':pd.DataFrame(ai_rows),
            'ai_reviews':pd.DataFrame(project.get('ai_reviews',[])),
            'long_jobs':pd.DataFrame(long_rows),'long_segments':pd.DataFrame(long_segments),
            'long_evidence':pd.DataFrame(long_evidence),'long_unresolved':pd.DataFrame(long_unresolved)}


def excel_policy(project):
    workbook=Workbook();workbook.remove(workbook.active)
    for name, frame in tables(project).items():
        if len(frame)>1048575 or len(frame.columns)>16384:
            raise ValueError('政策表格超出 Excel 行列容量。')
        sheet=workbook.create_sheet(name)
        for row in [list(frame.columns), *frame.itertuples(index=False,name=None)]:
            values=[]
            for value in row:
                if pd.isna(value): value=None
                if isinstance(value,str) and (len(value)>32767 or ILLEGAL_CHARACTERS_RE.search(value)):
                    raise ValueError('存在 Excel 无法无损保存的超长字段或控制字符，请使用 CSV 和项目 ZIP。')
                values.append(value)
            sheet.append(values)
            for cell in sheet[sheet.max_row]:
                if isinstance(cell.value,str): cell.data_type='s'
        sheet.freeze_panes='A2'
    metadata=workbook.create_sheet('calculation_rule')
    metadata.append(['setting','value_json'])
    for key,value in RULE.items():
        metadata.append([key,json.dumps(value,ensure_ascii=False)])
    buffer=io.BytesIO();workbook.save(buffer);output=io.BytesIO()
    with zipfile.ZipFile(buffer) as src, zipfile.ZipFile(output,'w') as dst:
        for info in src.infolist():
            content=src.read(info.filename)
            if info.filename.startswith('xl/worksheets/') and info.filename.endswith('.xml'):
                content=content.replace(b'\r',b'&#13;')
            dst.writestr(info,content)
    return output.getvalue()
