"""Human-coded, document-level basic PMC evaluation; no automatic judgments."""
import copy
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from studio.corpus import missing_text_formulae, prepare

RULE = {'method': 'binary_equal_weight_basic_pmc',
        'primary': 'sum(binary secondary values) / number of secondary indicators',
        'raw': 'sum(primary scores)', 'normalized_10': 'raw / number of primary indicators * 10',
        'incomplete': 'missing, pending, not_applicable or stale coding blocks total; no denominator adjustment',
        'unit': 'whole_policy_document'}
VALUES = ('yes', 'no', 'pending', 'not_applicable')


def now():
    return datetime.now(ZoneInfo('Asia/Seoul')).isoformat()


def new_id(prefix):
    return f'{prefix}-{uuid.uuid4().hex}'


def create_project(corpus, text_column, mapping=None, filename=''):
    mapping = mapping or {}
    if missing_text_formulae(corpus, text_column):
        raise ValueError('正文列存在没有缓存结果的公式，请在 Excel 重算并保存或改为固定文本。')
    for key, column in mapping.items():
        if key not in ('title','date','agency','region','source') or (column is not None and column not in corpus.frame.columns):
            raise ValueError('政策字段映射无效。')
    prepared = prepare(corpus, text_column, 'keep'); f = prepared.fields
    if prepared.rows.empty:
        raise ValueError('文件没有政策数据行。')
    documents = []
    for ordinal, (_, row) in enumerate(prepared.rows.iterrows(), 1):
        original = {column: str(row[column]) for column in corpus.frame.columns}
        text = str(row[f['original_text']]); revision = new_id('rev')
        selected = {key: original.get(mapping.get(key), '') for key in ('title','date','agency','region','source')}
        if not selected['title'].strip():
            selected['title'] = filename if corpus.source_format == 'txt' and filename else f'未命名政策 {ordinal}'
        documents.append({'policy_id': str(row[f['record_id']]), **selected,
                          'original_fields': original, 'original_text': text,
                          'current_revision': revision,
                          'revisions': [{'revision_id': revision, 'text': text, 'created_at': now(), 'note': '导入原文'}]})
    return {'schema_version': 1, 'project_id': new_id('pmc'), 'created_at': now(),
            'source': {'sha256': corpus.sha256, 'format': corpus.source_format, 'encoding': corpus.encoding,
                       'details': corpus.source_details, 'text_column': text_column,
                       'field_mapping': mapping, 'original_columns': list(corpus.frame.columns), 'filename': filename},
            'policies': documents, 'draft': [], 'versions': [], 'active_version': None, 'encodings': [], 'rule': copy.deepcopy(RULE)}


def plan_append(project, corpus, text_column, mapping=None, filename='', id_column=None, duplicate_policy='keep', existing_id_column=None):
    """Preview an append without changing project or source; source IDs never replace system IDs."""
    if duplicate_policy not in ('keep', 'exclude'):
        raise ValueError('请选择保留或排除重复政策。')
    if id_column is not None and id_column not in corpus.frame.columns:
        raise ValueError('所选原始政策 ID 列不存在。')
    if existing_id_column is not None and not any(existing_id_column in p['original_fields'] for p in project['policies']):
        raise ValueError('原项目所选 ID 列不存在。')
    candidate = create_project(corpus, text_column, mapping, filename)
    existing_ids = {p['policy_id'] for p in project['policies']}
    existing_texts = {}
    known_external = {}
    id_names = {'policy_id', 'id', 'ID', '编号', '政策ID', '政策id', '政策编号'}
    if id_column:
        id_names.add(id_column)
    if existing_id_column:
        id_names.add(existing_id_column)
    for p in project['policies']:
        if p['original_text'].strip():
            existing_texts.setdefault(p['original_text'], p['policy_id'])
        values = [p['policy_id'], p.get('source_original_id', '')] + [v for k,v in p['original_fields'].items() if k in id_names]
        for value in values:
            if isinstance(value,str) and value.strip(): known_external.setdefault(value.strip(), p['policy_id'])
    rows = []; accepted = []; reserved = set(existing_ids)
    for ordinal, p in enumerate(candidate['policies'], 1):
        text = p['original_text']; raw_id = p['original_fields'].get(id_column, '') if id_column else ''
        duplicate_of = existing_texts.get(text, '') if text.strip() else ''
        external_conflict = known_external.get(raw_id.strip(), '') if raw_id.strip() else ''
        base_id = p['policy_id']; target_id = base_id; suffix = 1
        while target_id in reserved:
            target_id = f'{base_id}:append-{suffix:06d}'; suffix += 1
        invalid = not text.strip()
        excluded = bool(duplicate_of) and duplicate_policy == 'exclude'
        accepted_row = not invalid and not excluded
        status = '正文为空，未追加' if invalid else '重复原文，按选择排除' if excluded else '将保留并追加'
        source_row = corpus.source_rows[ordinal-1] if corpus.source_rows else ordinal
        rows.append({'source_row':source_row, 'status':status, 'title':p['title'], 'source_original_id':raw_id,
                     'source_id_conflict_with':external_conflict, 'duplicate_of':duplicate_of,
                     'system_id_collision':target_id != base_id, 'policy_id':target_id if accepted_row else ''})
        if accepted_row:
            p.update({'source_record_id':base_id, 'source_original_id':raw_id,
                      'source_id_conflict_with':external_conflict, 'duplicate_of':duplicate_of,
                      'policy_id':target_id})
            accepted.append(p); reserved.add(target_id)
            existing_texts.setdefault(text, target_id)
            if raw_id.strip(): known_external.setdefault(raw_id.strip(),target_id)
    return {'rows':rows, 'accepted':accepted, 'source':candidate['source'], 'duplicate_policy':duplicate_policy,
            'id_column':id_column, 'existing_id_column':existing_id_column,
            'valid_count':sum(bool(p['original_text'].strip()) for p in candidate['policies']),
            'empty_count':sum(not p['original_text'].strip() for p in candidate['policies']),
            'duplicate_count':sum(bool(row['duplicate_of']) for row in rows),
            'external_id_conflicts':sum(bool(row['source_id_conflict_with']) for row in rows),
            'system_id_collisions':sum(bool(row['system_id_collision']) for row in rows)}


def append_policies(project, corpus, text_column, mapping=None, filename='', id_column=None, duplicate_policy='keep', existing_id_column=None):
    plan = plan_append(project, corpus, text_column, mapping, filename, id_column, duplicate_policy, existing_id_column)
    if not plan['accepted']:
        raise ValueError('没有可追加的有效政策；请检查正文和重复处理选择。')
    source_id = new_id('source')
    source = {**plan['source'], 'source_id':source_id, 'original_id_column':id_column,
              'existing_id_column':existing_id_column,
              'imported_at':now(), 'duplicate_policy':duplicate_policy,
              'valid_count':plan['valid_count'], 'empty_count':plan['empty_count'],
              'duplicate_count':plan['duplicate_count'], 'accepted_count':len(plan['accepted'])}
    new_policies = [{**item, 'source_id':source_id} for item in plan['accepted']]
    project.setdefault('sources', []).append(source)
    project['policies'].extend(new_policies)
    return plan


def coding_matrix(project):
    """A missing record is '待编码', not an invented manual coding event."""
    version = active_version(project)
    if not version:
        return []
    latest = {(e['policy_id'],e['indicator_id']):e for e in project['encodings'] if e['version_id']==version['version_id']}
    result=[]
    for policy in project['policies']:
        for indicator in version['indicators']:
            if indicator['level'] != 2:
                continue
            event = latest.get((policy['policy_id'],indicator['id']))
            state = ('待编码' if event is None else '文本修订后待重新确认' if event['revision_id'] != policy['current_revision'] else
                     {'yes':'符合','no':'不符合','pending':'待核查','not_applicable':'不适用'}[event['value']])
            result.append({'policy_id':policy['policy_id'], 'title':policy['title'], 'version_id':version['version_id'],
                           'indicator_id':indicator['id'], 'indicator_name':indicator['name'], 'status':state,
                           'event_id':event['event_id'] if event else '',
                           'binary_value':(1 if event and state=='符合' else 0 if event and state=='不符合' else None)})
    return result


def get_policy(project, policy_id):
    return next((p for p in project['policies'] if p['policy_id'] == policy_id), None)


def revision(policy, revision_id=None):
    revision_id = revision_id or policy['current_revision']
    found = next((r for r in policy['revisions'] if r['revision_id'] == revision_id), None)
    if found is None:
        raise ValueError('政策文本版本不存在。')
    return found


def revise_policy(project, policy_id, text, note=''):
    policy = get_policy(project, policy_id)
    if policy is None or not isinstance(text, str) or not isinstance(note, str):
        raise ValueError('政策文本或版本说明无效。')
    if text == revision(policy)['text']:
        return False
    item = {'revision_id': new_id('rev'), 'text': text, 'created_at': now(), 'note': note}
    policy['revisions'].append(item); policy['current_revision'] = item['revision_id']
    return True


def add_indicator(project, level, name, definition, parent_id=None):
    if level not in (1, 2) or not isinstance(name, str) or not name.strip() or len(name) > 200 or not isinstance(definition, str):
        raise ValueError('请填写指标名称（最多 200 字符）及编码说明。')
    if level == 2 and not any(i['id'] == parent_id and i['level'] == 1 for i in project['draft']):
        raise ValueError('二级指标必须选择现有一级指标。')
    item = {'id': new_id('ind'), 'level': level, 'parent_id': parent_id if level == 2 else None,
            'name': name.strip(), 'definition': definition}
    project['draft'].append(item)
    return item['id']


def edit_indicator(project, indicator_id, name, definition, parent_id=None):
    if not isinstance(name, str) or not name.strip() or len(name) > 200 or not isinstance(definition, str):
        raise ValueError('请填写指标名称（最多 200 字符）及编码说明。')
    item = next((i for i in project['draft'] if i['id'] == indicator_id), None)
    if item is None:
        raise ValueError('指标不存在。')
    if item['level'] == 2 and parent_id is not None:
        if not any(i['id'] == parent_id and i['level'] == 1 for i in project['draft']):
            raise ValueError('二级指标必须选择现有一级指标。')
        item['parent_id'] = parent_id
    item['name'] = name.strip(); item['definition'] = definition


def delete_indicator(project, indicator_id):
    item = next((i for i in project['draft'] if i['id'] == indicator_id), None)
    if item is None:
        raise ValueError('指标不存在。')
    project['draft'] = [i for i in project['draft'] if i['id'] != indicator_id and i['parent_id'] != indicator_id]


def validate_indicators(items, require_complete=True):
    if not isinstance(items, list):
        raise ValueError('指标体系无效。')
    ids = set()
    for item in items:
        if (not isinstance(item, dict) or set(item) != {'id','level','parent_id','name','definition'}
            or not isinstance(item['id'], str) or not item['id'] or item['id'] in ids
            or isinstance(item['level'], bool) or item['level'] not in (1,2)
            or not isinstance(item['name'], str) or not item['name'].strip() or len(item['name']) > 200
            or not isinstance(item['definition'], str)):
            raise ValueError('指标编号、名称或说明无效。')
        ids.add(item['id'])
    parents = {i['id'] for i in items if i['level'] == 1}
    for item in items:
        if (item['level'] == 1 and item['parent_id'] is not None) or (item['level'] == 2 and item['parent_id'] not in parents):
            raise ValueError('二级指标的一级指标不存在。')
    if require_complete and (not parents or any(not any(j['parent_id'] == parent for j in items) for parent in parents)):
        raise ValueError('请至少设置一个一级指标，且每个一级指标至少包含一个二级指标。')
    if require_complete and any(not item['definition'].strip() for item in items):
        raise ValueError('确认前请为每个一级和二级指标填写定义或编码说明。')


def confirm_indicators(project):
    validate_indicators(project['draft'])
    active = active_version(project)
    if active and active['indicators'] == project['draft']:
        return False
    item = {'version_id': new_id('scheme'), 'confirmed_at': now(), 'indicators': copy.deepcopy(project['draft'])}
    project['versions'].append(item); project['active_version'] = item['version_id']
    return True


def active_version(project):
    return next((v for v in project['versions'] if v['version_id'] == project['active_version']), None)


def record_encoding(project, policy_id, indicator_id, value, evidence='', start=None, end=None, rationale='', coder=''):
    policy, version = get_policy(project, policy_id), active_version(project)
    if policy is None or version is None or not any(i['id'] == indicator_id and i['level'] == 2 for i in version['indicators']):
        raise ValueError('请先确认指标体系，并选择有效政策及二级指标。')
    if value not in VALUES or not all(isinstance(x, str) for x in (evidence, rationale, coder)):
        raise ValueError('编码状态或记录内容无效。')
    text = revision(policy)['text']
    if not text.strip():
        raise ValueError('政策正文为空，请先补充修订文本；不能对空正文进行编码。')
    if (start is None) != (end is None):
        raise ValueError('精确位置必须同时提供起点和终点。')
    if start is not None:
        if (isinstance(start, bool) or not isinstance(start, int) or isinstance(end, bool) or not isinstance(end, int)
            or not 0 <= start < end <= len(text) or not evidence or text[start:end] != evidence):
            raise ValueError('证据与所选原文位置不一致；位置从 0 开始，终点不包含。')
    event = {'event_id': new_id('coding'), 'version_id': version['version_id'], 'policy_id': policy_id,
             'indicator_id': indicator_id, 'value': value, 'evidence': evidence, 'start': start, 'end': end,
             'rationale': rationale, 'coder': coder, 'revision_id': policy['current_revision'], 'changed_at': now()}
    previous = next((e for e in reversed(project['encodings']) if all(e[k] == event[k] for k in ('version_id','policy_id','indicator_id'))), None)
    comparable = ('value','evidence','start','end','rationale','coder','revision_id')
    if previous and all(previous[k] == event[k] for k in comparable):
        return False
    project['encodings'].append(event)
    return True


def score(project, version_id=None):
    version_id = version_id or project['active_version']
    version = next((v for v in project['versions'] if v['version_id'] == version_id), None)
    latest = {(e['policy_id'], e['indicator_id']):e for e in project['encodings'] if e['version_id'] == version_id}
    primary_rows, policy_rows = [], []
    for policy in project['policies']:
        p_id = policy['policy_id']; reasons = []
        if version is None:
            policy_rows.append({'policy_id':p_id, 'title':policy['title'], 'version_id':version_id,
                                'primary_count':0, 'theoretical_max':0, 'raw_pmc':None, 'normalized_10':None, 'reason':'指标体系尚未确认'})
            continue
        indicators = version['indicators']; parents = [x for x in indicators if x['level'] == 1]
        if not parents:
            reasons.append('无一级指标')
        total = 0.0
        if not revision(policy)['text'].strip():
            reasons.append('政策正文为空')
        for parent in parents:
            children = [x for x in indicators if x['level'] == 2 and x['parent_id'] == parent['id']]
            numerator = 0; incomplete = []
            for child in children:
                item = latest.get((p_id, child['id']))
                state = ('missing' if item is None else 'stale' if item['revision_id'] != policy['current_revision'] else item['value'])
                if state in ('yes','no'):
                    numerator += int(state == 'yes')
                else:
                    incomplete.append(f"{child['id']} {child['name']}：" + {'missing':'缺失编码','stale':'政策修订后待重新确认','pending':'待核查','not_applicable':'不适用'}[state])
            if not children:
                incomplete.append('没有二级指标')
            value = numerator / len(children) if children and not incomplete and revision(policy)['text'].strip() else None
            if value is None:
                reasons.extend(incomplete or [f"{parent['name']}：正文为空"])
            else:
                total += value
            primary_rows.append({'policy_id':p_id,'title':policy['title'],'version_id':version_id,
                                 'primary_id':parent['id'],'primary_name':parent['name'],
                                 'numerator':numerator,'denominator':len(children),'score':value,
                                 'reason':'；'.join(incomplete)})
        complete = not reasons and bool(parents)
        policy_rows.append({'policy_id':p_id,'title':policy['title'],'version_id':version_id,
                            'primary_count':len(parents),'theoretical_max':len(parents),
                            'raw_pmc':total if complete else None,
                            'normalized_10':total/len(parents)*10 if complete else None,
                            'reason':'；'.join(reasons)})
    return primary_rows, policy_rows
