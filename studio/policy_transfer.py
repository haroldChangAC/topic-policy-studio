"""Transfer whole documents from a validated theme project, preserving a snapshot."""
import copy
import re

import pandas as pd

from studio.corpus import Corpus
from studio.policy import create_project, new_id, now, plan_append
from studio.topic_names import materialize_topics


def documents(result, topic_id=None):
    """Return one item per original document, optionally filtered by segment topic."""
    meta = result['metadata']; f = meta['generated_fields']
    segments = result['segments']; docs = result['rows']
    names = {int(row.Topic): str(row.Display_Name) for row in materialize_topics(result).itertuples()}
    grouped = {}
    for _, s in segments.iterrows():
        topic = int(s[f['topic_id']])
        grouped.setdefault(str(s[f['document_id']]), []).append({
            'segment_id': str(s[f['segment_id']]), 'segment_order': int(s[f['segment_order']]),
            'start': int(s[f['start_char']]), 'end': int(s[f['end_char']]),
            'topic_id': topic, 'topic_name': names[topic]})
    output = []
    for position, (_, row) in enumerate(docs.iterrows(), 1):
        doc_id = str(row[f['document_id']]); parts = sorted(grouped.get(doc_id, []),key=lambda x:x['segment_order'])
        if topic_id is not None and not any(part['topic_id'] == topic_id for part in parts):
            continue
        original = {column: str(row[column]) for column in meta['original_columns']}
        text = str(row[f['original_text']])
        text_column = meta.get('text_column')
        if text_column in original and original[text_column] != text:
            raise ValueError('主题项目原始正文与字段不一致，不能转入。')
        output.append({'document_id': doc_id, 'position': position, 'original_text': text,
                       'original_fields': original, 'status': str(row[f['status']]),
                       'segments': parts})
    return output


def plan_transfer(result, project_digest, selected_ids, mapping=None, target=None, duplicate_policy='keep', text_column=None):
    if not re.fullmatch(r'[0-9a-f]{64}', project_digest or ''):
        raise ValueError('主题项目的文件哈希无效。')
    available = documents(result)
    by_id = {d['document_id']: d for d in available}
    if not selected_ids or len(set(selected_ids)) != len(selected_ids) or any(x not in by_id for x in selected_ids):
        raise ValueError('请选择有效且不重复的原始文档。')
    chosen = [d for d in available if d['document_id'] in set(selected_ids)]
    meta = result['metadata']; columns = meta['original_columns']
    text_column = meta['text_column'] if meta.get('text_column') in columns else text_column
    if text_column not in columns:
        raise ValueError('历史项目未记录正文列，请明确选择原始正文对应的字段。')
    if any(d['original_fields'][text_column] != d['original_text'] for d in chosen):
        raise ValueError('所选正文列与原始文档的完整原文不一致，请重新选择。')
    frame = pd.DataFrame([d['original_fields'] for d in chosen], columns=columns, dtype=str)
    corpus = Corpus(b'', frame, project_digest, 'saved-project', 'theme_project',
                    {'project_run_id': meta['run_id'], 'project_sha256': project_digest,
                     'source_sha256': meta.get('source_sha256', ''),
                     'description': '主题项目中的原始文档快照；不与源项目同步'},
                    [d['position'] for d in chosen])
    plan = plan_append(target or {'policies': []}, corpus, text_column, mapping or {},
                       '主题分析项目', duplicate_policy=duplicate_policy)
    existing = {(p['theme_origin']['project_run_id'], p['theme_origin']['document_id']): p['policy_id']
                for p in (target or {}).get('policies', []) if isinstance(p.get('theme_origin'), dict)}
    existing_ids = {}
    for policy in (target or {}).get('policies', []):
        values = [policy['policy_id'], policy.get('source_original_id', '')]
        values += [v for k,v in policy['original_fields'].items() if k in ('policy_id','id','ID','编号','政策ID','政策编号')]
        for value in values:
            if isinstance(value,str) and value.strip(): existing_ids.setdefault(value.strip(), policy['policy_id'])
    for row, document in zip(plan['rows'], chosen):
        row.update({'document_id': document['document_id'], 'original_text_preview': document['original_text'][:200],
                    'associated_topics': ', '.join(f"{p['topic_id']} · {p['topic_name']}" for p in
                                                   {p['topic_id']: p for p in document['segments']}.values()),
                    'segment_count': len(document['segments']),
                    'already_transferred_as': existing.get((meta['run_id'], document['document_id']), ''),
                    'document_id_conflict_with': existing_ids.get(document['document_id'].strip(), '')})
    plan.update({'corpus': corpus, 'documents': chosen, 'mapping': mapping or {},
                 'project_run_id': meta['run_id'], 'project_digest': project_digest,
                 'theme_id_conflicts': sum(bool(row['already_transferred_as']) for row in plan['rows']),
                 'document_id_conflicts': sum(bool(row['document_id_conflict_with']) for row in plan['rows'])})
    by_policy_id = {row['policy_id']: row for row in plan['rows'] if row['policy_id']}
    for policy in plan['accepted']:
        row = by_policy_id[policy['policy_id']]
        policy['source_id_conflict_with'] = (row['document_id_conflict_with'] or row['already_transferred_as']
                                             or policy['source_id_conflict_with'])
    return plan


def apply_transfer(plan, target=None):
    """Apply the last preview atomically; caller must replan if target changed."""
    if not plan['accepted']:
        raise ValueError('没有可转入的有效原始文档，请检查空文本和重复处理选择。')
    corpus = plan['corpus']; timestamp = now()
    by_record = {f"{corpus.sha256}:{d['position']:08d}": d for d in plan['documents']}
    accepted = copy.deepcopy(plan['accepted'])
    for p in accepted:
        document = by_record[p['source_record_id']]
        p['theme_origin'] = {'project_run_id': plan['project_run_id'], 'project_sha256': plan['project_digest'],
                             'document_id': document['document_id'], 'transferred_at': timestamp,
                             'segments': copy.deepcopy(document['segments'])}
    if target is None:
        project = create_project(corpus, plan['source']['text_column'], plan['mapping'], '主题分析项目')
        project['policies'] = accepted
        project['source']['imported_at'] = timestamp
        project['source']['duplicate_policy'] = plan['duplicate_policy']
        return project
    source_id = new_id('source')
    source = {**copy.deepcopy(plan['source']), 'source_id': source_id, 'original_id_column': None,
              'existing_id_column': None, 'imported_at': timestamp, 'duplicate_policy': plan['duplicate_policy'],
              'valid_count': plan['valid_count'], 'empty_count': plan['empty_count'],
              'duplicate_count': plan['duplicate_count'], 'accepted_count': len(accepted)}
    target.setdefault('sources', []).append(source)
    target['policies'].extend([{**p, 'source_id': source_id} for p in accepted])
    return target
