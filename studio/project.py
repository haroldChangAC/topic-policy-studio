"""Portable, data-only analysis projects. Never extract archive contents."""
import hashlib
import io
import json
import numbers
import stat
import zipfile
import zlib
from datetime import datetime
from zoneinfo import ZoneInfo

from studio.visualization import default_visualization, validate_visualization

import pandas as pd

LEGACY_FORMAT_VERSION = 3
GROUPED_FORMAT_VERSION = 4
FORMAT_VERSION = 5
YEARLY_FORMAT_VERSION = 6
YEARLY_FILE = 'yearly_analysis.json'
MAX_ARCHIVE = 50 * 1024 * 1024
MAX_ENTRY = 100 * 1024 * 1024
MAX_TOTAL = 200 * 1024 * 1024
MAX_RECONSTRUCTED = 512 * 1024 * 1024
FILES = {'project.json', 'records.json', 'segments.json', 'topics.json', 'run.json'}
EDIT_FILE = 'topic_edits.json'
VISUAL_FILE = 'visualization.json'


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8')


def _table(frame):
    return frame.to_json(orient='split', force_ascii=False, double_precision=15).encode('utf-8')


def _compact_segments(result):
    """Store document-owned columns once; save the original order for restoration."""
    frame = result['segments']
    f = result['metadata']['generated_fields']
    derived = set(result['metadata']['original_columns']) | {f['original_text'], f['segment_text']}
    docs = result['rows'].set_index(f['document_id'])
    sizes = {doc_id: sum(len(row[column].encode('utf-8'))
                         for column in derived - {f['segment_text']} if isinstance(row[column], str))
             for doc_id, row in docs.iterrows()}
    total = 0
    for doc_id, excerpt in frame[[f['document_id'], f['segment_text']]].itertuples(index=False, name=None):
        total += sizes[doc_id] + len(excerpt.encode('utf-8'))
        if total > MAX_RECONSTRUCTED:
            raise ValueError('恢复后的片段字段超过 512 MiB 上限，无法保存可重新打开的项目。')
    reduced = json.loads(_table(frame.drop(columns=list(derived))))
    reduced.pop('index', None)
    reduced['column_order'] = list(frame.columns)
    return _json(reduced)


def _restore_segments(value, docs, meta):
    """Reconstruct and check v5 references before the ordinary project checks."""
    if not isinstance(value, dict) or set(value) != {'columns', 'data', 'column_order'}:
        raise ValueError('精简片段表结构无效。')
    columns, records, order = value['columns'], value['data'], value['column_order']
    if (not isinstance(columns, list) or not isinstance(order, list)
            or any(not isinstance(x, str) for x in columns + order)
            or len(columns) != len(set(columns)) or len(order) != len(set(order))
            or not isinstance(records, list)
            or any(not isinstance(row, list) or len(row) != len(columns) for row in records)):
        raise ValueError('精简片段表结构无效。')
    f = meta['generated_fields']
    derived = set(meta['original_columns']) | {f['original_text'], f['segment_text']}
    required = {f[k] for k in ('document_id', 'segment_id', 'segment_order', 'start_char', 'end_char', 'topic_id')}
    if (derived & set(columns) or set(order) != (set(columns) | derived)
            or not required.issubset(columns) or not (derived - {f['segment_text']}).issubset(docs.columns)):
        raise ValueError('精简片段表字段或顺序无效。')
    doc_id = f['document_id']
    if doc_id not in docs or docs[doc_id].isna().any() or docs[doc_id].duplicated().any():
        raise ValueError('项目文档 ID 无效。')
    # Convert each parent row once. Repeated Series lookups can manufacture a
    # separate Python string for every segment of the same long document.
    # Strings are immutable, so sharing these references preserves semantics.
    lookup = docs.set_index(doc_id).to_dict(orient='index')
    segs = pd.DataFrame(records, columns=columns)
    parents, excerpts = [], []
    # Compact references can expand substantially on restoration. Bound the
    # reconstructed UTF-8 payload before materializing repeated columns.
    document_sizes = {}
    reconstructed_size = 0
    for item in segs.to_dict('records'):
        parent = item[doc_id]
        if not isinstance(parent, str) or parent not in lookup:
            raise ValueError('片段指向不存在的文档。')
        doc = lookup[parent]
        text = doc[f['original_text']]
        start, end = item[f['start_char']], item[f['end_char']]
        if (not isinstance(text, str) or isinstance(start, bool) or isinstance(end, bool)
                or not isinstance(start, numbers.Integral) or not isinstance(end, numbers.Integral)
                or not 0 <= start < end <= len(text)):
            raise ValueError('片段位置无效。')
        excerpt = text[start:end]
        if parent not in document_sizes:
            document_sizes[parent] = sum(len(doc[column].encode('utf-8'))
                                         for column in derived - {f['segment_text']}
                                         if isinstance(doc[column], str))
        reconstructed_size += document_sizes[parent] + len(excerpt.encode('utf-8'))
        if reconstructed_size > MAX_RECONSTRUCTED:
            raise ValueError('恢复后的片段字段超过 512 MiB 上限。')
        parents.append(doc)
        excerpts.append(excerpt)
    for column in derived - {f['segment_text']}:
        values = [doc[column] for doc in parents]
        # Pandas' inferred Arrow string column duplicates the UTF-8 payload
        # for every row. An object column instead keeps references to the one
        # immutable string per parent document for substantial text fields.
        if any(isinstance(value, str) and len(value) >= 1024 for value in values):
            segs[column] = pd.Series(values, dtype=object)
        else:
            segs[column] = values
    segs[f['segment_text']] = excerpts
    return segs[order]


def save_project(result):
    visualization = result.get('visualization') or default_visualization()
    _validate(result)
    validate_visualization(result, visualization)
    yearly = result.get('yearly_analysis')
    if yearly is not None:
        from studio.yearly import validate as validate_yearly
        validate_yearly(result, yearly)
    version=YEARLY_FORMAT_VERSION if yearly is not None else FORMAT_VERSION
    info = {'format': 'humanities-policy-studio', 'version': version,
            'saved_at': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
            'name': '人文与政策文本研究项目', 'run_id': result['metadata']['run_id'],
            'scope': 'saved_analysis_results', 'model_included': False}
    content = {'project.json': _json(info), 'records.json': _table(result['rows']),
               'segments.json': _compact_segments(result), 'topics.json': _table(result['topics']),
               'run.json': _json(result['metadata']),
               EDIT_FILE: _json(result.get('topic_edits', {'labels': {}, 'history': []})),
               VISUAL_FILE: _json(visualization)}
    if yearly is not None:
        content[YEARLY_FILE] = _json(yearly)
    if any(len(data) > MAX_ENTRY for data in content.values()) or sum(map(len, content.values())) > MAX_TOTAL:
        raise ValueError('项目内容超出大小上限。')
    manifest = {'algorithm': 'sha256', 'files': {name: {'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                                                for name, data in content.items()}}
    content['manifest.json'] = _json(manifest)
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in content.items():
            archive.writestr(name, data)
    if out.tell() > MAX_ARCHIVE:
        raise ValueError('项目 ZIP 超出 50 MiB。')
    return out.getvalue()


def open_project(raw):
    if not raw or len(raw) > MAX_ARCHIVE:
        raise ValueError('项目 ZIP 为空或超过 50 MiB。')
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entries = archive.infolist()
            names = [i.filename for i in entries]
            if len(names) != len(set(names)):
                raise ValueError('项目包含重复文件名。')
            expected = FILES | {'manifest.json'}
            if EDIT_FILE in names:
                expected = expected | {EDIT_FILE}
            if VISUAL_FILE in names:
                expected = expected | {VISUAL_FILE}
            if YEARLY_FILE in names:
                expected = expected | {YEARLY_FILE}
            if any(name.startswith(('/', '\\')) or '\\' in name or '/' in name or name in ('.', '..') for name in names):
                raise ValueError('项目包含异常文件路径。')
            if set(names) != expected:
                raise ValueError('项目文件缺失或包含不支持的文件。')
            if any(i.is_dir() or stat.S_ISLNK(i.external_attr >> 16) or i.flag_bits & 1 or i.file_size > MAX_ENTRY for i in entries):
                raise ValueError('项目包含异常文件或文件过大。')
            if sum(i.file_size for i in entries) > MAX_TOTAL:
                raise ValueError('项目解压后超过大小上限。')
            data = {name: archive.read(name) for name in names}
        manifest = json.loads(data['manifest.json'])
        if manifest['algorithm'] != 'sha256' or set(manifest['files']) != (set(names) - {'manifest.json'}):
            raise ValueError('项目完整性清单无效。')
        for name in set(names) - {'manifest.json'}:
            item = manifest['files'][name]
            if item['size'] != len(data[name]) or item['sha256'] != hashlib.sha256(data[name]).hexdigest():
                raise ValueError(f'项目完整性校验失败：{name}')
        info = json.loads(data['project.json'])
        if not isinstance(info, dict) or info.get('format') != 'humanities-policy-studio' or info.get('version') not in (1, 2, LEGACY_FORMAT_VERSION, GROUPED_FORMAT_VERSION, FORMAT_VERSION, YEARLY_FORMAT_VERSION):
            raise ValueError('项目格式版本不兼容。')
        if not isinstance(info.get('name'), str) or not isinstance(info.get('saved_at'), str) or info.get('model_included') is not False:
            raise ValueError('项目基本信息不完整或无效。')
        if (info['version'] == 1 and EDIT_FILE in data) or (info['version'] >= 2 and EDIT_FILE not in data) or (info['version'] < 3 and VISUAL_FILE in data) or (info['version'] >= 3 and VISUAL_FILE not in data):
            raise ValueError('项目版本与主题修改记录不一致。')
        if (info['version'] == YEARLY_FORMAT_VERSION) != (YEARLY_FILE in data):
            raise ValueError('项目版本与年度独立建模记录不一致。')
        meta = json.loads(data['run.json'])
        if meta.get('schema_version') != 2 or info.get('run_id') != meta.get('run_id'):
            raise ValueError('项目运行记录版本或标识不兼容。')
        def table(name):
            value = json.loads(data[name])
            columns, records = value['columns'], value['data']
            if not isinstance(columns, list) or len(columns) != len(set(columns)) or not all(isinstance(c,str) for c in columns):
                raise ValueError('项目表格列名无效。')
            if not isinstance(records, list) or any(not isinstance(row,list) or len(row) != len(columns) for row in records):
                raise ValueError('项目表格结构无效。')
            return pd.DataFrame(records, columns=columns)
        docs = table('records.json')
        segments = (_restore_segments(json.loads(data['segments.json']), docs, meta)
                    if info['version'] >= FORMAT_VERSION else table('segments.json'))
        result = {'rows':docs, 'segments':segments,
                  'topics':table('topics.json'), 'metadata':meta, 'project_info':info}
        _validate(result)
        edits = json.loads(data[EDIT_FILE]) if EDIT_FILE in data else {'labels': {}, 'history': []}
        from studio.topic_names import validate_edits
        validate_edits(result, edits)
        result['topic_edits'] = edits
        from studio.visualization import default_visualization, validate_visualization
        visualization = json.loads(data[VISUAL_FILE]) if VISUAL_FILE in data else default_visualization()
        if YEARLY_FILE in data:
            from studio.yearly import validate as validate_yearly
            yearly = json.loads(data[YEARLY_FILE])
            validate_yearly(result, yearly)
            result['yearly_analysis'] = yearly
        validate_visualization(result, visualization)
        result['visualization'] = visualization
        return result
    except json.JSONDecodeError as exc:
        raise ValueError('项目 JSON 内容无法读取。') from exc
    except ValueError:
        raise
    except (zipfile.BadZipFile, OSError, EOFError, RuntimeError, KeyError, TypeError, UnicodeError, OverflowError, AttributeError, IndexError, zlib.error, json.JSONDecodeError) as exc:
        raise ValueError('项目损坏或内容无法读取。') from exc


def _validate(result):
    meta, docs, segs, topics = result['metadata'], result['rows'], result['segments'], result['topics']
    f = meta['generated_fields']
    required = ['record_id','document_id','original_text','status','segment_id','segment_order','start_char','end_char','segment_text','topic_id']
    names = [f[k] for k in required]
    original = meta['original_columns']
    if any(not frame.columns.is_unique for frame in (docs, segs, topics)):
        raise ValueError('项目表格列名重复。')
    if len(names) != len(set(names)) or any(n in original for n in f.values()):
        raise ValueError('项目字段映射冲突。')
    if not set(original + [f[k] for k in ['record_id','document_id','original_text','status']]).issubset(docs.columns):
        raise ValueError('原始文档表缺少必要字段。')
    if not set(original + names[1:]).issubset(segs.columns):
        raise ValueError('片段表缺少必要字段。')
    if not {'Topic','Count','Keywords'}.issubset(topics.columns):
        raise ValueError('主题表缺少必要字段。')
    if docs[f['document_id']].isna().any() or docs[f['document_id']].duplicated().any() or segs[f['segment_id']].isna().any() or segs[f['segment_id']].duplicated().any():
        raise ValueError('项目文档或片段 ID 无效。')
    lookup = docs.set_index(f['document_id'])
    topic_ids = set(topics['Topic'])
    for _, row in segs.iterrows():
        parent = row[f['document_id']]
        if parent not in lookup.index or row[f['topic_id']] not in topic_ids:
            raise ValueError('片段指向不存在的文档或主题。')
        doc = lookup.loc[parent]
        start, end = row[f['start_char']], row[f['end_char']]
        if not isinstance(start, numbers.Integral) or not isinstance(end, numbers.Integral) or not (0 <= start < end <= len(doc[f['original_text']])):
            raise ValueError('片段位置无效。')
        if doc[f['original_text']][start:end] != row[f['segment_text']]:
            raise ValueError('片段与原文位置不一致。')
        if (doc[f['status']] != 'modeled' or row[f['original_text']] != doc[f['original_text']]
                or any(row[col] != doc[col] for col in original)):
            raise ValueError('片段的文档状态或原始字段不一致。')
    for doc_id, parts in segs.groupby(f['document_id'], sort=False):
        original_text = lookup.loc[doc_id, f['original_text']]
        parts = parts.sort_values(f['segment_order'])
        if parts[f['segment_order']].tolist() != list(range(1, len(parts) + 1)) or ''.join(parts[f['segment_text']]) != original_text:
            raise ValueError('片段顺序或原文覆盖不一致。')
        cursor = 0
        for _, part in parts.iterrows():
            if part[f['start_char']] != cursor:
                raise ValueError('片段位置出现间隙或重叠。')
            cursor = part[f['end_char']]
        if cursor != len(original_text):
            raise ValueError('片段位置未覆盖全文。')
    if set(segs[f['document_id']]) != set(docs.loc[docs[f['status']].eq('modeled'), f['document_id']]):
        raise ValueError('已建模文档与片段不一致。')
    if topics['Topic'].duplicated().any() or int(topics['Count'].sum()) != len(segs):
        raise ValueError('主题统计与片段数量不一致。')
    for _, topic in topics.iterrows():
        if int(topic['Count']) != int(segs[f['topic_id']].eq(topic['Topic']).sum()):
            raise ValueError('主题统计与片段分配不一致。')
    if len(docs) != meta['counts']['input'] or len(segs) != meta['counts']['segments']:
        raise ValueError('项目记录数量与运行记录不一致。')
