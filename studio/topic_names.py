"""Human topic names are annotations, never changes to modeling results."""
from datetime import datetime
from zoneinfo import ZoneInfo
import unicodedata

MAX_NAME = 100


def ensure_edits(result):
    result.setdefault('topic_edits', {'labels': {}, 'history': []})
    return result['topic_edits']


def original_label(result, topic_id):
    found = result['topics'].loc[result['topics']['Topic'].eq(topic_id)]
    if len(found) != 1:
        raise ValueError('主题编号不存在或重复。')
    row = found.iloc[0]
    return str(row['Name']) if 'Name' in found.columns else str(row['Keywords'])


def rename(result, topic_id, text):
    if not isinstance(text, str):
        raise ValueError('名称必须是文字。')
    if any(unicodedata.category(c).startswith('C') for c in text):
        raise ValueError('名称不能包含换行或其他控制字符。')
    name = text.strip()
    if len(name) > MAX_NAME:
        raise ValueError(f'名称不能超过 {MAX_NAME} 个字符。')
    original = original_label(result, topic_id)
    if name == original:
        name = ''
    edits = ensure_edits(result)
    key = str(int(topic_id))
    before = edits['labels'].get(key, '')
    if before == name:
        return False
    operation = 'clear' if not name else ('add' if not before else 'change')
    edits['history'].append({'topic_id': int(topic_id), 'before': before or original,
                             'after': name or original, 'previous_custom_name': before,
                             'new_custom_name': name, 'operation': operation,
                             'timestamp': datetime.now(ZoneInfo('Asia/Seoul')).isoformat()})
    if name:
        edits['labels'][key] = name
    else:
        edits['labels'].pop(key, None)
    return True


def materialize_topics(result):
    """Create topic labels without copying the segment table."""
    edits = ensure_edits(result)
    labels = edits['labels']
    topics = result['topics'].copy(deep=True)
    topics['Is_Outlier'] = topics['Topic'].eq(-1)
    topics['Original_Label'] = topics.apply(lambda row: str(row['Name']) if 'Name' in topics.columns else str(row['Keywords']), axis=1)
    topics['Custom_Name'] = topics['Topic'].map(lambda value: labels.get(str(int(value)), ''))
    topics['Display_Name'] = topics.apply(lambda row: row['Custom_Name'] or row['Original_Label'], axis=1)
    return topics


def materialize(result):
    """Derive presentation/export columns without modifying stored source tables."""
    topics = materialize_topics(result)
    f = result['metadata']['generated_fields']
    segments = result['segments'].copy(deep=True)
    occupied = set(segments.columns)
    names = {}
    for key in ('original_label', 'custom_name', 'display_name', 'is_outlier'):
        candidate = f['topic_id'] + '_' + key
        while candidate in occupied:
            candidate = '_' + candidate
        occupied.add(candidate); names[key] = candidate
    by_id = topics.set_index('Topic')
    for key, column in [('original_label','Original_Label'), ('custom_name','Custom_Name'), ('display_name','Display_Name'), ('is_outlier','Is_Outlier')]:
        segments[names[key]] = segments[f['topic_id']].map(by_id[column])
    return {**result, 'topics': topics, 'segments': segments}, names


def validate_edits(result, edits):
    if not isinstance(edits, dict) or set(edits) != {'labels', 'history'} or not isinstance(edits['labels'], dict) or not isinstance(edits['history'], list):
        raise ValueError('主题名称记录格式无效。')
    replay = {'topics': result['topics'], 'topic_edits': {'labels': {}, 'history': []}}
    for event in edits['history']:
        if not isinstance(event, dict) or not isinstance(event.get('topic_id'), int) or not isinstance(event.get('timestamp'), str):
            raise ValueError('主题修改记录无效。')
        from datetime import datetime
        try:
            datetime.fromisoformat(event['timestamp'])
        except ValueError as exc:
            raise ValueError('主题修改时间无效。') from exc
        if not rename(replay, event['topic_id'], event.get('new_custom_name')):
            raise ValueError('主题修改记录包含无效的重复操作。')
        actual = replay['topic_edits']['history'][-1]
        for field in ('before', 'after', 'previous_custom_name', 'new_custom_name', 'operation'):
            if actual[field] != event.get(field):
                raise ValueError('主题修改记录与名称不一致。')
    if replay['topic_edits']['labels'] != edits['labels']:
        raise ValueError('主题名称与修改记录不一致。')
