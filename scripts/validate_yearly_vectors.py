"""Recompute one saved topic centroid per year with the locked offline encoder."""
import json
from unittest.mock import patch
import numpy as np
from studio.project import open_project
from studio.modeling import load_encoder
from studio.runtime import ROOT, configure

configure()
out = ROOT / 'outputs/yearly_trial'
result = open_project((out / 'six_reports_yearly.hpts.zip').read_bytes())
f = result['metadata']['generated_fields']
checks = []
with patch('socket.socket.connect', side_effect=AssertionError('network prohibited')):
    encoder = load_encoder('cpu')
    for year in result['yearly_analysis']['results']:
        if year['status'] != 'completed':
            continue
        ids = [item['topic_id'] for item in year['topics'] if item['topic_id'] != -1]
        if not ids:
            continue
        topic = ids[0]
        vector = next(item['values'] for item in year['topic_vectors'] if item['topic_id'] == topic)
        if vector is None:
            raise ValueError(f"{year['year']} 年主题 {topic} 缺少向量")
        segment_ids = {item['segment_id'] for item in year['assignments'] if item['topic_id'] == topic}
        texts = result['segments'].loc[result['segments'][f['segment_id']].isin(segment_ids), f['segment_text']].tolist()
        mean = encoder.encode(texts, batch_size=16, show_progress_bar=False, convert_to_numpy=True).mean(axis=0)
        error = float(np.max(np.abs(mean - np.asarray(vector))))
        if error >= 1e-5:
            raise AssertionError(f"{year['year']} 年主题向量与片段均值不一致：{error}")
        checks.append({'year': year['year'], 'topic_id': topic, 'segments': len(texts),
                       'dimension': len(vector), 'max_absolute_difference': error})
(out / 'vector_check.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(checks, ensure_ascii=False, indent=2))
