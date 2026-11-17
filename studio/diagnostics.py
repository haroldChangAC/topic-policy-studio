"""Diagnostics derived from saved assignments, without guessing historical parameters."""
import pandas as pd


def summarize(result):
    fields = result['metadata']['generated_fields']
    documents, segments = result['rows'], result['segments']
    ids = segments[fields['topic_id']].astype(int)
    counts = ids.value_counts()
    n = len(segments)
    topics = result['topics'].copy()
    topics['Count'] = topics['Topic'].map(counts).fillna(0).astype(int)
    topics['Segment_Share'] = topics['Count'] / n if n else 0.0
    totals = {'original_documents': len(documents),
              'valid_documents': int(documents[fields['status']].eq('modeled').sum()),
              'segments': n, 'topics': len(set(ids) - {-1}),
              'outliers': int(counts.get(-1, 0)),
              'outlier_share': int(counts.get(-1, 0)) / n if n else 0.0}
    return totals, topics


def parameter_record(metadata):
    """Report absent historical fields as unknown, never fill from today's defaults."""
    missing = '历史记录未保存'
    params = metadata.get('parameters') or {}
    umap = params.get('umap') or {}
    hdbscan = params.get('hdbscan') or {}
    return {'随机种子': params.get('random_state', missing),
            'UMAP 邻居数': umap.get('n_neighbors', missing),
            'UMAP 最小距离': umap.get('min_dist', missing),
            'HDBSCAN 最小聚类规模': hdbscan.get('min_cluster_size', missing),
            'HDBSCAN 最小样本数': hdbscan.get('min_samples', missing),
            '用户输入': metadata.get('clustering_settings_requested', missing)}
