"""Traceable segment-level keyword co-occurrence, independent of c-TF-IDF."""
from collections import Counter
from itertools import combinations
import io
import math
from pathlib import Path

import pandas as pd

from studio.keywords import keyword_input, make_settings, vectorizer_for

UNIT = '同一建模文本片段内是否出现；每词每片段最多计一次'
MAX_WORDS = 5000
MAX_WORDS_PER_SEGMENT = 200
MAX_PAIRS = 200000


def saved_tokenizer_settings(metadata):
    """Only accept recorded settings actually reproducible by this runtime."""
    saved = metadata.get('keyword_processing')
    if not isinstance(saved, dict):
        return None
    try:
        current = make_settings(saved['mode'], saved['stopwords_enabled'], saved['stopwords'], saved['user_dictionary'])
        keys = ('component', 'component_version', 'stopwords_applied', 'dictionary_frequency', 'hmm', 'vectorizer_lowercase')
        return current if all(saved.get(k) == current[k] for k in keys) else None
    except (KeyError, TypeError, ValueError):
        return None


def validate_options(options, result):
    if not isinstance(options, dict) or set(options) != {'topic_id', 'include_outliers', 'max_keywords', 'min_frequency', 'min_cooccurrence', 'max_edges'}:
        raise ValueError('共现网络范围或设置不完整。')
    topic = options['topic_id']
    if topic is not None and (isinstance(topic, bool) or not isinstance(topic, int) or topic not in set(result['topics']['Topic'].astype(int))):
        raise ValueError('所选共现主题不存在。')
    if not isinstance(options['include_outliers'], bool) or (topic == -1 and not options['include_outliers']):
        raise ValueError('离群类设置与所选主题不一致。')
    for name, low, high in [('max_keywords', 2, 50), ('min_frequency', 1, 1000000),
                            ('min_cooccurrence', 1, 1000000), ('max_edges', 1, 300)]:
        value = options[name]
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f'共现网络参数 {name} 超出允许范围。')
    return dict(options)


def scoped_segments(result, options):
    f = result['metadata']['generated_fields']
    segments = result['segments']
    if options['topic_id'] is not None:
        segments = segments[segments[f['topic_id']].eq(options['topic_id'])]
    if not options['include_outliers']:
        segments = segments[~segments[f['topic_id']].eq(-1)]
    return segments


def _summarize(token_rows):
    words, pairs = Counter(), Counter()
    for item in token_rows:
        terms = item['tokens']
        if len(terms) > MAX_WORDS_PER_SEGMENT:
            raise ValueError(f'单个片段有效词超过 {MAX_WORDS_PER_SEGMENT} 个，请先缩小范围或调整词表。')
        words.update(terms)
        if len(words) > MAX_WORDS:
            raise ValueError(f'范围内不同词超过 {MAX_WORDS} 个，请缩小范围或调整词表。')
        for pair in combinations(terms, 2):
            pairs[pair] += 1
        if len(pairs) > MAX_PAIRS:
            raise ValueError(f'不同词对超过 {MAX_PAIRS} 个，请缩小范围或调整词表。')
    nodes = [{'keyword': word, 'segment_frequency': count} for word, count in sorted(words.items(), key=lambda x: (-x[1], x[0]))]
    edges = [{'source': a, 'target': b, 'cooccurrence_segments': count}
             for (a, b), count in sorted(pairs.items(), key=lambda x: (-x[1], x[0]))]
    return nodes, edges


def calculate(result, options, keyword_settings, settings_source):
    options = validate_options(options, result)
    if settings_source not in ('project_saved', 'user_selected'):
        raise ValueError('共现分词设置来源无效。')
    canonical = make_settings(keyword_settings['mode'], keyword_settings['stopwords_enabled'],
                              keyword_settings['stopwords'], keyword_settings['user_dictionary'])
    if keyword_settings != canonical:
        raise ValueError('当前分词设置不完整或与可用组件不一致。')
    analyze = vectorizer_for(canonical).build_analyzer()
    f = result['metadata']['generated_fields']
    token_rows = []
    for _, segment in scoped_segments(result, options).iterrows():
        tokens = sorted(set(analyze(keyword_input(segment[f['segment_text']]))))
        token_rows.append({'segment_id': str(segment[f['segment_id']]), 'tokens': tokens})
    nodes, edges = _summarize(token_rows)
    return {'schema_version': 1, 'unit': UNIT, 'directed': False, 'settings': options,
            'keyword_settings': canonical, 'settings_source': settings_source,
            'segment_count': len(token_rows), 'segment_tokens': token_rows, 'nodes': nodes, 'edges': edges}


def validate_cooccurrence(result, obj):
    if obj is None:
        return
    if not isinstance(obj, dict) or obj.get('schema_version') != 1 or obj.get('unit') != UNIT or obj.get('directed') is not False:
        raise ValueError('共现网络记录格式无效。')
    options = validate_options(obj.get('settings'), result)
    if obj.get('settings_source') not in ('project_saved', 'user_selected') or not isinstance(obj.get('keyword_settings'), dict):
        raise ValueError('共现网络分词记录无效。')
    settings = obj['keyword_settings']
    try:
        canonical = make_settings(settings['mode'], settings['stopwords_enabled'], settings['stopwords'], settings['user_dictionary'])
        if (not isinstance(settings.get('component_version'), str) or settings.get('component') != canonical['component']
            or any(settings.get(k) != canonical[k] for k in ('stopwords_applied', 'dictionary_frequency', 'hmm', 'vectorizer_lowercase'))):
            raise ValueError('共现网络分词配置无效。')
    except (KeyError, TypeError) as exc:
        raise ValueError('共现网络分词配置不完整。') from exc
    rows = obj.get('segment_tokens')
    if not isinstance(rows, list) or obj.get('segment_count') != len(rows):
        raise ValueError('共现网络片段记录无效。')
    f = result['metadata']['generated_fields']
    expected = scoped_segments(result, options)[f['segment_id']].astype(str).tolist()
    if len(rows) != len(expected):
        raise ValueError('共现网络范围与片段数量不符。')
    for row, segment_id in zip(rows, expected):
        if not isinstance(row, dict) or set(row) != {'segment_id', 'tokens'} or row['segment_id'] != segment_id:
            raise ValueError('共现网络片段标识无效。')
        terms = row['tokens']
        if not isinstance(terms, list) or any(not isinstance(w, str) or not w or len(w) > 10000 for w in terms) or terms != sorted(set(terms)):
            raise ValueError('共现网络分词记录无效。')
    nodes, edges = _summarize(rows)
    if obj.get('nodes') != nodes or obj.get('edges') != edges:
        raise ValueError('共现网络计数与片段分词记录不一致。')


def tables(obj):
    options = obj['settings']
    chosen = [x['keyword'] for x in obj['nodes'] if x['segment_frequency'] >= options['min_frequency']][:options['max_keywords']]
    chosen_set = set(chosen)
    available = [x for x in obj['edges'] if x['source'] in chosen_set and x['target'] in chosen_set
                 and x['cooccurrence_segments'] >= options['min_cooccurrence']]
    displayed = available[:options['max_edges']]
    if not displayed:
        chosen_set = set()
    chosen_pairs = {(e['source'], e['target']) for e in displayed}
    context = {'unit': UNIT, 'topic_scope': '全部主题' if options['topic_id'] is None else f"主题 {options['topic_id']}",
               'include_outliers': options['include_outliers'], 'segment_count': obj['segment_count'],
               'max_keywords': options['max_keywords'], 'min_frequency': options['min_frequency'],
               'min_cooccurrence': options['min_cooccurrence'], 'max_edges': options['max_edges']}
    nodes = pd.DataFrame([{**n, 'in_graph': n['keyword'] in chosen_set, **context} for n in obj['nodes']],
                         columns=['keyword', 'segment_frequency', 'in_graph', *context])
    edges = pd.DataFrame([{**e, 'in_graph': (e['source'], e['target']) in chosen_pairs, **context} for e in obj['edges']],
                         columns=['source', 'target', 'cooccurrence_segments', 'in_graph', *context])
    return nodes, edges, len(available)


def evidence(result, obj, source, target):
    ids = {item['segment_id'] for item in obj['segment_tokens'] if source in item['tokens'] and target in item['tokens']}
    field = result['metadata']['generated_fields']['segment_id']
    return result['segments'][result['segments'][field].isin(ids)].copy()


def render_network(nodes, edges, width, height, dpi, font_path, appearance, fmt='png', *, translate=str):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.font_manager import FontProperties
    from matplotlib.text import Text
    from studio.plot_style import palette, resolve_appearance
    from studio.visualization import missing_glyphs

    if fmt not in ('png', 'svg') or not font_path or not Path(font_path).is_file():
        raise ValueError('缺少中文字体或图表格式不支持。')
    if not (4 <= width <= 16 and 3 <= height <= 12 and dpi in (150, 300, 600)) or width * height * dpi**2 > 36000000:
        raise ValueError('图表尺寸或分辨率超限。')
    nodes = nodes[nodes['in_graph']].reset_index(drop=True)
    edges = edges[edges['in_graph']].reset_index(drop=True)
    if len(nodes) < 2 or edges.empty:
        raise ValueError('有效词或连线不足，无法绘制网络。')
    cfg = resolve_appearance(appearance); colors = palette(cfg)
    fig = Figure(figsize=(width, height), layout='constrained', facecolor='white'); FigureCanvasAgg(fig)
    ax = fig.subplots(); ax.set_aspect('equal'); ax.axis('off'); ax.set_xticks([]); ax.set_yticks([])
    positions = {row.keyword: (math.cos(math.pi/2 + 2*math.pi*i/len(nodes)), math.sin(math.pi/2 + 2*math.pi*i/len(nodes)))
                 for i, row in enumerate(nodes.itertuples())}
    maximum = max(edges['cooccurrence_segments'])
    for row in edges.itertuples():
        x, y = positions[row.source]; xx, yy = positions[row.target]
        ax.plot([x, xx], [y, yy], color=colors['line'], alpha=.48,
                linewidth=.35 + 1.15 * row.cooccurrence_segments / maximum, zorder=1)
    max_frequency = max(nodes['segment_frequency'])
    for row in nodes.itertuples():
        x, y = positions[row.keyword]
        ax.scatter([x], [y], s=45 + 230*row.segment_frequency/max_frequency, color=colors['keywords'],
                   edgecolor=colors['edge'], linewidth=.5, zorder=2)
        ax.text(x*1.15, y*1.15, row.keyword, ha='left' if x > .15 else 'right' if x < -.15 else 'center',
                va='center', color=colors['text'], fontsize=min(cfg['font_size'], max(6, 11-len(nodes)/12)))
    ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.38, 1.38)
    if cfg['show_title']:
        ax.set_title(translate('关键词共现网络'), loc='left', fontsize=cfg['font_size']+1, pad=6)
    ax.text(.5, .01, translate('节点：出现片段数　·　连线：共现片段数'), transform=ax.transAxes,
            ha='center', fontsize=cfg['font_size']*.8, color=colors['text'])
    for text in fig.findobj(match=Text):
        text.set_fontproperties(FontProperties(fname=font_path, size=text.get_fontsize(), weight=text.get_fontweight()))
    fig.canvas.draw()
    box = fig.get_tightbbox(fig.canvas.get_renderer())
    if box.x0 < -.02 or box.y0 < -.02 or box.x1 > width+.02 or box.y1 > height+.02:
        raise ValueError('网络标签超出画布，请减少关键词或增大图表尺寸。')
    buffer = io.BytesIO()
    if fmt == 'svg':
        from matplotlib import rc_context
        with rc_context({'svg.fonttype':'path'}): fig.savefig(buffer, format='svg', dpi=dpi)
    else:
        fig.savefig(buffer, format='png', dpi=dpi)
    fig.clear()
    return buffer.getvalue(), missing_glyphs(font_path, list(nodes['keyword']) + ['关键词共现网络节点出现片段数连线共现'])
