"""Cosine similarity of BERTopic's actual segment-centroid topic embeddings."""
import io
import math
import textwrap

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text

from studio.topic_names import materialize
from studio.plot_style import resolve_appearance, topic_label, palette, set_chart_title, style_axes, check_y_label_spacing

VECTOR_TYPE = 'BERTopic.topic_embeddings_：主题内原始文本片段嵌入的算术均值'
METHOD = '余弦相似度：点积除以两向量长度之积'


def default_similarity(vectors, dimension):
    ids = sorted(int(row['topic_id']) for row in vectors)
    return {'schema_version': 1, 'vector_type': VECTOR_TYPE, 'method': METHOD,
            'dimension': int(dimension), 'vectors': vectors,
            'settings': {'show_outliers': False, 'show_values': True, 'selected_topics': ids[:20]}}


def validate_similarity(result, obj):
    if obj is None:  # Old projects contain no such data.
        return
    if not isinstance(obj, dict) or obj.get('schema_version') != 1 or obj.get('vector_type') != VECTOR_TYPE or obj.get('method') != METHOD:
        raise ValueError('主题相似度数据的版本或计算口径不受支持。')
    dim = obj.get('dimension')
    if isinstance(dim, bool) or not isinstance(dim, int) or not 1 <= dim <= 10000:
        raise ValueError('主题向量维度无效。')
    ids = set(result['topics']['Topic'].astype(int))
    vectors = obj.get('vectors')
    if not isinstance(vectors, list):
        raise ValueError('主题向量数据无效。')
    seen = set()
    for row in vectors:
        if not isinstance(row, dict) or set(row) != {'topic_id', 'values'}:
            raise ValueError('主题向量记录无效。')
        topic, values = row['topic_id'], row['values']
        if isinstance(topic, bool) or not isinstance(topic, int) or topic not in ids or topic in seen:
            raise ValueError('主题向量编号无效。')
        if values is not None and (not isinstance(values, list) or len(values) != dim or
            any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in values)):
            raise ValueError('主题向量包含无效数值或维度不一致。')
        seen.add(topic)
    settings = obj.get('settings')
    if not isinstance(settings, dict) or not isinstance(settings.get('show_outliers'), bool) or not isinstance(settings.get('show_values'), bool):
        raise ValueError('主题相似度图设置无效。')
    selected = settings.get('selected_topics')
    if (not isinstance(selected, list) or any(isinstance(x, bool) or not isinstance(x, int) or x not in ids for x in selected)
        or len(set(selected)) != len(selected)):
        raise ValueError('主题相似度图所选主题无效。')


def similarity_data(result, selected_ids):
    """Return numeric matrix, names, and explicit exclusions; no imputed values."""
    obj = (result.get('visualization') or {}).get('topic_similarity')
    if obj is None:
        return None, {}, []
    validate_similarity(result, obj)
    view, _ = materialize(result)
    labels = {int(row.Topic): str(row.Display_Name) for row in view['topics'].itertuples()}
    vectors = {row['topic_id']: row['values'] for row in obj['vectors']}
    valid, normalized, excluded = [], [], []
    for topic in selected_ids:
        if topic not in labels:
            raise ValueError('选择了不存在的主题。')
        values = vectors.get(topic)
        if values is None:
            excluded.append({'topic_id': topic, 'reason': '缺少主题向量'})
            continue
        vec = np.asarray(values, dtype=np.float64)
        scale = np.max(np.abs(vec))
        if scale == 0:
            excluded.append({'topic_id': topic, 'reason': '零向量，无法计算余弦相似度'})
            continue
        unit = vec / scale
        norm = np.linalg.norm(unit)
        if not np.isfinite(norm) or norm == 0:
            excluded.append({'topic_id': topic, 'reason': '向量无法计算'})
            continue
        valid.append(topic); normalized.append(unit / norm)
    if not valid:
        return pd.DataFrame(dtype=float), labels, excluded
    unit = np.stack(normalized)
    matrix = np.clip(unit @ unit.T, -1.0, 1.0)
    return pd.DataFrame(matrix, index=valid, columns=valid), labels, excluded


def export_matrix(matrix, labels):
    ids = list(matrix.index)
    result = pd.DataFrame({'topic_id': ids, 'display_name': [labels[i] for i in ids],
                           'vector_type': VECTOR_TYPE, 'method': METHOD})
    for topic in matrix.columns:
        label = f'主题 {topic}{"（离群）" if topic == -1 else ""} · {labels[topic]}'
        result[label] = matrix[topic].to_numpy()
    return result


def render_similarity(matrix, labels, width, height, dpi, font_path, show_values=True, fmt='png', appearance=None, *, translate=str):
    from pathlib import Path
    from studio.visualization import missing_glyphs
    if fmt not in ('png', 'svg') or not font_path or not Path(font_path).is_file():
        raise ValueError('不支持的图形格式或缺少中文字体。')
    if not (4 <= width <= 16 and 3 <= height <= 12 and dpi in (150, 300, 600)) or width*height*dpi*dpi > 36_000_000:
        raise ValueError('图表尺寸或分辨率超限。')
    if matrix.empty or len(matrix) > 30 or list(matrix.index) != list(matrix.columns):
        raise ValueError('请选择 1 至 30 个有效主题。')
    config = resolve_appearance(appearance); colors = palette(config)
    ids = list(matrix.index)
    names = [topic_label(i, labels[i], translate=translate) for i in ids]
    wrapped = [textwrap.fill(name, width=18, break_long_words=True) for name in names]
    fig = Figure(figsize=(width, height), layout='constrained')
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax.imshow(matrix.to_numpy(), norm=norm, cmap=colors['cmap'], interpolation='nearest')
    ax.set_xticks(range(len(ids)), wrapped, rotation=45, ha='right', rotation_mode='anchor')
    ax.set_yticks(range(len(ids)), wrapped)
    style_axes(ax,config)
    ax.tick_params(labelsize=max(6, min(config['font_size']*.9, 90 / max(len(ids), 1))), length=0)
    title=set_chart_title(ax,'similarity',translate('主题向量余弦相似度'),config)
    ax.set_xticks(np.arange(-.5,len(ids),1),minor=True); ax.set_yticks(np.arange(-.5,len(ids),1),minor=True)
    ax.grid(which='minor',color='white',linewidth=.5);ax.tick_params(which='minor',bottom=False,left=False)
    bar = fig.colorbar(im, ax=ax, label=translate('余弦相似度（-1 至 1；0 为中点）'), shrink=.8)
    bar.set_ticks([-1, -.5, 0, .5, 1], labels=['-1.0', '-0.5', '0.0', '0.5', '1.0'])
    bar.outline.set_linewidth(.5);bar.ax.tick_params(labelsize=config['font_size']*.8,width=.5,length=3)
    bar.ax.yaxis.label.set_size(config['font_size']*.9)
    if show_values:
        for i in range(len(ids)):
            for j in range(len(ids)):
                value = matrix.iloc[i, j]
                rgba=im.cmap(im.norm(value)); luminance=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
                ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                        fontsize=max(5, min(config['font_size']*.85, 90/max(len(ids), 1))), color='white' if luminance < .48 else colors['text'])
    for text in fig.findobj(match=Text):
        text.set_fontproperties(FontProperties(fname=font_path, size=text.get_fontsize(), weight=text.get_fontweight()))
    fig.canvas.draw()
    check_y_label_spacing(ax)
    box = fig.get_tightbbox(fig.canvas.get_renderer())
    if box.x0 < -.02 or box.y0 < -.02 or box.x1 > width+.02 or box.y1 > height+.02:
        raise ValueError('当前尺寸不足以完整显示标签，请增大图表尺寸或减少主题。')
    out = io.BytesIO()
    if fmt == 'svg':
        from matplotlib import rc_context
        with rc_context({'svg.fonttype': 'path'}):
            fig.savefig(out, format='svg', dpi=dpi)
    else:
        fig.savefig(out, format='png', dpi=dpi)
    fig.clear()
    return out.getvalue(), missing_glyphs(font_path, names + [title, '主题向量余弦相似度离群为中点'])
