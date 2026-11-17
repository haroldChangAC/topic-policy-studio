"""Pure chart data and local Matplotlib rendering; no inference or network."""
import io
import math
import textwrap
from functools import lru_cache
from pathlib import Path

import pandas as pd
from matplotlib import font_manager, ft2font
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text

from studio.topic_names import materialize
from studio.plot_style import resolve_appearance, topic_label as format_topic_label, palette, set_chart_title, style_axes, check_y_label_spacing

WEIGHT_SOURCE = 'BERTopic.get_topic：默认 c-TF-IDF 权重'


def default_visualization(weights=None):
    return {'schema_version': 1, 'keyword_weights': weights,
            'weight_source': WEIGHT_SOURCE if weights is not None else None,
            'settings': {'unit': '文本片段', 'show_outliers': True,
                         'width_inches': 8.0, 'height_inches': 5.0, 'dpi': 150,
                         'keyword_topic_id': None}}


def validate_visualization(result, obj):
    if not isinstance(obj, dict) or obj.get('schema_version') != 1 or not isinstance(obj.get('settings'), dict):
        raise ValueError('图表数据或设置格式无效。')
    from studio.similarity import validate_similarity
    validate_similarity(result, obj.get('topic_similarity'))
    from studio.hierarchy import validate_hierarchy
    validate_hierarchy({**result, 'visualization': obj}, obj.get('topic_hierarchy'))
    from studio.cooccurrence import validate_cooccurrence
    validate_cooccurrence(result, obj.get('cooccurrence'))
    from studio.annual import validate_settings as validate_annual_settings
    validate_annual_settings(result, obj.get('annual_panel'))
    from studio.topic_groups import validate as validate_topic_groups
    validate_topic_groups(result, obj.get('topic_groups'))
    from studio.grouped_charts import validate as validate_grouped_charts
    validate_grouped_charts(result, obj, obj.get('grouped_annual_charts'))
    if obj.get('annual_view','original') not in ('original','grouped'):raise ValueError('年度统计视图无效。')
    from studio.annual_heatmap import validate_heatmap
    validate_heatmap(result, obj, obj.get('annual_heatmap'))
    if obj.get('cross_year_links') is not None:
        from studio.cross_year import validate_settings
        validate_settings(result, obj['cross_year_links'])
    if obj.get('cross_year_export') is not None:
        from studio.cross_year_static import validate_export
        validate_export(obj['cross_year_export'])
    settings = obj['settings']
    resolve_appearance(settings.get('appearance'))
    if (settings.get('unit') != '文本片段' or not isinstance(settings.get('show_outliers'), bool)
        or not isinstance(settings.get('width_inches'), (int, float)) or not 4 <= settings['width_inches'] <= 16
        or not isinstance(settings.get('height_inches'), (int, float)) or not 3 <= settings['height_inches'] <= 12
        or settings.get('dpi') not in (150, 300, 600)):
        raise ValueError('图表设置超出允许范围。')
    if settings['width_inches'] * settings['height_inches'] * settings['dpi']**2 > 36_000_000:
        raise ValueError('图表像素过大，请降低尺寸或分辨率。')
    selected = settings.get('keyword_topic_id')
    if selected is not None and (isinstance(selected, bool) or not isinstance(selected, int)
                                 or selected not in set(result['topics']['Topic'].astype(int))):
        raise ValueError('图表所选主题不存在。')
    weights = obj.get('keyword_weights')
    if weights is None:
        if obj.get('weight_source') is not None:
            raise ValueError('缺少关键词权重数据。')
        return
    if not isinstance(weights, list) or obj.get('weight_source') != WEIGHT_SOURCE:
        raise ValueError('关键词权重来源无效。')
    topic_ids = set(result['topics']['Topic'].astype(int))
    seen = set()
    for item in weights:
        if not isinstance(item, dict) or set(item) != {'topic_id', 'rank', 'keyword', 'weight'}:
            raise ValueError('关键词权重记录无效。')
        topic, rank, word, weight = item['topic_id'], item['rank'], item['keyword'], item['weight']
        if (isinstance(topic, bool) or not isinstance(topic, int) or topic not in topic_ids
            or isinstance(rank, bool) or not isinstance(rank, int) or rank < 1
            or not isinstance(word, str) or not isinstance(weight, (int, float))
            or isinstance(weight, bool) or not math.isfinite(weight) or (topic, rank) in seen):
            raise ValueError('关键词权重值或主题编号无效。')
        seen.add((topic, rank))


def distribution_data(result, show_outliers=True):
    view, _ = materialize(result)
    topics = view['topics'][['Topic', 'Original_Label', 'Custom_Name', 'Display_Name', 'Is_Outlier']].copy()
    field = result['metadata']['generated_fields']['topic_id']
    counts = result['segments'][field].value_counts()
    topics['Count'] = topics['Topic'].map(counts).fillna(0).astype(int)
    topics['Unit'] = '文本片段'
    if not show_outliers:
        topics = topics[~topics['Is_Outlier']]
    return topics.sort_values(['Count', 'Topic'], ascending=[False, True]).reset_index(drop=True)


def keyword_data(result, topic_id):
    visual = result.get('visualization') or default_visualization()
    if visual['keyword_weights'] is None:
        return None
    frame = pd.DataFrame([x for x in visual['keyword_weights'] if x['topic_id'] == int(topic_id)
                          and x['keyword'].strip() and math.isfinite(x['weight'])],
                         columns=['topic_id', 'rank', 'keyword', 'weight']).sort_values('rank').reset_index(drop=True)
    frame['weight_source'] = visual['weight_source']
    frame['bertopic_version'] = result['metadata'].get('versions', {}).get('bertopic', '未知')
    return frame


@lru_cache(maxsize=1)
def find_chinese_font():
    preferred = ['/System/Library/Fonts/Hiragino Sans GB.ttc',
                 '/System/Library/Fonts/Supplemental/Songti.ttc',
                 '/System/Library/Fonts/STHeiti Medium.ttc']
    candidates = preferred + font_manager.findSystemFonts()
    for path in dict.fromkeys(candidates):
        if not Path(path).is_file():
            continue
        try:
            font = ft2font.FT2Font(path)
            if all(font.get_char_index(ord(c)) for c in '主题片段数量关键词权重分布文本'):
                return path, font.family_name
        except (OSError, RuntimeError):
            continue
    return None, None


def missing_glyphs(font_path, texts):
    font = ft2font.FT2Font(font_path)
    return sorted({c for text in texts for c in str(text) if not c.isspace() and not font.get_char_index(ord(c))})


def render_chart(frame, kind, width, height, dpi, font_path, *, topic_id=None, topic_label='', fmt='png', appearance=None, translate=str):
    if fmt not in ('png', 'svg') or kind not in ('distribution', 'keywords'):
        raise ValueError('不支持的图表类型或格式。')
    if not font_path or not Path(font_path).is_file():
        raise ValueError('未找到支持中文的本机字体，无法导出中文图表。')
    if not (4 <= width <= 16 and 3 <= height <= 12 and dpi in (150, 300, 600)) or width*height*dpi*dpi > 36_000_000:
        raise ValueError('图表尺寸或分辨率超限；最多 3600 万像素。')
    if frame.empty:
        raise ValueError('没有可绘制的数据。')
    config = resolve_appearance(appearance)
    colors = palette(config)
    fig = Figure(figsize=(width, height), layout='constrained', facecolor='white')
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    if kind == 'distribution':
        labels = [format_topic_label(row.Topic, row.Display_Name, row.Is_Outlier, translate=translate) for row in frame.itertuples()]
        vals = frame['Count'].tolist()
        title = set_chart_title(ax,kind,translate('主题分布'),config)
        ax.set_xlabel(translate('文本片段数'), fontsize=config['font_size'])
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=2))
        draw_values = config['show_distribution_values']
    else:
        labels = frame['keyword'].tolist(); vals = frame['weight'].tolist()
        title = set_chart_title(ax,kind,textwrap.fill(translate('{0}：关键词权重').format(format_topic_label(topic_id, topic_label, translate=translate)), width=36),config)
        ax.set_xlabel(translate('c-TF-IDF 权重（BERTopic.get_topic）'), fontsize=config['font_size'])
        draw_values = config['show_keyword_values']
    labels = [textwrap.fill(str(label), width=max(10, int(width*2.4)), break_long_words=True) for label in labels]
    bars = ax.barh(range(len(labels)), vals, color=colors[kind], edgecolor=colors['edge'], linewidth=.45, height=.48)
    if kind == 'distribution':
        for bar, outlier in zip(bars, frame['Is_Outlier']):
            if outlier: bar.set_hatch('//')
    if draw_values:
        span=max(max((abs(float(v)) for v in vals), default=0), 1e-9)
        for bar, value in zip(bars, vals):
            ax.annotate(str(int(value)) if kind=='distribution' else f'{float(value):.3g}',
                        (bar.get_width(),bar.get_y()+bar.get_height()/2), xytext=(4,0), textcoords='offset points',
                        va='center', fontsize=config['font_size']*.85, color=colors['text'])
        left=min(0.0, min(vals,default=0)); right=max(0.0, max(vals,default=0)) + span*.20
        ax.set_xlim(left=left,right=right)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.invert_yaxis(); ax.grid(axis='x', color=colors['grid'], linewidth=.45)
    style_axes(ax,config);ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0, labelsize=min(config['font_size'],max(6,height*42/max(len(labels),1))))
    for text in fig.findobj(match=Text):
        text.set_fontproperties(FontProperties(fname=font_path, size=text.get_fontsize(), weight=text.get_fontweight()))
    fig.canvas.draw()
    check_y_label_spacing(ax)
    box = fig.get_tightbbox(fig.canvas.get_renderer())
    if box.x0 < -0.02 or box.y0 < -0.02 or box.x1 > width+0.02 or box.y1 > height+0.02:
        raise ValueError('当前尺寸不足以完整显示标签，请增大图表尺寸。')
    out = io.BytesIO()
    if fmt == 'svg':
        from matplotlib import rc_context
        with rc_context({'svg.fonttype': 'path'}):
            fig.savefig(out, format='svg', dpi=dpi)
    else:
        fig.savefig(out, format='png', dpi=dpi)
    fig.clear()
    return out.getvalue(), missing_glyphs(font_path, labels + [title, topic_label, ax.get_xlabel()])
