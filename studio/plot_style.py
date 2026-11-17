"""Shared, data-independent publication chart appearance."""
import re
import unicodedata

from matplotlib.colors import LinearSegmentedColormap


def default_appearance():
    return {'style': 'color', 'font_size': 10, 'show_title': True,
            'show_distribution_values': True, 'show_keyword_values': False,
            'titles': {'distribution': '', 'keywords': '', 'similarity': '', 'hierarchy': ''}}


def resolve_appearance(value=None):
    defaults = default_appearance()
    if value is None:
        return defaults
    if not isinstance(value, dict):
        raise ValueError('图表外观设置无效。')
    config = {**defaults, **value}
    if config['style'] not in ('color', 'mono'):
        raise ValueError('图表样式不受支持。')
    size = config['font_size']
    if isinstance(size, bool) or not isinstance(size, (int, float)) or not 8 <= size <= 18:
        raise ValueError('图表字号须在 8 至 18 磅之间。')
    for key in ('show_title', 'show_distribution_values', 'show_keyword_values'):
        if not isinstance(config[key], bool):
            raise ValueError('图表显示选项无效。')
    titles = config['titles']
    if not isinstance(titles, dict) or set(titles) != set(defaults['titles']):
        raise ValueError('图表标题设置无效。')
    if any(not isinstance(title, str) or len(title) > 120 or
           any(unicodedata.category(c).startswith('C') for c in title) for title in titles.values()):
        raise ValueError('图表标题不能包含控制字符，最多 120 字符。')
    return config


def topic_label(topic_id, name, outlier=None, *, translate=str):
    """Show the ID once when the machine label already starts with it."""
    text = str(name)
    match = re.match(r'^(-?\d+)_', text)
    if match and int(match.group(1)) == int(topic_id):
        text = text[match.end():]
    marker = translate('（离群）') if (int(topic_id) == -1 if outlier is None else outlier) else ''
    return f'{int(topic_id)}{marker} · {text}'


def palette(config):
    if config['style'] == 'mono':
        return {'distribution':'#b8b8b8', 'keywords':'#999999', 'edge':'#4d4d4d',
                'line':'#383838', 'grid':'#e6e6e6', 'text':'#292929', 'cmap':'Greys_r'}
    return {'distribution':'#9dbccc', 'keywords':'#83b5ae', 'edge':'#577b88',
            'line':'#527e88', 'grid':'#e6edef', 'text':'#263a40',
            'cmap':LinearSegmentedColormap.from_list('paper_diverging',['#7796b0','#f8faf9','#68a69e'])}


def set_chart_title(ax, kind, default, config):
    if config['show_title']:
        title = config['titles'][kind].strip() or default
        ax.set_title(title, fontsize=config['font_size']+1, loc='left', pad=9, fontweight='medium')
        return title
    return ''


def style_axes(ax, config):
    colors = palette(config)
    ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    for side in ('bottom','left'):
        ax.spines[side].set_color('#8b999b' if config['style']=='color' else '#686868')
        ax.spines[side].set_linewidth(.55)
    ax.tick_params(axis='both', colors=colors['text'], width=.55, length=3,
                   labelsize=config['font_size']*.88, pad=4)


def check_y_label_spacing(ax):
    """Refuse a crowded export instead of silently overlapping row labels."""
    renderer = ax.figure.canvas.get_renderer()
    boxes = sorted((label.get_window_extent(renderer) for label in ax.get_yticklabels()
                    if label.get_visible() and label.get_text()), key=lambda box: box.y0)
    if any(lower.y1 > upper.y0 + 1 for lower, upper in zip(boxes, boxes[1:])):
        raise ValueError('标签发生重叠，请增大图表高度、减小字号或减少显示项目。')
