"""Average-linkage hierarchy using saved BERTopic topic vectors only."""
import io
import importlib.metadata
import math
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter
from scipy.cluster.hierarchy import dendrogram, linkage, is_valid_linkage
from scipy.spatial.distance import squareform

from studio.similarity import VECTOR_TYPE, similarity_data
from studio.plot_style import resolve_appearance, topic_label, palette, set_chart_title, style_axes, check_y_label_spacing

DISTANCE = '1 - 主题向量余弦相似度'
LINKAGE = 'average（平均连接法）'
MAX_TOPICS = 30


def calculate_hierarchy(result, selected_topics):
    """Each leaf index maps to a sorted topic ID; cluster n+i is merge i."""
    selected = sorted(selected_topics)
    if len(selected) != len(set(selected)):
        raise ValueError('层级图主题不能重复。')
    if len(selected) > MAX_TOPICS:
        raise ValueError(f'层级计算最多选择 {MAX_TOPICS} 个主题，请减少选择。')
    matrix, _, excluded = similarity_data(result, selected)
    if matrix is None:
        return None
    ids = [int(i) for i in matrix.index]
    if len(ids) < 2:
        merges = []
    else:
        distances = np.clip(1.0 - matrix.to_numpy(dtype=np.float64), 0.0, 2.0)
        distances = (distances + distances.T) / 2.0
        np.fill_diagonal(distances, 0.0)
        merges = linkage(squareform(distances, checks=True), method='average').tolist()
    return {'leaf_topics': ids, 'linkage_matrix': merges, 'excluded': excluded}


def update_hierarchy(result, selected_topics, show_outliers=False):
    selected = sorted(int(x) for x in selected_topics)
    if not show_outliers and -1 in selected:
        raise ValueError('离群主题未启用，不能纳入层级图。')
    tree = calculate_hierarchy(result, selected)
    if tree is None:
        return None
    state = {'schema_version': 1, 'vector_type': VECTOR_TYPE, 'distance': DISTANCE,
             'linkage_method': LINKAGE, 'scipy_version': importlib.metadata.version('scipy'),
             'settings': {'show_outliers': bool(show_outliers), 'selected_topics': selected},
             'result': tree}
    result['visualization']['topic_hierarchy'] = state
    result['metadata']['topic_hierarchy'] = {k: state[k] for k in ('vector_type','distance','linkage_method','scipy_version','settings')}
    result['metadata']['topic_hierarchy']['leaf_topics'] = tree['leaf_topics']
    result['metadata']['topic_hierarchy']['excluded'] = tree['excluded']
    return state


def ensure_hierarchy(result):
    visual = result.get('visualization') or {}
    if visual.get('topic_similarity') is None:
        return None
    if visual.get('topic_hierarchy') is not None:
        return visual['topic_hierarchy']
    ids = sorted(int(i) for i in result['topics']['Topic'] if int(i) != -1)[:20]
    return update_hierarchy(result, ids)


def validate_hierarchy(result, obj):
    if obj is None:
        return
    if not isinstance(obj, dict) or obj.get('schema_version') != 1 or obj.get('vector_type') != VECTOR_TYPE or obj.get('distance') != DISTANCE or obj.get('linkage_method') != LINKAGE:
        raise ValueError('主题层级数据的版本或计算口径无效。')
    if not isinstance(obj.get('scipy_version'), str) or not obj['scipy_version']:
        raise ValueError('主题层级计算的软件版本缺失。')
    settings, tree = obj.get('settings'), obj.get('result')
    if not isinstance(settings, dict) or not isinstance(settings.get('show_outliers'), bool) or not isinstance(settings.get('selected_topics'), list):
        raise ValueError('主题层级设置无效。')
    selected = settings['selected_topics']
    ids = set(result['topics']['Topic'].astype(int))
    if (any(isinstance(i, bool) or not isinstance(i, int) or i not in ids for i in selected)
        or selected != sorted(set(selected)) or len(selected) > MAX_TOPICS
        or (-1 in selected and not settings['show_outliers'])):
        raise ValueError('主题层级所选主题无效。')
    if not isinstance(tree, dict) or set(tree) != {'leaf_topics','linkage_matrix','excluded'}:
        raise ValueError('主题层级结果格式无效。')
    matrix, _, excluded = similarity_data(result, selected)
    if matrix is None or tree['leaf_topics'] != list(matrix.index) or tree['excluded'] != excluded:
        raise ValueError('主题层级叶节点与已保存的主题向量不一致。')
    n = len(matrix)
    rows = tree['linkage_matrix']
    if not isinstance(rows, list) or len(rows) != max(0, n-1):
        raise ValueError('主题层级合并记录数量无效。')
    if n < 2:
        return
    if any(not isinstance(row, list) or len(row) != 4 or any(isinstance(v, bool) or not isinstance(v, (int,float)) or not math.isfinite(v) for v in row) for row in rows):
        raise ValueError('主题层级合并记录含无效数值。')
    z = np.array(rows, dtype=float)
    if not is_valid_linkage(z, warning=False, throw=False):
        raise ValueError('主题层级合并节点无效。')
    # Verify the stored distances and average-linkage decisions without relying
    # on tie-breaking behavior of the currently installed SciPy version.
    distance = np.clip(1.0 - matrix.to_numpy(dtype=float), 0.0, 2.0)
    clusters = {i: [i] for i in range(n)}
    for step, row in enumerate(z):
        a, b = int(row[0]), int(row[1])
        if row[0] != a or row[1] != b or a not in clusters or b not in clusters or a == b:
            raise ValueError('主题层级节点映射无效。')
        members_a, members_b = clusters[a], clusters[b]
        value = float(distance[np.ix_(members_a, members_b)].mean())
        all_pairs = [float(distance[np.ix_(clusters[x],clusters[y])].mean())
                     for index,x in enumerate(clusters) for y in list(clusters)[index+1:]]
        if not np.isclose(row[2], value, atol=1e-10, rtol=1e-10) or value > min(all_pairs)+1e-10 or row[3] != len(members_a)+len(members_b):
            raise ValueError('主题层级距离或合并数量与向量不一致。')
        del clusters[a]; del clusters[b]
        clusters[n+step] = members_a + members_b


def hierarchy_table(state, labels):
    tree = state['result']; ids = tree['leaf_topics']; n = len(ids)
    columns = ['node_id','node_type','topic_id','display_name','left_node','right_node',
               'merge_distance','topic_count','contained_topic_ids','vector_type','distance_method','linkage_method','scipy_version',
               'include_outliers','selected_topic_ids']
    tail = [VECTOR_TYPE,DISTANCE,LINKAGE,state['scipy_version'],state['settings']['show_outliers'],
            ','.join(map(str,state['settings']['selected_topics']))]
    rows, members = [], {}
    for node, topic in enumerate(ids):
        members[node] = [topic]
        rows.append([node,'leaf',topic,labels[topic],None,None,None,1,str(topic),*tail])
    for step,(left,right,value,count) in enumerate(tree['linkage_matrix']):
        node=n+step;left=int(left);right=int(right)
        members[node]=sorted(members[left]+members[right])
        rows.append([node,'merge',None,'',left,right,value,int(count),','.join(map(str,members[node])),*tail])
    frame = pd.DataFrame(rows,columns=columns)
    for name in ('topic_id','left_node','right_node'):
        frame[name] = pd.array(frame[name], dtype='Int64')
    return frame


def render_hierarchy(state, labels, width, height, dpi, font_path, fmt='png', appearance=None, *, translate=str):
    from studio.visualization import missing_glyphs
    if fmt not in ('png','svg') or not font_path or not Path(font_path).is_file():
        raise ValueError('不支持的图形格式或缺少中文字体。')
    if not (4 <= width <= 16 and 3 <= height <= 12 and dpi in (150,300,600)) or width*height*dpi*dpi > 36_000_000:
        raise ValueError('图表尺寸或分辨率超限。')
    ids = state['result']['leaf_topics']
    if len(ids) < 2 or len(ids) > MAX_TOPICS:
        raise ValueError('层级图需要 2 至 30 个有效主题。')
    config=resolve_appearance(appearance); colors=palette(config)
    names = [topic_label(i, labels[i], translate=translate) for i in ids]
    wrapped = [textwrap.fill(name,width=22,break_long_words=True) for name in names]
    fig=Figure(figsize=(width,height),layout='constrained');FigureCanvasAgg(fig);ax=fig.subplots()
    dendrogram(np.asarray(state['result']['linkage_matrix'],dtype=float),labels=wrapped,orientation='right',
               color_threshold=0,above_threshold_color=colors['line'],leaf_font_size=max(6,min(config['font_size'],height*38/len(ids))),ax=ax)
    for collection in ax.collections:
        collection.set_linewidth(.8)
    title=set_chart_title(ax,'hierarchy',translate('主题向量层级图'),config)
    ax.set_xlabel(translate('距离（1 - 余弦相似度） · 平均连接法'),fontsize=config['font_size'])
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5,min_n_ticks=3))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value,_: f'{value:g}'))
    style_axes(ax,config);ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y',length=0,pad=7,labelsize=max(6,min(config['font_size'],height*38/len(ids))))
    ax.grid(axis='x',color=colors['grid'],linewidth=.4)
    for text in fig.findobj(match=Text):
        text.set_fontproperties(FontProperties(fname=font_path,size=text.get_fontsize(),weight=text.get_fontweight()))
    fig.canvas.draw();check_y_label_spacing(ax);box=fig.get_tightbbox(fig.canvas.get_renderer())
    if box.x0 < -.02 or box.y0 < -.02 or box.x1 > width+.02 or box.y1 > height+.02:
        raise ValueError('当前尺寸不足以完整显示标签，请增大图表尺寸或减少主题。')
    output=io.BytesIO()
    if fmt == 'svg':
        from matplotlib import rc_context
        with rc_context({'svg.fonttype':'path'}):fig.savefig(output,format='svg',dpi=dpi)
    else:
        fig.savefig(output,format='png',dpi=dpi)
    fig.clear()
    return output.getvalue(), missing_glyphs(font_path,names+[title,'主题向量层级图距离余弦相似度离群平均连接法'])
