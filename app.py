"""CSV to traceable, offline BERTopic results."""
from studio.i18n import diagnostic, tr, localize_options
from studio.ui_layout import anchor, help_menu, section_links, source_text
import json
import hashlib
import threading

from studio.runtime import ROOT, configure

configure()

import streamlit as st
from studio.corpus import prepare, read_csv, read_txt, read_xlsx, xlsx_sheets, missing_text_formulae
from studio.modeling import load_encoder, validate_clustering_settings
from studio.diagnostics import summarize, parameter_record
from studio.segmentation import plan_segments
from studio.workflow import analyze, persist, csv_bytes
from studio.project import open_project
from studio.topic_names import ensure_edits, rename, materialize, original_label
from studio.visualization import (default_visualization, distribution_data, keyword_data,
                                  find_chinese_font, render_chart, WEIGHT_SOURCE)
from studio.keywords import decode_word_file, make_settings, parse_word_list, preview
from studio.similarity import similarity_data, export_matrix, render_similarity, VECTOR_TYPE, METHOD
from studio.hierarchy import (DISTANCE, LINKAGE, MAX_TOPICS, ensure_hierarchy,
                              update_hierarchy, hierarchy_table, render_hierarchy)
from studio.plot_style import resolve_appearance
from studio.cooccurrence import (calculate as calculate_cooccurrence, evidence as cooccurrence_evidence,
                                 render_network, saved_tokenizer_settings, tables as cooccurrence_tables, UNIT as CO_UNIT)

st.set_page_config(page_title="人文与政策文本研究工作台", page_icon="📚", layout="wide")
st.sidebar.title('Humanities & Policy Text Studio')
st.sidebar.radio('语言 / Language', ['中文', 'English'], key='studio_language', horizontal=True)
anchor('top')
st.caption(tr('HUMANITIES & POLICY TEXT STUDIO · 本地研究工作台'))
st.title(tr('人文与政策文本研究工作台'))
st.write(tr('主题分析使用本地模型；基础政策评价由研究者人工编码。'))


@st.cache_resource
def encoder_resource(device):
    return load_encoder(device), threading.Lock()


def show_diagnostics(result):
    stats, topics = summarize(result)
    st.subheader(tr('分析结果诊断'))
    a, b, c = st.columns(3)
    a.metric(tr('原始文档数'), stats['original_documents'])
    b.metric(tr('建模片段数'), stats['segments'])
    c.metric(tr('有效文档数'), stats['valid_documents'])
    a, b, c = st.columns(3)
    a.metric(tr('非离群主题数'), stats['topics'])
    b.metric(tr('离群片段数'), stats['outliers'])
    c.metric(tr('离群片段占比'), f"{stats['outlier_share']:.2%}")
    st.caption(tr('比例以全部建模片段为分母；一篇文档可产生多个片段。离群比例或主题数量不自动代表分析质量。'))
    st.dataframe(topics[['Topic', 'Is_Outlier', 'Count', 'Segment_Share', 'Name', 'Keywords', 'Custom_Name', 'Display_Name']],
                 hide_index=True, width='stretch', column_config={'Segment_Share': st.column_config.NumberColumn(tr('片段占比'), format='%.2f')})
    st.caption(tr('片段占比为 0–1 的比例；主题关键词和名称供结合下方原文判断。'))
    with st.expander(tr('聚类参数与随机种子')):
        st.json(parameter_record(result['metadata']))
        st.caption(tr('固定种子和环境有助于复现；不同设备、依赖或运行环境不保证完全一致。'))
    return stats


def topic_editor(result, scope):
    edits = ensure_edits(result)
    st.subheader(tr('手动设置主题名称'))
    st.caption(tr('只修改显示名称，不改变主题编号、关键词或片段分配；-1 始终是离群主题。空名称会清除自定义名称。修改仅保留在当前会话，关闭或刷新前请下载新项目 ZIP。'))
    options = sorted(result['topics']['Topic'].astype(int).tolist())
    topic_id = st.selectbox(tr('选择要命名的主题（-1 为离群类）'), options, key=f"edit_topic_{scope}")
    st.write(tr('原始标签：{0}', f'{original_label(result, topic_id)}'))
    original = result['topics'].set_index('Topic').loc[topic_id]
    st.write(tr('原始关键词：{0}', f"{original['Keywords']}"))
    f = result['metadata']['generated_fields']
    st.dataframe(result['segments'][result['segments'][f['topic_id']].eq(topic_id)][[f['document_id'],f['segment_text']]], hide_index=True, width="stretch")
    key = f"name_input_{scope}_{topic_id}"
    with st.form(f"name_form_{scope}_{topic_id}"):
        name = st.text_input(tr('自定义名称（最多 100 字符）'), value=edits['labels'].get(str(topic_id), ''), key=key)
        save = st.form_submit_button(tr('保存名称'), key=f'name_save_{scope}_{topic_id}')
        clear = st.form_submit_button(tr('清除自定义名称'), key=f'name_clear_{scope}_{topic_id}')
    if save or clear:
        try:
            changed = rename(result, topic_id, '' if clear else name)
            if changed:
                st.session_state.pop(key, None)
                st.rerun()
            st.info(tr('名称未变化，没有新增修改记录。'))
        except ValueError as exc:
            st.error(diagnostic(exc))
    with st.expander(tr('主题名称修改记录')):
        if edits['history']:
            st.dataframe(edits['history'], hide_index=True, width="stretch")
        else:
            st.write(tr('尚无修改记录。'))


def show_charts(result, scope):
    anchor('charts')
    st.subheader(tr('基础可视化'))
    visual = result.setdefault('visualization', default_visualization())
    settings = visual['settings']
    font_path, family = find_chinese_font()
    if font_path:
        st.caption(tr('本机绘图字体：{0}；中文显示仍取决于字体对具体字符的覆盖。', f'{family}'))
    else:
        st.warning(tr('未找到支持中文的本机字体；可查看图表数据，但无法可靠导出中文图。'))
    st.caption(tr('图表设置会随新下载的项目保存；修改后请重新下载项目 ZIP。'))
    a,b,c,d = st.columns(4)
    show = a.checkbox(tr('显示离群主题 -1'), value=settings['show_outliers'], key=f'chart_outliers_{scope}')
    width = b.number_input(tr('宽度（英寸）'), min_value=4.0, max_value=16.0, step=0.5,
                           value=float(settings['width_inches']), key=f'chart_width_{scope}')
    height = c.number_input(tr('高度（英寸）'), min_value=3.0, max_value=12.0, step=0.5,
                            value=float(settings['height_inches']), key=f'chart_height_{scope}')
    dpi = d.selectbox(tr('PNG 分辨率（DPI）'), [150,300,600], index=[150,300,600].index(settings['dpi']), key=f'chart_dpi_{scope}')
    appearance = resolve_appearance(settings.get('appearance'))
    a,b,c = st.columns(3)
    style = a.selectbox(tr('论文图表样式'), ['论文彩色','论文黑白'], format_func=localize_options(), index=0 if appearance['style']=='color' else 1, key=f'paper_style_{scope}')
    font_size = b.number_input(tr('基础字号（磅）'), min_value=8, max_value=18, value=int(appearance['font_size']), key=f'paper_font_{scope}')
    show_title = c.checkbox(tr('显示图内标题'), value=appearance['show_title'], key=f'paper_title_{scope}')
    a,b=st.columns(2)
    dist_values=a.checkbox(tr('主题分布显示数量'),value=appearance['show_distribution_values'],key=f'paper_dist_values_{scope}')
    keyword_values=b.checkbox(tr('关键词权重显示数值'),value=appearance['show_keyword_values'],key=f'paper_keyword_values_{scope}')
    with st.expander(tr('自定义图内标题（留空使用默认标题）')):
        titles={}
        for key,label in [('distribution','主题分布标题'),('keywords','关键词权重标题'),('similarity','相似度标题'),('hierarchy','层级图标题')]:
            titles[key]=st.text_input(tr(label),value=appearance['titles'][key],max_chars=120,key=f'paper_title_{key}_{scope}')
    try:
        appearance=resolve_appearance({'style':'color' if style=='论文彩色' else 'mono','font_size':int(font_size),
                                       'show_title':show_title,'show_distribution_values':dist_values,
                                       'show_keyword_values':keyword_values,'titles':titles})
    except ValueError as exc:
        st.warning(tr('图表设置无效：{0}', diagnostic(exc)))
        return
    if width * height * dpi**2 > 36_000_000:
        st.warning(tr('设置超过 3600 万像素，请缩小尺寸或降低 DPI。未保存超限设置。'))
        return
    settings.update({'unit':'文本片段','show_outliers':show,'width_inches':width,'height_inches':height,'dpi':dpi,'appearance':appearance})
    st.caption(tr('PNG 导出：{0} × {1} 像素，{2} × {3} 英寸，{4} DPI。', f'{int(width * dpi)}', f'{int(height * dpi)}', f'{width:g}', f'{height:g}', f'{dpi}'))
    st.caption(tr('SVG 中文文字以轮廓保存，显示稳定，但不能直接作为文字编辑。'))
    dist = distribution_data(result, show)
    st.markdown(tr('#### 主题分布'))
    st.caption(tr('统计单位：文本片段；一篇文档可能对应多个片段和主题。'))
    if not dist.empty:
        show_chart_output(dist, 'distribution', width, height, dpi, font_path, scope, appearance=appearance)
    else:
        st.info(tr('当前筛选下没有主题；可尝试显示离群主题。'))
    st.dataframe(dist, hide_index=True, width="stretch")
    st.markdown(tr('#### 主题关键词权重'))
    ids = sorted(result['topics']['Topic'].astype(int).tolist())
    selected = settings.get('keyword_topic_id')
    topic = st.selectbox(tr('选择关键词所属主题'), ids, index=ids.index(selected) if selected in ids else 0,
                         key=f'chart_topic_{scope}')
    settings['keyword_topic_id'] = int(topic)
    weights = keyword_data(result, topic)
    if topic == -1:
        st.warning(tr('－1 是离群片段集合，关键词不代表一个普通主题。'))
    if weights is None:
        st.warning(tr('此项目未保存真实关键词权重，无法绘制权重图；不会根据排序估算或重新训练。主题分布仍可使用。'))
    elif weights.empty:
        st.info(tr('该主题没有有效关键词或权重，无法绘制权重图。'))
    else:
        view, _ = materialize(result)
        label = view['topics'].set_index('Topic').loc[topic, 'Display_Name']
        st.caption(tr('来源：{0}；这是词在主题表示中的权重，不是概率。', tr(visual['weight_source'])))
        show_chart_output(weights, 'keywords', width, height, dpi, font_path, scope, topic, label, appearance)
        st.dataframe(weights, hide_index=True, width="stretch")
    show_similarity(result, scope, width, height, dpi, font_path, appearance)
    show_hierarchy(result, scope, width, height, dpi, font_path, appearance)
    show_cooccurrence(result, scope, width, height, dpi, font_path, appearance)
    settings['font_family'] = family
    chart_record = {'settings': dict(settings), 'weight_source': visual['weight_source'],
                    'model_id': result['metadata']['model_id'], 'model_revision': result['metadata']['model_revision'],
                    'data_unit': '文本片段', 'keyword_weights_available': visual['keyword_weights'] is not None}
    similarity = visual.get('topic_similarity')
    chart_record['topic_similarity'] = ({k: similarity[k] for k in ('vector_type','method','dimension','settings')}
                                       if similarity is not None else None)
    chart_record['topic_hierarchy'] = visual.get('topic_hierarchy')
    st.download_button(tr('下载图表设置 JSON'), json.dumps(chart_record, ensure_ascii=False, indent=2).encode('utf-8'),
                       'chart_settings.json', 'application/json', key=f'chart_settings_{scope}')


def show_cooccurrence(result, scope, width, height, dpi, font_path, appearance):
    st.markdown(tr('#### 关键词共现网络'))
    st.caption(tr('单位：同一建模文本片段；每词及每个不同词对在一个片段中最多计一次。无向、无自连线；不是 c-TF-IDF 权重，不自动表示因果或语义等价。'))
    st.caption(tr('节点面积依据出现片段数缩放，连线粗细依据共现片段数缩放。完整统计表的 in_graph 标记按参数选入图的项目；若图像导出失败则未生成图片。修改设置后须点击重新计算。'))
    st.caption(tr('重新计算替换当前会话中的网络结果；若需保留旧网络，请先下载项目 ZIP。保存后重新打开只查看已有结果，不重新训练。'))
    visual = result['visualization']
    previous = visual.get('cooccurrence')
    old_options = previous['settings'] if previous else {}
    ids = sorted(result['topics']['Topic'].astype(int).tolist())
    available = saved_tokenizer_settings(result['metadata'])
    sources = ['使用项目记录的分词设置', '指定当前分词设置'] if available else ['指定当前分词设置']
    if not available:
        st.warning(tr('该项目未保存完整分词设置，或记录的组件版本与当前环境不一致。请选择可复现的当前处理方式；不能推断与历史分析相同。'))
    source_index = 1 if previous and previous['settings_source'] == 'user_selected' and available else 0
    source = st.selectbox(tr('共现分词设置来源'), sources, format_func=localize_options(), index=source_index, key=f'co_source_{scope}')
    if source == '使用项目记录的分词设置':
        word_settings, source_id, confirmed = available, 'project_saved', True
    else:
        base = previous['keyword_settings'] if previous and previous['settings_source'] == 'user_selected' else available
        base = base or make_settings()
        modes = {'原有方式': 'legacy', '中文分词': 'chinese', '英文分词': 'english'}
        selected_mode = st.selectbox(tr('共现分词方式'), list(modes), format_func=localize_options(), index=list(modes.values()).index(base['mode']),
                                     key=f'co_mode_{scope}')
        stop_enabled = st.checkbox(tr('共现启用停用词'), value=base['stopwords_enabled'], key=f'co_stop_enabled_{scope}')
        stop_text = st.text_area(tr('共现停用词（每行一个）'), value='\n'.join(base['stopwords']), key=f'co_stop_{scope}')
        dict_text = st.text_area(tr('共现自定义词典（每行一个）'), value='\n'.join(base['user_dictionary']), key=f'co_dictionary_{scope}', disabled=selected_mode == '英文分词')
        try:
            word_settings = make_settings(modes[selected_mode], stop_enabled, parse_word_list(stop_text),
                                          [] if selected_mode == '英文分词' else parse_word_list(dict_text))
        except ValueError as exc:
            st.error(tr('共现词表无效：{0}', diagnostic(exc)))
            return
        source_id = 'user_selected'
        confirmed = (st.checkbox(tr('确认使用上述当前分词和词表计算，不推断历史设置'), key=f'co_confirm_{scope}')
                     if not available else True)
    with st.expander(tr('查看本次选用的分词设置及词表')):
        st.json(word_settings)
    include = st.checkbox(tr('共现范围包含离群主题 -1'), value=old_options.get('include_outliers', False), key=f'co_outliers_{scope}')
    possible_ids = [x for x in ids if include or x != -1]
    choices = [None] + possible_ids
    old_topic = old_options.get('topic_id')
    topic = st.selectbox(tr('共现统计范围'), choices, index=choices.index(old_topic) if old_topic in choices else 0,
                         format_func=localize_options(lambda x: tr('全部主题（按离群开关筛选）') if x is None else tr('主题 {0}', f'{x}') + (tr('（离群）') if x == -1 else '')),
                         key=f'co_topic_{scope}')
    a,b,c,d = st.columns(4)
    maximum = a.number_input(tr('最多展示关键词'), min_value=2, max_value=50, value=old_options.get('max_keywords', 20), step=1, key=f'co_max_words_{scope}')
    frequency = b.number_input(tr('最小片段词频'), min_value=1, max_value=1000000, value=old_options.get('min_frequency', 1), step=1, key=f'co_min_freq_{scope}')
    minimum = c.number_input(tr('最小共现片段数'), min_value=1, max_value=1000000, value=old_options.get('min_cooccurrence', 1), step=1, key=f'co_min_pair_{scope}')
    max_edges = d.number_input(tr('图中最多连线'), min_value=1, max_value=300, value=old_options.get('max_edges', 40), step=1, key=f'co_max_edges_{scope}')
    options = {'topic_id': None if topic is None else int(topic), 'include_outliers': include,
               'max_keywords': int(maximum), 'min_frequency': int(frequency), 'min_cooccurrence': int(minimum), 'max_edges': int(max_edges)}
    if st.button(tr('计算共现网络'), disabled=not confirmed, key=f'co_compute_{scope}'):
        try:
            with st.spinner(tr('正在逐片段分词并统计无向词对……')):
                previous = calculate_cooccurrence(result, options, word_settings, source_id)
            visual['cooccurrence'] = previous
            st.success(tr('共现统计完成；请下载新的项目 ZIP 保存结果。'))
        except (ValueError, KeyError, TypeError) as exc:
            st.error(tr('共现统计未完成：{0}', diagnostic(exc)))
    if previous and (previous['settings'] != options or previous['keyword_settings'] != word_settings or previous['settings_source'] != source_id):
        st.warning(tr('以下是上次计算并保存的网络；当前控件已更改，点击“计算共现网络”才会生成新结果。'))
    if not previous:
        st.info(tr('选择范围和词表后点击计算；不会自动计算或重新训练。'))
        return
    nodes, edges, eligible_edges = cooccurrence_tables(previous)
    shown_nodes, shown_edges = nodes[nodes['in_graph']], edges[edges['in_graph']]
    st.caption(tr('已计算范围：{0} 个片段；完整统计 {1} 个词、{2} 个词对；图中 {3} 个词、{4} 条线。满足门槛的候选连线 {5} 条。', f"{previous['segment_count']}", f'{len(nodes)}', f'{len(edges)}', f'{len(shown_nodes)}', f'{len(shown_edges)}', f'{eligible_edges}'))
    with st.expander(tr('查看已计算网络的实际范围与词表')):
        st.json({'unit': previous['unit'], 'directed': previous['directed'], 'settings': previous['settings'],
                 'settings_source': previous['settings_source'], 'keyword_settings': previous['keyword_settings']})
    if len(shown_nodes) < 2 or shown_edges.empty:
        st.info(tr('当前范围或门槛下有效词或连线不足；可调整词表、范围或门槛后重新计算。'))
    else:
        if len(shown_nodes) > 20 or len(shown_edges) > 40:
            st.warning(tr('网络较密，请提高门槛、减少关键词或增大画布，并检查标签是否重叠。'))
        try:
            png, missing = render_network(nodes, edges, width, height, dpi, font_path, appearance, translate=tr)
            if missing:
                st.warning(tr('当前字体缺少字形：') + ''.join(missing[:30]))
            preview_png = png if dpi == 150 else render_network(nodes, edges, width, height, 150, font_path, appearance, translate=tr)[0]
            st.image(preview_png, width='stretch')
            st.download_button(tr('下载共现网络 PNG'), png, 'cooccurrence.png', 'image/png', key=f'co_png_{scope}')
            svg, _ = render_network(nodes, edges, width, height, dpi, font_path, appearance, fmt='svg', translate=tr)
            st.download_button(tr('下载共现网络 SVG'), svg, 'cooccurrence.svg', 'image/svg+xml', key=f'co_svg_{scope}')
        except (ValueError, RuntimeError) as exc:
            st.warning(tr('共现图无法完整导出：{0}', diagnostic(exc)))
    st.markdown(tr('##### 完整节点与词对统计'))
    st.dataframe(nodes, hide_index=True, width='stretch')
    st.dataframe(edges, hide_index=True, width='stretch')
    st.download_button(tr('下载完整节点 CSV'), csv_bytes(nodes), 'cooccurrence_nodes.csv', 'text/csv', key=f'co_nodes_{scope}')
    st.download_button(tr('下载完整连线 CSV'), csv_bytes(edges), 'cooccurrence_edges.csv', 'text/csv', key=f'co_edges_{scope}')
    record = {k: previous[k] for k in ('schema_version', 'unit', 'directed', 'settings', 'keyword_settings', 'settings_source', 'segment_count')}
    record['graph_style'] = appearance
    record['graph_size'] = {'width_inches':width, 'height_inches':height, 'dpi':dpi}
    st.download_button(tr('下载共现计算记录 JSON'), json.dumps(record, ensure_ascii=False, indent=2).encode('utf-8'),
                       'cooccurrence_settings.json', 'application/json', key=f'co_record_{scope}')
    if not shown_edges.empty:
        pairs = list(zip(shown_edges['source'], shown_edges['target']))
        pair = st.selectbox(tr('选择图中词对回查原文'), pairs, format_func=lambda p: f'{p[0]} × {p[1]}', key=f'co_pair_{scope}')
        matches = cooccurrence_evidence(result, previous, *pair)
        st.caption(tr('以下 {0} 个片段同时含有该词对；显示文档 ID、片段 ID、原文位置及原始字段。', f'{len(matches)}'))
        st.dataframe(matches, hide_index=True, width='stretch')
        f = result['metadata']['generated_fields']
        sid = st.selectbox(tr('选择共现片段查看完整原文'), matches[f['segment_id']].tolist(), key=f'co_segment_{scope}')
        if sid:
            segment = matches.set_index(f['segment_id']).loc[sid]
            doc = result['rows'].set_index(f['document_id']).loc[segment[f['document_id']]]
            st.caption(tr('文档 ID：{0}；片段 ID：{1}；位置 [{2}, {3})', f"{segment[f['document_id']]}", f'{sid}', f"{segment[f['start_char']]}", f"{segment[f['end_char']]}"))
            with st.expander(tr('共现片段与完整原文')):
                source_text(segment[f['segment_text']]); source_text(doc[f['original_text']])


def show_similarity(result, scope, width, height, dpi, font_path, appearance):
    st.markdown(tr('#### 主题相似度热力图'))
    obj = result['visualization'].get('topic_similarity')
    if obj is None:
        st.info(tr('该项目未保存计算主题相似度所需的数据，可使用原始语料重新分析后生成。'))
        return
    st.caption(tr('{0}；{1}。数值范围 −1 至 1，颜色使用固定的完整范围；相似度并非因果关系。', tr(VECTOR_TYPE), tr(METHOD)))
    settings = obj['settings']
    show_outlier = st.checkbox(tr('热力图显示离群主题 -1（默认隐藏）'), value=settings['show_outliers'], key=f'sim_outliers_{scope}')
    show_values = st.checkbox(tr('在热力图格子中显示数值'), value=settings['show_values'], key=f'sim_values_{scope}')
    ids = sorted(result['topics']['Topic'].astype(int).tolist())
    options = [i for i in ids if show_outlier or i != -1]
    names = materialize(result)[0]['topics'].set_index('Topic')['Display_Name'].to_dict()
    selected = st.multiselect(tr('选择热力图主题（最多绘制 30 个）'), options,
                              default=[i for i in settings['selected_topics'] if i in options],
                              format_func=localize_options(lambda i: str(i) + (tr('（离群）') if i == -1 else '') + f' · {names[i]}'),
                              key=f'sim_topics_{scope}')
    hidden_outlier = [-1] if not show_outlier and -1 in settings['selected_topics'] else []
    settings.update({'show_outliers': show_outlier, 'show_values': show_values,
                     'selected_topics': list(dict.fromkeys(selected + hidden_outlier))})
    matrix, labels, excluded = similarity_data(result, selected)
    if excluded:
        st.warning(tr('部分主题无法计算相似度：{0}', ' · '.join(
            f"{item['topic_id']}: {diagnostic(item['reason'])}" for item in excluded)))
    if matrix.empty:
        st.info(tr('当前没有可计算的主题；请选择有非零向量的主题。'))
        return
    if len(matrix) == 1:
        st.info(tr('当前仅有一个可计算主题，只能查看它与自身的相似度。'))
    data = export_matrix(matrix, labels)
    st.dataframe(data, hide_index=True, width='stretch')
    st.caption(tr('表格和 CSV 保留计算精度；图内数字仅为显示时四舍五入。尺寸及 DPI 沿用上方图表设置。'))
    st.caption(tr('横轴标签很长或主题较多时，请减少选择、增大画布并检查导出的图像。'))
    st.download_button(tr('下载相似度矩阵 CSV'), csv_bytes(data), 'topic_similarity.csv', 'text/csv', key=f'sim_csv_{scope}')
    if len(matrix) > 30:
        st.warning(tr('主题超过 30 个，请选择较少主题后绘制热力图；矩阵 CSV 仍可下载。'))
        return
    if not font_path:
        st.warning(tr('未找到中文字体，无法可靠绘制或导出热力图。'))
        return
    try:
        png, missing = render_similarity(matrix, labels, width, height, dpi, font_path, show_values, appearance=appearance, translate=tr)
        preview_png = png if dpi == 150 else render_similarity(matrix, labels, width, height, 150, font_path, show_values, appearance=appearance, translate=tr)[0]
        if missing:
            st.warning(tr('当前字体缺少以下字符的字形，图中可能无法显示：') + ''.join(missing[:30]))
        st.image(preview_png, width='stretch')
        st.download_button(tr('下载相似度热力图 PNG'), png, 'topic_similarity.png', 'image/png', key=f'sim_png_{scope}')
        svg, _ = render_similarity(matrix, labels, width, height, dpi, font_path, show_values, 'svg', appearance=appearance, translate=tr)
        st.download_button(tr('下载相似度热力图 SVG'), svg, 'topic_similarity.svg', 'image/svg+xml', key=f'sim_svg_{scope}')
    except (ValueError, RuntimeError) as exc:
        st.warning(tr('热力图无法完整导出：{0}', diagnostic(exc)))


def show_hierarchy(result, scope, width, height, dpi, font_path, appearance):
    st.markdown(tr('#### 主题向量层级图'))
    state = ensure_hierarchy(result)
    if state is None:
        st.info(tr('该项目未保存主题向量，无法生成层级图；可使用原始语料重新分析。'))
        return
    st.caption(tr('向量：{0}；距离：{1}；连接：{2}。该图基于当前选择的主题重新计算，展示模型空间关系，不自动构成研究者确认的主题分类。', tr(VECTOR_TYPE), tr(DISTANCE), tr(LINKAGE)))
    settings = state['settings']
    show_outlier = st.checkbox(tr('层级图纳入离群主题 -1（默认不纳入）'), value=settings['show_outliers'], key=f'hier_outliers_{scope}')
    ids = sorted(result['topics']['Topic'].astype(int).tolist())
    options = [i for i in ids if show_outlier or i != -1]
    labels = materialize(result)[0]['topics'].set_index('Topic')['Display_Name'].to_dict()
    selected = st.multiselect(tr('选择层级图主题（最多 30 个）'), options,
                              default=[i for i in settings['selected_topics'] if i in options],
                              format_func=localize_options(lambda i: str(i) + (tr('（离群）') if i == -1 else '') + f' · {labels[i]}'),
                              key=f'hier_topics_{scope}')
    if len(selected) > MAX_TOPICS:
        st.warning(tr('最多选择 {0} 个主题；请减少选择，当前超限选择不会保存。', f'{MAX_TOPICS}'))
        return
    if sorted(selected) != settings['selected_topics'] or show_outlier != settings['show_outliers']:
        state = update_hierarchy(result, selected, show_outlier)
    tree = state['result']
    if tree['excluded']:
        st.warning(tr('以下主题没有可用的非零向量，已排除：{0}', ' · '.join(
            f"{x['topic_id']}: {diagnostic(x['reason'])}" for x in tree['excluded'])))
    leaves = tree['leaf_topics']
    if len(leaves) < 2:
        st.info(tr('当前有 {0} 个可计算主题；层级图至少需要 2 个。', f'{len(leaves)}'))
        return
    table = hierarchy_table(state, labels)
    st.caption(tr('叶节点编号按主题编号排序从 0 开始；合并节点从叶节点数开始递增。表中列出左右子节点及其包含的原始主题编号。'))
    st.dataframe(table, hide_index=True, width='stretch')
    st.download_button(tr('下载层级数据 CSV'), csv_bytes(table), 'topic_hierarchy.csv', 'text/csv', key=f'hier_csv_{scope}')
    if not font_path:
        st.warning(tr('未找到中文字体，无法可靠绘制或导出层级图。'))
        return
    if len(leaves) > height * 3:
        st.warning(tr('主题较多，请增大图表高度或选择更少主题以改善可读性。'))
    try:
        png, missing = render_hierarchy(state, labels, width, height, dpi, font_path, appearance=appearance, translate=tr)
        preview_png = png if dpi == 150 else render_hierarchy(state, labels, width, height, 150, font_path, appearance=appearance, translate=tr)[0]
        if missing:
            st.warning(tr('当前字体缺少以下字符的字形，图中可能无法显示：') + ''.join(missing[:30]))
        st.image(preview_png, width='stretch')
        st.download_button(tr('下载层级图 PNG'), png, 'topic_hierarchy.png', 'image/png', key=f'hier_png_{scope}')
        svg, _ = render_hierarchy(state, labels, width, height, dpi, font_path, 'svg', appearance=appearance, translate=tr)
        st.download_button(tr('下载层级图 SVG'), svg, 'topic_hierarchy.svg', 'image/svg+xml', key=f'hier_svg_{scope}')
    except (ValueError, RuntimeError) as exc:
        st.warning(tr('层级图无法完整导出：{0}', diagnostic(exc)))


def show_chart_output(frame, kind, width, height, dpi, font_path, scope, topic_id=None, topic_label='', appearance=None):
    st.download_button(tr('下载图表数据 CSV'), csv_bytes(frame), f'{kind}_data.csv', 'text/csv', key=f'{kind}_csv_{scope}')
    if not font_path:
        return
    try:
        png, missing = render_chart(frame, kind, width, height, dpi, font_path, topic_id=topic_id, topic_label=topic_label, appearance=appearance, translate=tr)
        preview = png
        if dpi > 150:
            preview, _ = render_chart(frame, kind, width, height, 150, font_path, topic_id=topic_id, topic_label=topic_label, appearance=appearance, translate=tr)
        if missing:
            st.warning(tr('当前字体缺少以下字符的字形，图中可能无法显示：') + ''.join(missing[:30]))
        if len(frame) > height * 3:
            st.warning(tr('当前图包含较多项目，请增大高度以改善可读性。'))
        st.image(preview, width="stretch")
        st.download_button(tr('下载 PNG'), png, f'{kind}.png', 'image/png', key=f'{kind}_png_{scope}')
        svg, _ = render_chart(frame, kind, width, height, dpi, font_path, topic_id=topic_id, topic_label=topic_label, fmt='svg', appearance=appearance, translate=tr)
        st.download_button(tr('下载 SVG'), svg, f'{kind}.svg', 'image/svg+xml', key=f'{kind}_svg_{scope}')
    except (ValueError, RuntimeError) as exc:
        st.warning(tr('图表无法完整导出：{0}', diagnostic(exc)))


def open_workspace():
    st.subheader(tr('打开已有项目'))
    st.info(tr('恢复的是已保存的分析结果；项目不含可继续训练或推理的模型。无需重新建模或下载模型。'))
    uploaded = st.file_uploader(tr('上传本工作台的项目 ZIP（最多 50 MiB）'), type=["zip"],
                                max_upload_size=50, key="project_upload")
    if not uploaded:
        st.session_state.pop('opened_project', None)
        st.session_state.pop('prepared_export_restored', None)
        return
    try:
        digest = hashlib.sha256(uploaded.getvalue()).hexdigest()
        cached = st.session_state.get('opened_project')
        if not cached or cached['digest'] != digest:
            st.session_state.pop('prepared_export_restored', None)
            cached = {'digest': digest, 'result': open_project(uploaded.getvalue())}
            st.session_state['opened_project'] = cached
        original_result = cached['result']
    except ValueError as exc:
        st.session_state.pop('opened_project', None)
        st.session_state.pop('prepared_export_restored', None)
        st.error(tr('无法打开项目：{0}', diagnostic(exc)))
        return
    anchor('groups')
    topic_editor(original_result, 'restored')
    from studio.topic_groups_ui import show as show_topic_groups
    show_topic_groups(original_result, f"restored_{cached['digest']}")
    result, export_names = materialize(original_result)
    meta, docs, segments, topics = result['metadata'], result['rows'], result['segments'], result['topics']
    f = meta['generated_fields']
    st.success(tr('项目完整性和片段原文位置检查通过。'))
    st.caption(tr('项目：{0}；保存时间：{1}；格式版本：{2}', f"{result['project_info']['name']}", f"{result['project_info']['saved_at']}", f"{result['project_info']['version']}"))
    if 'keyword_processing' not in meta:
        st.info(tr('历史项目未保存本次分析的词表和详细分词设置，不能推断其使用了当前词表。'))
    st.caption(tr('已保存项目只能查看历史关键词；如需采用新词表，须对原始数据新建并重新运行分析。'))
    show_diagnostics(result)
    with st.expander(tr('原始文档与处理状态')):
        st.dataframe(docs, hide_index=True, width="stretch")
    from studio.policy_transfer_ui import render as render_policy_transfer
    render_policy_transfer(original_result, cached['digest'])
    anchor('topics')
    st.subheader(tr('主题与片段'))
    st.dataframe(topics[['Topic','Is_Outlier','Count','Name','Keywords','Custom_Name','Display_Name']], hide_index=True, width="stretch")
    topic = st.selectbox(tr('按主题查看已保存片段'), ["全部片段", *sorted(topics['Topic'].astype(int).tolist())], format_func=localize_options(), key='restored_topic')
    shown = segments if isinstance(topic, str) else segments[segments[f['topic_id']].eq(topic)]
    st.dataframe(shown, hide_index=True, width="stretch")
    segment_id = st.selectbox(tr('选择已保存片段核对原文'), shown[f['segment_id']].tolist(), key='restored_segment')
    if segment_id:
        segment = segments.set_index(f['segment_id']).loc[segment_id]
        doc = docs.set_index(f['document_id']).loc[segment[f['document_id']]]
        start, end = int(segment[f['start_char']]), int(segment[f['end_char']])
        st.caption(tr('文档 ID：{0}；片段顺序：{1}；位置 [{2}, {3})', f"{segment[f['document_id']]}", f"{segment[f['segment_order']]}", f'{start}', f'{end}'))
        with st.expander(tr('查看片段与完整原文')):
            source_text(segment[f['segment_text']]); source_text(doc[f['original_text']])
    with st.expander(tr('运行参数与处理记录')):
        if 'keyword_processing' in meta:
            st.write(tr('已保存的关键词处理设置：'))
            st.json(meta['keyword_processing'])
        st.json(meta)
    show_charts(original_result, f"restored_{cached['digest']}")
    from studio.annual_ui import show as show_annual
    show_annual(original_result, f"restored_{cached['digest']}")
    from studio.yearly_ui import show as show_yearly
    show_yearly(original_result, f"restored_{cached['digest']}", encoder_resource)
    from studio.cross_year_ui import show as show_cross_year
    show_cross_year(original_result, f"restored_{cached['digest']}")
    anchor('exports')
    st.subheader(tr('再次导出'))
    st.caption(tr('另存为新项目：包含按年建模结果时为版本 6，否则为版本 5。旧版工作台不能打开新版本，请保留原文件并自行选择保存位置。'))
    from studio.deferred_exports import show as show_deferred_exports
    show_deferred_exports(original_result, 'restored')
    st.download_button(tr('下载原始文档 CSV'), csv_bytes(docs), 'bertopic_records.csv', 'text/csv', key='restored_records')
    st.download_button(tr('下载主题 CSV'), csv_bytes(topics), 'bertopic_topics.csv', 'text/csv', key='restored_topics')
    st.download_button(tr('下载运行记录 JSON'), json.dumps(meta, ensure_ascii=False, indent=2).encode(),
                       'bertopic_run.json', 'application/json', key='restored_run')


def keyword_controls():
    st.subheader(tr('关键词处理设置'))
    st.caption(tr('这些设置仅用于 BERTopic 主题关键词与 c-TF-IDF 权重；语义向量模型仍接收未经这些设置修改的原始文本片段。'))
    mode = st.radio(tr('关键词分词方式'), ['原有方式（默认）', '中文分词', '英文分词'], format_func=localize_options(), key='keyword_mode', horizontal=True)
    st.caption(tr('原有方式使用 jieba 并仅保留多于 1 个字符的词；中文分词使用独立 jieba 词典，保留单字并过滤标点。两者均不转换简繁体。'))
    st.caption(tr('英文分词保留英文单词、数字及词内撇号和连字符，可使用停用词。中文和混合语料可选择中文分词。英文分词不使用自定义词典。'))
    enabled = st.checkbox(tr('启用停用词'), value=False, key='stop_enabled')
    st.caption(tr('每行一个词；空行忽略。上传的词表与下方手动编辑的词合并；移除上传文件即可移除其中词条。默认没有领域停用词。'))
    stop_file = st.file_uploader(tr('上传停用词表（UTF-8，每行一个词）'), type=['txt'], max_upload_size=1, key='stop_file')
    stop_text = st.text_area(tr('编辑停用词（每行一个）'), key='stop_text', height=100)
    dictionary_file = st.file_uploader(tr('上传自定义词典（UTF-8，每行一个词）'), type=['txt'], max_upload_size=1, key='dictionary_file', disabled=mode == '英文分词')
    dictionary_text = st.text_area(tr('编辑自定义词典（每行一个术语）'), key='dictionary_text', height=100, disabled=mode == '英文分词',
                                   placeholder=tr('阿弥陀佛\n称名念佛\n外国人政策'))
    try:
        stop_list = parse_word_list(stop_text + '\n' + (decode_word_file(stop_file.getvalue()) if stop_file else ''))
        dictionary = [] if mode == '英文分词' else parse_word_list(
            dictionary_text + '\n' + (decode_word_file(dictionary_file.getvalue()) if dictionary_file else ''))
        keyword_mode = {'中文分词': 'chinese', '英文分词': 'english'}.get(mode, 'legacy')
        settings = make_settings(keyword_mode, enabled, stop_list, dictionary)
    except ValueError as exc:
        st.error(tr('词表无法使用：{0}', diagnostic(exc)))
        return None
    st.download_button(tr('下载当前停用词表'), ('\n'.join(stop_list) + ('\n' if stop_list else '')).encode('utf-8'),
                       'stopwords.txt', 'text/plain', key='download_stopwords')
    st.download_button(tr('下载当前自定义词典'), ('\n'.join(dictionary) + ('\n' if dictionary else '')).encode('utf-8'),
                       'dictionary.txt', 'text/plain', key='download_dictionary')
    st.caption(tr('预览与关键词提取采用相同设置：BERTopic 将换行和制表符替换为空格，向量化器将英文小写；原始文本不变。'))
    sample = st.text_area(tr('分词预览文本'), key='keyword_preview', height=80)
    if sample:
        st.write(tr('当前设置下的关键词分词：'), preview(sample, settings))
    return settings


def workspace():
    mode = st.sidebar.radio(tr('工作模式'), ["新建分析", "打开已有项目", "基础政策评价"], format_func=localize_options(), key="work_mode")
    if st.session_state.get('_export_workspace_mode') != mode:
        st.session_state.pop('prepared_export_new', None)
        st.session_state.pop('prepared_export_restored', None)
        st.session_state['_export_workspace_mode'] = mode
    if mode == "打开已有项目":
        open_workspace()
        return
    if mode == '基础政策评价':
        from studio.policy_ui import workspace as policy_workspace
        policy_workspace(encoder_resource)
        return
    st.subheader(tr('1 · 导入与检查'))
    uploaded = st.file_uploader(tr('上传 CSV、XLSX 或 TXT 文件（不超过 10 MiB）'), type=["csv", "xlsx", "txt"], key="csv_upload")
    with st.expander(tr('示例语料')):
        st.download_button(tr('下载 XLSX 测试文件'), (ROOT / "data/examples/import_formats.xlsx").read_bytes(),
                           file_name="import_formats.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="example_xlsx")
        st.download_button(tr('下载 TXT 测试文件'), (ROOT / "data/examples/import_long.txt").read_bytes(),
                           file_name="import_long.txt", mime="text/plain", key="example_txt")
        st.download_button(tr('下载合成测试 CSV'), (ROOT / "data/examples/import_edge_cases.csv").read_bytes(),
                           file_name="import_edge_cases.csv", mime="text/csv", key="example_download")
        st.download_button(tr('下载中文分词测试 CSV'), (ROOT / "data/examples/keyword_terms_zh.csv").read_bytes(),
                           file_name="keyword_terms_zh.csv", mime="text/csv", key="keyword_example_download")
        st.download_button(tr('下载长文本测试 CSV'), (ROOT / "data/examples/segmentation_zh.csv").read_bytes(),
                           file_name="segmentation_zh.csv", mime="text/csv", key="long_example_download")
    if uploaded is None:
        st.session_state.pop("analysis_result", None)
        st.session_state.pop('prepared_export_new', None)
        st.session_state.pop('keyword_context', None)
        st.info(tr('请选择 CSV、XLSX 或 TXT 文件。'))
        return
    source_format = uploaded.name.rsplit('.', 1)[-1].lower()
    raw = uploaded.getvalue()
    try:
        if source_format == 'xlsx':
            digest = hashlib.sha256(raw).hexdigest()
            cache = st.session_state.get('xlsx_import_cache')
            if not cache or cache['digest'] != digest:
                cache = {'digest': digest, 'sheets': xlsx_sheets(raw), 'parsed': {}}
                st.session_state['xlsx_import_cache'] = cache
            sheets = cache['sheets']
            if not sheets:
                raise ValueError('XLSX 中没有工作表。')
            sheet = st.selectbox(tr('选择工作表'), sheets, key="xlsx_sheet")
            if sheet not in cache['parsed']:
                cache['parsed'][sheet] = read_xlsx(raw, sheet)
            corpus = cache['parsed'][sheet]
            encoding = 'xlsx'
            st.caption(tr('所选工作表：{0}；原始数据行：{1}。首行为表头。', f'{sheet}', f'{len(corpus.frame)}'))
            st.caption(tr('公式读取文件中已有的缓存结果，不计算公式；缓存可能过期，请在 Excel 中重算并保存。无缓存的非正文公式保留原公式，正文公式无缓存时不能建模。数字和日期按值转文本，不模拟 Excel 显示格式。'))
            formulas = corpus.source_details['formula_cells']
            if formulas:
                missing_count = sum(not item['cached_available'] for item in formulas)
                st.warning(tr('发现 {0} 个公式单元格，其中 {1} 个无可用缓存。未计算公式；保留的公式表达式在 CSV 中重新打开时可能被表格软件识别。', f'{len(formulas)}', f'{missing_count}'))
                with st.expander(tr('查看公式读取记录')):
                    st.dataframe(formulas, hide_index=True, width="stretch")
        elif source_format in ('csv', 'txt'):
            encoding = st.selectbox(tr('文件编码'), ["utf-8-sig", "gb18030", "utf-16"], key="encoding")
            corpus = read_txt(raw, encoding) if source_format == 'txt' else read_csv(raw, encoding)
            if source_format == 'txt':
                st.caption(tr('TXT 文件作为一个原始文档导入；所有换行、空格和标点均保留。'))
        else:
            raise ValueError('不支持此文件格式。')
    except ValueError as exc:
        st.session_state.pop("analysis_result", None)
        st.session_state.pop('prepared_export_new', None)
        st.error(diagnostic(exc))
        return
    columns = corpus.frame.columns.tolist()
    preferred = next((c for c in ("text", "正文", "文本", "内容") if c in columns), columns[0])
    column_context = (corpus.sha256, corpus.source_format, corpus.source_details.get('sheet'))
    if st.session_state.get('text_column_context') != column_context:
        st.session_state.pop('text_column', None)
        st.session_state['text_column_context'] = column_context
    text_column = st.selectbox(tr('用于主题分析的文本列'), columns, index=columns.index(preferred), key="text_column")
    if corpus.source_format == 'xlsx' and text_column not in corpus.source_details['detected_text_columns']:
        st.warning(tr('所选列未发现非空文本单元格；数字和日期会转成文本，请确认是否选择了正文列。'))
    missing_formulas = missing_text_formulae(corpus, text_column)
    if missing_formulas:
        st.error(tr('正文列有 {0} 个公式缺少可用缓存；请在 Excel 中计算并保存，或替换为固定文本。', f'{len(missing_formulas)}'))
    if corpus.frame[text_column].str.strip().eq('').all():
        st.warning(tr('所选文本列没有有效文本，请选择其他列或检查文件。'))
    policy_label = st.radio(tr('重复文本处理'), ["保留全部（默认）", "去重，仅首条参与建模"], format_func=localize_options(), key="duplicate_policy")
    policy = "keep" if policy_label == "保留全部（默认）" else "deduplicate"
    prepared = prepare(corpus, text_column, policy)
    f = prepared.fields
    counts = prepared.counts
    a, b, c, d = st.columns(4)
    a.metric(tr('输入记录'), counts["input"])
    b.metric(tr('空文本'), counts["empty"])
    c.metric(tr('重复文本（多出的副本）'), counts["duplicate_extra"])
    d.metric(tr('原始有效文档'), counts["selected"])
    st.caption(tr('空文本按空字符串或纯空白判断；重复按非空原文完全一致判断，不自动去空格。去重时仅首条参与建模。原始文档表保留所有记录，排除行不会生成建模片段。'))
    st.caption(tr('记录 ID 由文件内容哈希与数据记录序号生成。XLSX 还包含工作表标识和 Excel 行号；同一文件和工作表重传或切换处理方式时保持稳定。'))
    st.dataframe(corpus.frame.head(100), hide_index=True, width="stretch")
    st.caption(tr('预览前 100 条记录，建模按完整文件处理；XLSX 记录序号为原表行号，CSV 为数据记录序号。'))
    with st.expander(tr('查看记录 ID、原文及排除明细')):
        st.dataframe(prepared.rows[[f["source_record"], f["record_id"], f["original_text"],
                                    f["status"], f["duplicate_of"]]], hide_index=True, width="stretch")

    if missing_formulas:
        st.session_state.pop('analysis_result', None)
        st.session_state.pop('prepared_export_new', None)
        st.info(tr('请先处理正文列中的无缓存公式后重新上传文件。'))
        return

    st.subheader(tr('2 · 本地主题分析'))
    device = st.selectbox(tr('嵌入计算设备'), ["cpu", "mps"], key="device")
    split_label = st.radio(tr('超长文本处理'), ["超长时停止（默认）", "自动切分超长文本"], format_func=localize_options(), key="split_mode")
    split_mode = "auto" if split_label == "自动切分超长文本" else "stop"
    st.caption(tr('聚类单位是文本片段：短文也对应一个片段，一篇文档可以对应多个主题。切分默认无重叠，不归纳整篇文档的唯一主题。'))
    keyword_context = (corpus.sha256, corpus.source_format, corpus.source_details.get('sheet'))
    if st.session_state.get('keyword_context') != keyword_context:
        for key in ('keyword_mode', 'stop_enabled', 'stop_file', 'stop_text', 'dictionary_file', 'dictionary_text', 'keyword_preview'):
            st.session_state.pop(key, None)
        st.session_state['keyword_context'] = keyword_context
    keyword_settings = keyword_controls()
    if keyword_settings is None:
        st.session_state.pop('analysis_result', None)
        st.session_state.pop('prepared_export_new', None)
        return
    st.markdown(tr('#### 基础设置'))
    cluster_size = st.number_input(tr('最小主题规模（HDBSCAN 最小聚类规模）'), min_value=1, value=4, step=1, key='cluster_min_size',
                                   help=tr('较大值要求形成主题的片段群达到更大的规模；不保证主题数量。'))
    with st.expander(tr('高级设置')):
        st.caption(tr('最小聚类规模由上方基础设置控制。以下设置作用于降维和聚类，不改变送入向量模型的原文。'))
        neighbors = st.number_input(tr('UMAP 邻居数'), min_value=1, value=8, step=1, key='cluster_neighbors', help=tr('影响降维时考虑的局部邻域大小。'))
        distance = st.number_input(tr('UMAP 最小距离'), min_value=0.0, max_value=1.0, value=0.0, step=0.05, key='cluster_min_dist', help=tr('影响降维后点之间的紧密程度。'))
        samples = st.number_input(tr('HDBSCAN 最小样本数'), min_value=1, value=2, step=1, key='cluster_min_samples', help=tr('影响密度估计和离群判断的保守程度。'))
    st.caption(tr('随机种子固定为 42，UMAP 使用单线程；同一环境有利于复现，不保证不同设备或环境结果完全相同。更改设置会清除当前页面结果，请先下载需保留的项目；重新开始会产生独立分析，不覆盖已保存项目。'))
    clustering = {'n_neighbors': int(neighbors), 'min_dist': float(distance), 'min_cluster_size': int(cluster_size), 'min_samples': int(samples)}
    fingerprint = (corpus.sha256, corpus.source_format, corpus.source_details.get('sheet'), encoding, text_column, policy, device, split_mode,
                   json.dumps(keyword_settings, ensure_ascii=False, sort_keys=True), json.dumps(clustering, sort_keys=True))
    existing = st.session_state.get("analysis_result")
    if existing and existing["fingerprint"] != fingerprint:
        st.session_state.pop("analysis_result", None)
        st.session_state.pop('prepared_export_new', None)
    try:
        cached_plan = st.session_state.get("segment_preview")
        encoder, lock = encoder_resource(device)
        if not cached_plan or cached_plan["fingerprint"] != fingerprint:
            with st.spinner(tr('正在用本地 tokenizer 检查长度与切分……')), lock:
                plan = plan_segments(prepared, encoder, split_mode)
            st.session_state["segment_preview"] = {"fingerprint": fingerprint, "plan": plan}
        else:
            plan = cached_plan["plan"]
    except Exception as exc:
        st.session_state.pop("analysis_result", None)
        st.session_state.pop('prepared_export_new', None)
        st.error(tr('长度检查未完成：{0}', diagnostic(exc)))
        return
    a, b, c = st.columns(3)
    a.metric(tr('原始有效文档数'), plan.counts["valid_documents"])
    b.metric(tr('超长文档数'), plan.counts["long_documents"])
    c.metric(tr('参与建模的片段数'), plan.counts["segments"])
    st.caption(tr('实际长度上限 {0} token，包含 {1} 个特殊 token。优先自然段、句子边界，必要时继续拆分；保留空白与标点。位置采用从 0 开始的 Unicode 字符索引，起点包含、终点不包含。', f"{plan.settings['effective_max_tokens']}", f"{plan.settings['special_tokens']}"))
    if plan.blocked:
        st.warning(tr('已检测到超长文档，当前方式将停止建模。请选择自动切分，或先处理原文。'))
    with st.expander(tr('查看片段计划与原文位置')):
        st.dataframe(plan.segments, hide_index=True, width="stretch")
    parameter_error = None
    if plan.counts['segments'] >= 10:
        try:
            validate_clustering_settings(clustering, plan.counts['segments'])
        except ValueError as exc:
            parameter_error = str(exc)
            st.error(tr('聚类参数无效：{0}', diagnostic(parameter_error)))
    ready = not missing_formulas and not plan.blocked and plan.counts["segments"] >= 10 and plan.counts["unique_segments"] >= 5 and not parameter_error
    if (plan.counts["segments"] < 10 or plan.counts["unique_segments"] < 5) and not plan.blocked:
        st.warning(tr('建模至少需要 10 个有效片段和 5 种不同片段。请检查文本列或更换语料。'))
    elif plan.counts["valid_documents"] < 50:
        st.warning(tr('原始文档数量较少；切分增加片段数不等于增加独立文档，主题仍需结合原文解释。'))
    if st.button(tr('开始主题分析'), disabled=not ready, type="primary", key="run_analysis"):
        st.session_state.pop("analysis_result", None)
        st.session_state.pop('prepared_export_new', None)
        message = st.empty()
        try:
            with st.spinner(tr('正在运行本地模型，请稍候……')):
                message.info(tr('正在加载本地模型……'))
                encoder, lock = encoder_resource(device)
                with lock:
                    result = analyze(prepared, encoder, device, lambda msg: message.info(tr(msg)), split_mode=split_mode, keyword_settings=keyword_settings, clustering_settings=clustering)
                message.info(tr('正在核对并保存导出文件……'))
                directory = persist(result, corpus, include_exports=False)
                st.session_state["analysis_result"] = {"fingerprint": fingerprint, "result": result,
                                                       "directory": str(directory)}
            message.empty()
        except Exception as exc:
            message.empty()
            st.error(tr('本次分析未完成：{0}', diagnostic(exc)))
    saved = st.session_state.get("analysis_result")
    if not saved:
        return
    original_result = saved["result"]
    anchor('groups')
    topic_editor(original_result, 'new')
    from studio.topic_groups_ui import show as show_topic_groups
    show_topic_groups(original_result, f"new_{saved['result']['metadata']['run_id']}")
    result, export_names = materialize(original_result)
    metadata = result["metadata"]
    rows = result["rows"]
    segments = result["segments"]
    anchor('topics')
    st.subheader(tr('3 · 主题与原文'))
    st.success(tr('完成：{0} 篇有效文档形成 {1} 个片段参与建模；全部 {2} 条原始记录均已保存。', f"{counts['selected']}", f'{len(segments)}', f'{len(rows)}'))
    st.caption(tr('运行目录只自动保存导入文件和运行记录；请在关闭页面前准备并下载项目 ZIP，以保存完整分析结果。'))
    show_diagnostics(result)
    if metadata["topic_count_excluding_outliers"] == 0:
        st.warning(tr('本次全部片段为离群类 -1，未形成主题；结果仍可核查和导出。'))
    if metadata.get('empty_keyword_topics'):
        st.warning(tr('以下主题在当前设置下没有有效关键词：{0}；原文和主题分配仍保留。', f"{metadata['empty_keyword_topics']}"))
    topic_ids = sorted(result["topics"]["Topic"].astype(int).tolist())
    topic = st.selectbox(tr('按主题查看片段'), ["全部片段", *topic_ids], format_func=localize_options(), key=f"topic_{metadata['run_id']}")
    mask = segments[f["status"]].eq("modeled") if isinstance(topic, str) else segments[f["topic_id"]].eq(topic).fillna(False)
    st.dataframe(segments.loc[mask], hide_index=True, width="stretch")
    segment_id = st.selectbox(tr('选择片段核对原文'), segments.loc[mask, f["segment_id"]].tolist(), key=f"segment_{metadata['run_id']}_{topic}")
    if segment_id:
        segment = segments.set_index(f["segment_id"]).loc[segment_id]
        document = rows.set_index(f["document_id"]).loc[segment[f["document_id"]]]
        start, end = int(segment[f["start_char"]]), int(segment[f["end_char"]])
        with st.expander(tr('片段与完整原文')):
            st.caption(tr('文档 ID：{0}；片段顺序：{1}；位置 [{2}, {3})', f"{segment[f['document_id']]}", f"{segment[f['segment_order']]}", f'{start}', f'{end}'))
            source_text(segment[f["segment_text"]])
            source_text(document[f["original_text"]])
    with st.expander(tr('原始文档处理状态')):
        st.dataframe(rows, hide_index=True, width="stretch")
    st.caption(tr('membership_strength 为可用时的 HDBSCAN 归属强度，不是校准概率或主题重要性。-1 表示离群类。'))

    show_charts(original_result, f"new_{metadata['run_id']}")
    from studio.annual_ui import show as show_annual
    show_annual(original_result, f"new_{metadata['run_id']}")
    from studio.yearly_ui import show as show_yearly
    show_yearly(original_result, f"new_{metadata['run_id']}", encoder_resource)
    from studio.cross_year_ui import show as show_cross_year
    show_cross_year(original_result, f"new_{metadata['run_id']}")
    anchor('exports')
    st.subheader(tr('4 · 下载结果'))
    st.caption(tr('项目包含原文和分析结果；包含按年建模结果时为格式版本 6，否则为版本 5。旧版工作台不能打开版本 6；项目不含可继续训练的模型。'))
    from studio.deferred_exports import show as show_deferred_exports
    show_deferred_exports(original_result, 'new')
    from pathlib import Path
    directory = Path(saved["directory"])
    a, c = st.columns(2)
    a.download_button(tr('下载原始文档 CSV'), csv_bytes(result["rows"]),
                      file_name="bertopic_records.csv", mime="text/csv", key="records_download")
    st.download_button(tr('下载主题汇总 CSV'), csv_bytes(result['topics']), 'bertopic_topics.csv', 'text/csv', key='topics_download')
    c.download_button(tr('下载运行记录 JSON'), json.dumps(metadata, ensure_ascii=False, indent=2).encode('utf-8'),
                      file_name="bertopic_run.json", mime="application/json", key="metadata_download")
    st.caption(tr('Excel 包含原始文档表、片段结果表、主题表和运行设置；文本单元格保留前导零。CSV 保存原始字段值，表格软件打开时可能自动识别类型。'))
    with st.expander(tr('保存位置与处理记录')):
        st.code(str(directory), language=None)
        st.json(metadata)


workspace()
section_links(st.session_state['work_mode'])
help_menu()
with st.expander(tr('环境与先前验证记录')):
    for label, path in (("环境检查", ROOT / "outputs/environment_check.json"),
                        ("先前合成语料验证", ROOT / "outputs/smoke/summary.json")):
        if path.is_file():
            st.caption(tr(label))
            st.json(json.loads(path.read_text(encoding="utf-8")))
