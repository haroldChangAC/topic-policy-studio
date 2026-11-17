"""Prepare one explicitly selected large export per session, with stale-data checks."""
from studio.i18n import diagnostic, tr, localize_options
import hashlib
import json
import math
import io
import zipfile

import pandas as pd
import streamlit as st

OPTIONS = {
    'project': ('项目 ZIP', 'application/zip', 'analysis', '.hpts.zip'),
    'segments': ('片段结果 CSV', 'text/csv', 'bertopic_segments', '.csv'),
    'excel': ('结果 Excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'bertopic_results', '.xlsx'),
    'group_trace': ('主题归组片段对应 CSV', 'text/csv', 'topic_group_segments', '.csv'),
    'yearly': ('年度建模数据 ZIP', 'application/zip', 'yearly_results', '.zip'),
    'cross_year': ('跨年关联数据及图表 ZIP', 'application/zip', 'cross_year_links', '.zip'),
    'cross_png': ('跨年桑基图 PNG', 'image/png', 'cross_year_sankey', '.png'),
    'cross_svg': ('跨年桑基图 SVG', 'image/svg+xml', 'cross_year_sankey', '.svg'),
}


def fingerprint(result):
    """Hash exact table values and saved annotations without serializing huge tables.

    Long equal strings share a temporary digest, avoiding repeated UTF-8 buffers.
    This is calculated only while a prepared file exists or is requested.
    """
    h = hashlib.sha256()
    memo = {}
    def put(data):
        h.update(len(data).to_bytes(8, 'big'));h.update(data)
    def scalar(value):
        if isinstance(value, str):
            if len(value) >= 1024:
                digest = memo.get(value)
                if digest is None:
                    digest = hashlib.sha256(value.encode('utf-8')).digest();memo[value] = digest
                return b's' + digest
            return b's' + hashlib.sha256(value.encode('utf-8')).digest()
        if value is None:
            return b'none'
        if value is pd.NA:
            return b'pd.NA'
        if isinstance(value, float) and math.isnan(value):
            return b'float.nan'
        return (type(value).__module__ + '.' + type(value).__qualname__ + ':' + repr(value)).encode('utf-8')
    for name in ('rows', 'segments', 'topics'):
        frame=result[name];put(name.encode());put(json.dumps(frame.columns.tolist(),ensure_ascii=False).encode())
        for row in frame.itertuples(index=True,name=None):
            for item in row:put(scalar(item))
    for name in ('metadata','topic_edits','visualization','yearly_analysis'):
        put(name.encode());put(json.dumps(result.get(name),sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf-8'))
    return h.hexdigest()


def create_export(result, kind):
    """Use the same serializers as the former eager download buttons."""
    from studio.project import save_project
    from studio.topic_names import materialize
    from studio.workflow import csv_bytes, excel_bytes
    if kind not in OPTIONS:
        raise ValueError('未知的导出类型。')
    if kind=='project':data=save_project(result)
    elif kind in ('cross_png','cross_svg'):
        from studio.cross_year import calculate
        from studio.cross_year_static import default_export, render
        visual=result.get('visualization') or {}
        settings=visual.get('cross_year_links')
        nodes,frame,_,_=calculate(result,settings)
        data,_=render(nodes,frame,settings,visual.get('cross_year_export') or default_export(),
                      'png' if kind=='cross_png' else 'svg')
    elif kind=='cross_year':
        from studio.cross_year import ADJACENCY, METHOD, WIDTH, calculate, graph
        from studio.cross_year_static import default_export
        settings=(result.get('visualization') or {}).get('cross_year_links')
        nodes,frame,excluded,warnings=calculate(result,settings)
        figure=graph(nodes,frame,settings)
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('similarity.csv',csv_bytes(frame))
            archive.writestr('graph_edges.csv',csv_bytes(frame[frame.in_graph]))
            archive.writestr('nodes.csv',csv_bytes(pd.DataFrame([{k:v for k,v in n.items() if k not in ('vector','unit')} for n in nodes])))
            archive.writestr('settings.json',json.dumps({'settings':settings,'static_export':(result.get('visualization') or {}).get('cross_year_export') or default_export(),'excluded':excluded,'warnings':warnings,
                'method':METHOD,'adjacency':ADJACENCY,'line_weight':WIDTH},ensure_ascii=False,indent=2,allow_nan=False).encode('utf-8'))
            if figure is not None:
                archive.writestr('graph.html',figure.to_html(full_html=True,include_plotlyjs=True,config={'displayModeBar':False}).encode('utf-8'))
        data=buf.getvalue()
    elif kind=='yearly':
        from studio.yearly import tables, validate
        obj=result.get('yearly_analysis')
        validate(result,obj)
        summary,assignments,vectors=tables(result)
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            for filename,frame in [('yearly_topics.csv',summary),('yearly_segments.csv',assignments),('yearly_vectors.csv',vectors)]:
                archive.writestr(filename,csv_bytes(frame))
            archive.writestr('yearly_run.json',json.dumps(obj,ensure_ascii=False,allow_nan=False,indent=2).encode('utf-8'))
        data=buf.getvalue()
    elif kind=='group_trace':
        from studio.topic_groups import trace
        if not (result.get('visualization') or {}).get('topic_groups'):
            raise ValueError('当前项目没有归组记录。')
        data=csv_bytes(trace(result))
    else:
        view,_=materialize(result)
        data=csv_bytes(view['segments']) if kind=='segments' else excel_bytes(view)
    label,mime,stem,suffix=OPTIONS[kind]
    # Streamlit's media store partially hashes large payloads. Including a full
    # content digest in the download name prevents stale same-size collisions.
    digest=hashlib.sha256(data).hexdigest()
    return {'kind':kind,'data':data,'mime':mime,'filename':f'{stem}-{digest[:24]}{suffix}','content_sha256':digest}


def clear(scope):
    st.session_state.pop('prepared_export_'+scope,None)


def show(result,scope):
    cache_key='prepared_export_'+scope;kind_key='export_kind_'+scope
    options=['project','segments','excel']
    if (result.get('visualization') or {}).get('topic_groups'):options.append('group_trace')
    if result.get('yearly_analysis') is not None:options.append('yearly')
    if (result.get('visualization') or {}).get('cross_year_links') is not None:
        options.extend(('cross_year','cross_png','cross_svg'))
    if st.session_state.get(kind_key) not in options:st.session_state.pop(kind_key,None)
    st.caption(tr('大文件按需准备；每次只保留一个。原文、主题命名、归组、图表设置或分析数据变化后需要重新准备。'))
    kind=st.selectbox(tr('选择需要准备的文件'),options,format_func=localize_options(lambda k:tr(OPTIONS[k][0])),key=kind_key)
    cache=st.session_state.get(cache_key)
    if cache is not None:
        if cache['kind']!=kind:
            clear(scope);cache=None
            st.info(tr('已切换导出类型；请选择“准备导出”。'))
        else:
            try:current=fingerprint(result)
            except Exception as exc:
                clear(scope);cache=None;st.error(tr('无法检查导出是否过期：{0}', diagnostic(exc)))
            else:
                if current!=cache['fingerprint']:
                    clear(scope);cache=None
                    st.warning(tr('分析内容或设置已变化，之前准备的文件已失效；请重新准备。'))
    prepare=st.button(tr('准备导出'),key='export_prepare_'+scope)
    if prepare:
        if cache is not None:
            st.info(tr('当前结果未变化，复用已准备的文件。'))
        else:
            try:
                with st.spinner(tr('正在准备导出，请稍候……')):
                    stamp=fingerprint(result)
                    cache=create_export(result,kind)
                    cache['fingerprint']=stamp
                    st.session_state[cache_key]=cache
            except Exception as exc:
                clear(scope);cache=None
                st.error(tr('准备导出失败：{0}；请检查数据后重试。', diagnostic(exc)))
    if cache and st.button(tr('释放已准备文件'),key='export_release_'+scope):
        clear(scope);cache=None
        st.info(tr('已释放本会话的文件引用；Streamlit 媒体缓存会按其清理机制回收，可能并非立即释放内存。'))
    if cache:
        st.success(tr('{0}已准备，可以下载（{1} MiB）。', tr(OPTIONS[kind][0]), f"{len(cache['data']) / 1024 ** 2:.2f}"))
        st.download_button(tr('下载已准备文件'),cache['data'],cache['filename'],cache['mime'],
                           key='export_download_'+scope,on_click='ignore')
    else:
        st.info(tr('尚未准备文件：先选择类型，再点击“准备导出”。'))
