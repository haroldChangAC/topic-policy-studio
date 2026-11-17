"""Basic PMC chart data and rendering; never infer missing indicator scores."""
import copy
import io
import math
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from studio.policy import RULE, active_version, score
from studio.visualization import missing_glyphs

METHOD = '基础 PMC：一级指标 = 二级 0/1 之和 / 二级指标数；原始 PMC = 一级指标得分之和'
SURFACE_METHOD = '每个相邻网格单元沿左上至右下对角线分为两个平面三角形，顶点间线性连接；中间位置不是新观测'
MAX_POLICIES = 8


def default_settings(version):
    primary=[i['id'] for i in version['indicators'] if i['level']==1]
    return {'radar_policy_ids':[], 'surface_policy_id':None, 'surface_order':primary,
            'style':'color', 'width_inches':8.0, 'height_inches':6.0, 'dpi':150,
            'font_size':10, 'show_title':True}


def chart_settings(project):
    """Settings are version-scoped; old scheme arrangements remain separately auditable."""
    version=active_version(project)
    if not version:
        return None
    container=project.setdefault('chart_settings',{'schema_version':1,'by_version':{}})
    return container['by_version'].setdefault(version['version_id'],default_settings(version))


def validate_chart_settings(project, container):
    if container is None:
        return
    if not isinstance(container,dict) or set(container)!={'schema_version','by_version'} or container['schema_version']!=1 or not isinstance(container['by_version'],dict):
        raise ValueError('政策图表设置无效。')
    versions={v['version_id']:v for v in project['versions']};policies={p['policy_id'] for p in project['policies']}
    keys={'radar_policy_ids','surface_policy_id','surface_order','style','width_inches','height_inches','dpi','font_size','show_title'}
    for version_id,settings in container['by_version'].items():
        if version_id not in versions or not isinstance(settings,dict) or set(settings)!=keys:
            raise ValueError('政策图表指标版本或设置无效。')
        primary=[i['id'] for i in versions[version_id]['indicators'] if i['level']==1]
        order=settings['surface_order']; selected=settings['radar_policy_ids']; surface=settings['surface_policy_id']
        if (not isinstance(order,list) or len(order)!=len(primary) or len(set(order))!=len(order) or set(order)!=set(primary)):
            raise ValueError('政策曲面指标排列无效。')
        if (not isinstance(selected,list) or len(selected)>MAX_POLICIES or len(set(selected))!=len(selected)
            or any(p not in policies for p in selected) or (surface is not None and surface not in policies)):
            raise ValueError('政策图表所选政策无效。')
        if settings['style'] not in ('color','mono') or not isinstance(settings['show_title'],bool):
            raise ValueError('政策图表样式无效。')
        for name,low,high in [('width_inches',4,16),('height_inches',3,12),('font_size',8,18)]:
            value=settings[name]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not low<=value<=high:
                raise ValueError('政策图表尺寸或字号无效。')
        if settings['dpi'] not in (150,300,600) or settings['width_inches']*settings['height_inches']*settings['dpi']**2>36000000:
            raise ValueError('政策图表分辨率或像素数超限。')


def drawing_data(project, policy_ids, version_id=None):
    version=active_version(project)
    if not version:
        raise ValueError('尚未确认指标体系。')
    if version_id is not None and version_id != version['version_id']:
        raise ValueError('只允许比较当前同一已确认指标版本的政策；不同版本不能混合。')
    if version['indicators'] != project['draft']:
        raise ValueError('指标草稿已修改；请确认新版本或撤销草稿修改后绘图。')
    known={p['policy_id']:p for p in project['policies']}
    if not isinstance(policy_ids,list) or len(set(policy_ids))!=len(policy_ids) or any(pid not in known for pid in policy_ids):
        raise ValueError('所选政策无效。')
    primary,totals=score(project,version['version_id']); totals={r['policy_id']:r for r in totals}
    ordered=[i for i in version['indicators'] if i['level']==1]; order={item['id']:n for n,item in enumerate(ordered,1)}
    by={(r['policy_id'],r['primary_id']):r for r in primary}
    rows=[];blocked=[]
    for pid in policy_ids:
        total=totals[pid]
        if total['raw_pmc'] is None:
            blocked.append({'policy_id':pid,'policy_name':known[pid]['title'],'reason':total['reason'] or '未完成编码'})
        for indicator in ordered:
            item=by[(pid,indicator['id'])]
            rows.append({'policy_id':pid,'policy_name':known[pid]['title'],'indicator_version':version['version_id'],
                         'indicator_id':indicator['id'],'indicator_name':indicator['name'],'indicator_order':order[indicator['id']],
                         'numerator':item['numerator'],'denominator':item['denominator'],'score':item['score'],
                         'policy_complete':total['raw_pmc'] is not None,'raw_pmc':total['raw_pmc'],
                         'text_revision_id':known[pid]['current_revision'],'calculation_method':METHOD})
    columns=['policy_id','policy_name','indicator_version','indicator_id','indicator_name','indicator_order',
             'numerator','denominator','score','policy_complete','raw_pmc','text_revision_id','calculation_method']
    return pd.DataFrame(rows,columns=columns),blocked


def surface_data(frame, order, require_complete=True):
    if len(order)!=9 or len(set(order))!=9:
        raise ValueError('3×3 曲面必须使用九个互不重复的一级指标。')
    if frame['policy_id'].nunique()!=1 or len(frame)!=9 or set(frame['indicator_id'])!=set(order):
        raise ValueError('3×3 曲面需要一份政策的完整九个一级指标。')
    if require_complete and (not frame['policy_complete'].all() or frame['score'].isna().any()):
        raise ValueError('政策编码尚未全部完成，不能生成曲面。')
    result=frame.set_index('indicator_id').loc[order].reset_index()
    result.insert(0,'matrix_row',[n//3+1 for n in range(9)])
    result.insert(1,'matrix_column',[n%3+1 for n in range(9)])
    result['surface_method']=SURFACE_METHOD
    return result


def _setup(settings,font_path,fmt):
    if fmt not in ('png','svg') or not font_path or not Path(font_path).is_file():
        raise ValueError('缺少支持中文的字体或图表格式不受支持。')
    w,h,dpi=settings['width_inches'],settings['height_inches'],settings['dpi']
    if not (4<=w<=16 and 3<=h<=12 and dpi in (150,300,600)) or w*h*dpi*dpi>36000000:
        raise ValueError('图表尺寸或分辨率超限。')
    fig=Figure(figsize=(w,h),layout='constrained',facecolor='white');FigureCanvasAgg(fig)
    return fig


def _save(fig, settings, font_path, labels, fmt, description):
    for text in fig.findobj(match=Text):
        text.set_fontproperties(FontProperties(fname=font_path,size=text.get_fontsize(),weight=text.get_fontweight()))
    fig.canvas.draw()
    box=fig.get_tightbbox(fig.canvas.get_renderer());w,h=settings['width_inches'],settings['height_inches']
    if box.x0 < -.02 or box.y0 < -.02 or box.x1 > w+.02 or box.y1 > h+.02:
        raise ValueError('标签超出图表画布，请增大尺寸、减小字号或缩短名称。')
    out=io.BytesIO()
    if fmt=='svg':
        from matplotlib import rc_context
        with rc_context({'svg.fonttype':'path'}):fig.savefig(out,format=fmt,dpi=settings['dpi'],metadata={'Description':description})
    else:fig.savefig(out,format=fmt,dpi=settings['dpi'],metadata={'Description':description})
    fig.clear()
    return out.getvalue(),missing_glyphs(font_path,labels)


def _verify_frame(frame):
    if frame['calculation_method'].ne(METHOD).any():
        raise ValueError('不同计算口径不能合并绘图。')
    for row in frame.itertuples():
        if (not isinstance(row.denominator,(int,float,np.integer,np.floating)) or row.denominator<=0
            or not isinstance(row.numerator,(int,float,np.integer,np.floating)) or not 0<=row.numerator<=row.denominator
            or not math.isclose(float(row.score),float(row.numerator)/float(row.denominator),rel_tol=1e-12,abs_tol=1e-12)):
            raise ValueError('绘图得分与实际分子、分母不一致。')


def render_radar(frame, settings, font_path, fmt='png', *, translate=str):
    if frame.empty or not frame['policy_complete'].all() or frame['score'].isna().any():
        raise ValueError('所选政策尚未全部完成编码，无法绘制雷达图。')
    versions=frame['indicator_version'].unique().tolist();policy_ids=frame['policy_id'].drop_duplicates().tolist()
    if len(versions)!=1 or len(policy_ids)>MAX_POLICIES:
        raise ValueError('雷达比较须使用同一指标版本，最多比较 8 份政策。')
    _verify_frame(frame)
    ref=frame[frame['policy_id']==policy_ids[0]].sort_values('indicator_order');ids=ref['indicator_id'].tolist();n=len(ids)
    if n<3: raise ValueError('雷达图至少需要三个一级指标；可查看数据表。')
    for pid in policy_ids:
        group=frame[frame['policy_id']==pid].sort_values('indicator_order')
        if group['indicator_id'].tolist()!=ids or group['indicator_name'].tolist()!=ref['indicator_name'].tolist() or group['indicator_order'].tolist()!=ref['indicator_order'].tolist():
            raise ValueError('政策的指标定义、顺序或版本不一致，不能比较。')
    fig=_setup(settings,font_path,fmt);ax=fig.add_subplot(111,polar=True)
    angles=np.linspace(0,2*np.pi,n,endpoint=False);angles_closed=np.r_[angles,angles[0]]
    ax.set_theta_offset(np.pi/2);ax.set_theta_direction(-1);ax.set_ylim(0,1)
    labels=[textwrap.fill(str(x),width=10,break_long_words=True) for x in ref['indicator_name']]
    ax.set_xticks(angles,labels=labels,fontsize=settings['font_size']*.9)
    ax.set_yticks([0,.25,.5,.75,1],labels=['0','0.25','0.5','0.75','1'])
    ax.tick_params(axis='x',pad=12);ax.tick_params(axis='y',labelsize=settings['font_size']*.75)
    ax.grid(color='#d7dedd' if settings['style']=='color' else '#d0d0d0',linewidth=.5)
    ax.spines['polar'].set_color('#9aa8a5' if settings['style']=='color' else '#777777');ax.spines['polar'].set_linewidth(.6)
    colors=['#588ba2','#57978c','#a57c62','#877da2','#b17778','#789369','#7e8795','#8d806b']
    lines=['-','--','-.',':','-','--','-.',':'];markers=['o','s','^','D','x','P','v','*']
    legend=[]
    for i,pid in enumerate(policy_ids):
        group=frame[frame['policy_id']==pid].sort_values('indicator_order');values=group['score'].astype(float).tolist()
        if any(not math.isfinite(v) or not 0<=v<=1 for v in values):raise ValueError('一级指标得分须在 0 到 1 之间。')
        label=textwrap.fill(str(group.iloc[0]['policy_name'])+' · '+pid[-8:],width=24,break_long_words=True);legend.append(label)
        ax.plot(angles_closed,values+[values[0]],color='#333333' if settings['style']=='mono' else colors[i],
                linestyle=lines[i],marker=markers[i],markersize=3.5,linewidth=1.2,label=label)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.13),ncol=min(2,len(policy_ids)),fontsize=settings['font_size']*.78,frameon=False)
    if settings['show_title']:ax.set_title(translate('基础 PMC 一级指标雷达图（轴范围 0–1）'),fontsize=settings['font_size']+1,pad=20)
    policy_descriptions=[f"{frame[frame['policy_id']==pid].iloc[0]['policy_name']} ({pid})" for pid in policy_ids]
    description=f"政策：{'；'.join(policy_descriptions)}；指标版本：{versions[0]}；{METHOD}"
    return _save(fig,settings,font_path,labels+legend+['基础 PMC 一级指标雷达图轴范围'],fmt,description)


def render_surface(matrix, settings, font_path, fmt='png', *, translate=str):
    if len(matrix)!=9 or not matrix['policy_complete'].all() or matrix['score'].isna().any():
        raise ValueError('3×3 曲面需要一份完整政策的九个一级指标。')
    if matrix['policy_id'].nunique()!=1 or matrix['indicator_version'].nunique()!=1 or list(zip(matrix['matrix_row'],matrix['matrix_column']))!=[(r,c) for r in (1,2,3) for c in (1,2,3)]:
        raise ValueError('曲面政策、指标版本或矩阵位置不一致。')
    if matrix['indicator_id'].nunique()!=9:
        raise ValueError('曲面指标不能重复。')
    _verify_frame(matrix)
    values=matrix['score'].astype(float).tolist()
    if any(not math.isfinite(v) or not 0<=v<=1 for v in values):raise ValueError('一级指标得分须在 0 到 1 之间。')
    fig=_setup(settings,font_path,fmt);ax=fig.add_subplot(111,projection='3d',computed_zorder=False)
    vertices=[(col-1,3-row,float(value)) for row,col,value in zip(matrix['matrix_row'],matrix['matrix_column'],values)]
    faces=[]
    for row in range(2):
        for col in range(2):
            nw=row*3+col;ne=nw+1;sw=nw+3;se=sw+1
            faces.extend([[vertices[nw],vertices[ne],vertices[se]],[vertices[nw],vertices[se],vertices[sw]]])
    mono=settings['style']=='mono'
    ax.add_collection3d(Poly3DCollection(faces,facecolors='#d9d9d9' if mono else '#cfe3e1',
                                        edgecolors='#676767' if mono else '#6c9092',linewidths=.45,alpha=.8,zorder=1))
    for (x,y,z),row,col in zip(vertices,matrix['matrix_row'],matrix['matrix_column']):
        ax.scatter([x],[y],[z],color='#222222',s=13,depthshade=False,zorder=5)
        ax.text(x,y,min(z+.025,1.0),f'{row}{col}:{z:.2f}',fontsize=settings['font_size']*.75,color='#222222',zorder=6)
    ax.set_xlim(-.15,2.15);ax.set_ylim(-.15,2.15);ax.set_zlim(0,1)
    ax.set_xticks([0,1,2],labels=['1','2','3']);ax.set_yticks([0,1,2],labels=['3','2','1'])
    ax.set_zticks([0,.25,.5,.75,1]);ax.set_xlabel(translate('列'),labelpad=3);ax.set_ylabel(translate('行'),labelpad=3);ax.set_zlabel(translate('一级得分（0–1）'),labelpad=3)
    ax.tick_params(labelsize=settings['font_size']*.7,pad=1);ax.view_init(elev=27,azim=-58)
    if settings['show_title']:ax.set_title(translate('基础 PMC 曲面（九个观测点）'),fontsize=settings['font_size']+1,pad=8)
    fig.text(.5,.02,translate('相邻节点按固定对角线划分三角面并线性连接；中间位置不是新观测'),ha='center',fontsize=settings['font_size']*.8)
    item=matrix.iloc[0]
    description=f"政策：{item['policy_name']} ({item['policy_id']})；指标版本：{item['indicator_version']}；{METHOD}；{SURFACE_METHOD}"
    return _save(fig,settings,font_path,['基础 PMC 曲面九个观测点列行一级得分相邻节点按固定对角线划分三角面并线性连接中间位置不是新观测'],fmt,description)
