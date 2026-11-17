"""Publication export from exactly the same selected links as the interactive chart."""
import io
import math
import re
import textwrap
from collections import defaultdict
from pathlib import Path

from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MPath

from studio.visualization import find_chinese_font, missing_glyphs


def default_export():
    return {'width':14.0,'height':9.0,'dpi':150,'font_size':9}


def validate_export(obj):
    if not isinstance(obj,dict) or set(obj)!={'width','height','dpi','font_size'}:
        raise ValueError('静态桑基图设置格式无效。')
    for name,low,high in [('width',6,16),('height',4,14),('font_size',7,16)]:
        value=obj[name]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not low<=value<=high:
            raise ValueError('静态桑基图尺寸或字号超出允许范围。')
    if obj['dpi'] not in (150,300,600) or obj['width']*obj['height']*obj['dpi']**2>36_000_000:
        raise ValueError('图像超过 3600 万像素上限，请降低尺寸或 DPI。')


def selected_data(nodes,frame):
    """Keep chart ordering and exact edge weights aligned with Plotly and CSV."""
    selected=frame[frame.in_graph]
    if selected.empty:raise ValueError('没有符合筛选条件的有效连线，不能生成静态图。')
    if len(selected)>80:raise ValueError('静态图连线超过 80 条，请提高门槛或降低连线上限。')
    keys=set(zip(selected.source_year,selected.source_topic_id))|set(zip(selected.target_year,selected.target_topic_id))
    active=sorted((n for n in nodes if (n['year'],n['topic_id']) in keys),key=lambda n:(n['year'],n['topic_id']))
    edges=[{'source':(int(e.source_year),int(e.source_topic_id)), 'target':(int(e.target_year),int(e.target_topic_id)),
            'weight':float(e.line_weight)} for e in selected.itertuples()]
    if any(not math.isfinite(e['weight']) or e['weight']<=0 for e in edges):raise ValueError('存在不能绘图的关联权重。')
    return active,edges


def render(nodes,frame,settings,export,fmt):
    validate_export(export)
    if fmt not in ('png','svg'):raise ValueError('不支持的静态图格式。')
    active,edges=selected_data(nodes,frame)
    if len(active)>30:raise ValueError('静态图节点超过 30 个，请减少主题以保持可读。')
    font_path,family=find_chinese_font()
    if not font_path or not Path(font_path).is_file():raise ValueError('未找到可用的中文字体，无法导出静态图。')
    width,height,dpi,size=export['width'],export['height'],export['dpi'],export['font_size']
    labels={}
    for n in active:
        label=re.sub(r'^'+re.escape(str(n['topic_id']))+r'_', '', str(n['label']))
        title=f"{n['year']} 主题 {n['topic_id']}"
        name=textwrap.fill(label.replace('_','、'),width=13,break_long_words=True,break_on_hyphens=False)
        labels[(n['year'],n['topic_id'])]=title+'\n'+name+f"\n片段 {n['segment_count']} · 文档 {n['document_count']}"
    note='带宽：余弦相似度权重；节点高度：流入或流出权重的较大值；不代表文档或片段流量。'
    title='相邻自然年份主题相似关联（非转移）'
    missing=missing_glyphs(font_path,list(labels.values())+[note,title,'年份'])
    if missing:raise ValueError('所选中文字体缺少字符：'+''.join(missing[:20]))
    years=sorted({n['year'] for n in active})
    if len(years)<2:raise ValueError('至少需要两个有效年份。')
    if width/(len(years)-1)<3:raise ValueError('年份太多或画布太窄；请减少年份或增大宽度。')
    fig=Figure(figsize=(width,height),dpi=dpi,facecolor='white');FigureCanvasAgg(fig)
    ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,width);ax.set_ylim(0,height);ax.axis('off')
    fp=FontProperties(fname=font_path,size=size)
    xleft=width*.13;xright=width*.70
    xpos={year:xleft+(xright-xleft)*i/(len(years)-1) for i,year in enumerate(years)}
    gap=height*.065;bottom=height*.12;top=height*.88;available=top-bottom
    incoming=defaultdict(float);outgoing=defaultdict(float)
    for edge in edges:outgoing[edge['source']]+=edge['weight'];incoming[edge['target']]+=edge['weight']
    capacity={key:max(incoming[key],outgoing[key]) for key in set(incoming)|set(outgoing)}
    years_nodes={y:[n for n in active if n['year']==y] for y in years}
    scale=min((available/len(group)*.54)/max(capacity[(n['year'],n['topic_id'])] for n in group) for group in years_nodes.values())
    positions={}
    for year,group in years_nodes.items():
        slot=available/len(group)
        for i,n in enumerate(group):
            key=(year,n['topic_id']);cy=top-(i+.5)*slot;half=capacity[key]*scale/2
            positions[key]={'low':cy-half,'high':cy+half,'center':cy,'slot':slot}
    out_cursor={k:p['center']-outgoing[k]*scale/2 for k,p in positions.items()}
    in_cursor={k:p['center']-incoming[k]*scale/2 for k,p in positions.items()}
    color='#8aaeb5' if settings['style']=='color' else '#888888'
    node_color='#608b94' if settings['style']=='color' else '#454545'
    node_width=width*.008
    for edge in edges:
        a,b=edge['source'],edge['target'];thickness=edge['weight']*scale
        y0=out_cursor[a];y1=in_cursor[b];out_cursor[a]+=thickness;in_cursor[b]+=thickness
        x0=xpos[a[0]]+node_width;x1=xpos[b[0]];dx=(x1-x0)*.48
        verts=[(x0,y0),(x0+dx,y0),(x1-dx,y1),(x1,y1),(x1,y1+thickness),
               (x1-dx,y1+thickness),(x0+dx,y0+thickness),(x0,y0+thickness),(x0,y0)]
        codes=[MPath.MOVETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.LINETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MPath(verts,codes),facecolor=color,edgecolor='none',alpha=.30,zorder=1))
    text_items=[]
    for n in active:
        key=(n['year'],n['topic_id']);p=positions[key];x=xpos[key[0]]
        ax.add_patch(Rectangle((x,p['low']),node_width,p['high']-p['low'],facecolor=node_color,edgecolor='none',zorder=3))
        text_items.append(ax.text(x+node_width+width*.008,p['center'],labels[key],ha='left',va='center',
                                  fontproperties=fp,linespacing=1.12,zorder=4,
                                  bbox={'facecolor':'white','edgecolor':'none','alpha':.92,'pad':1.3}))
    other_text=[ax.text(xpos[year],height*.915,str(year),ha='left',va='center',fontproperties=FontProperties(fname=font_path,size=size+1)) for year in years]
    other_text.append(ax.text(width*.035,height*.975,title,ha='left',va='top',fontproperties=FontProperties(fname=font_path,size=size+2)))
    other_text.append(ax.text(width*.035,height*.035,note,ha='left',va='bottom',fontproperties=FontProperties(fname=font_path,size=max(7,size-1))))
    fig.canvas.draw();renderer=fig.canvas.get_renderer();box=fig.bbox
    extents=[t.get_window_extent(renderer).expanded(1.02,1.05) for t in text_items]
    if any(b.x0<0 or b.y0<box.height*bottom/height or b.x1>box.x1 or b.y1>box.height*top/height for b in extents):
        raise ValueError('中文标签超出画布，请增大尺寸或减少主题。')
    if any(a.overlaps(b) for i,a in enumerate(extents) for b in extents[i+1:]):
        raise ValueError('静态图标签重叠，请增大尺寸、减小字号或减少主题。')
    for text in other_text:
        bound=text.get_window_extent(renderer)
        if bound.x0<0 or bound.y0<0 or bound.x1>box.x1 or bound.y1>box.y1:
            raise ValueError('图题或说明可能被裁切，请增大画布。')
    buffer=io.BytesIO()
    if fmt=='svg':
        with rc_context({'svg.fonttype':'path'}):fig.savefig(buffer,format=fmt,dpi=dpi)
    else:fig.savefig(buffer,format=fmt,dpi=dpi)
    fig.clear()
    return buffer.getvalue(),{'font':family,'font_path':font_path,'node_count':len(active),'edge_count':len(edges),
                              'edge_weights':[e['weight'] for e in edges], 'node_ids':[[n['year'],n['topic_id']] for n in active],
                              'ribbon_thickness_inches':[e['weight']*scale for e in edges],
                              'node_heights_inches':{f'{year}:{topic}':p['high']-p['low'] for (year,topic),p in positions.items()},
                              'layout':'静态版单独排布，节点和连线数据与交互图一致','label_count':len(text_items)}
