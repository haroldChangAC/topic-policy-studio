import copy
import io
import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from PIL import Image
from streamlit.testing.v1 import AppTest

from test_topic_groups import example
from studio.annual import calculate,group_key
from studio.annual_heatmap import ALL,render as heat_render
from studio.grouped_charts import basis,cells,display_labels,evidence,for_scope,line_arrays,render_line,signature,source
from studio.project import open_project,save_project
from studio.runtime import ROOT
from studio.topic_groups import create,delete,update
from studio.visualization import find_chinese_font
from studio.workflow import csv_bytes


def sample():
    p,settings=example();f=p['metadata']['generated_fields']
    for index,year,group,topic,char in [(3,'2021','上海',1,'丙'),(4,'2022','北京',2,'丁'),(5,'2021','上海',-1,'戊'),(6,'','北京',0,'己'),(7,'2022','上海',3,'庚')]:
        doc=p['rows'].iloc[0].copy();doc[f['document_id']]=f'doc-{index}';doc[f['record_id']]=f'doc-{index}'
        doc['id']=f'D{index}';doc['正文']=char;doc['年份']=year;doc['地区']=group;doc[f['original_text']]=char
        seg=p['segments'].iloc[0].copy()
        for col in p['metadata']['original_columns']+[f['document_id'],f['record_id'],f['original_text']]:seg[col]=doc[col]
        seg[f['segment_id']]=f'seg-{index}';seg[f['segment_order']]=1;seg[f['start_char']]=0;seg[f['end_char']]=1;seg[f['segment_text']]=char;seg[f['topic_id']]=topic
        p['rows']=pd.concat([p['rows'],pd.DataFrame([doc])],ignore_index=True);p['segments']=pd.concat([p['segments'],pd.DataFrame([seg])],ignore_index=True)
    counts=p['segments'][f['topic_id']].value_counts()
    p['topics']=pd.DataFrame([{'Topic':int(t),'Count':int(counts[t]),'Name':f'{t}_标签','Keywords':'原始关键词'} for t in sorted(counts.index)])
    p['metadata']['counts'].update(input=len(p['rows']),segments=len(p['segments']))
    settings['exclude_missing_year']=False;settings['include_outliers']=True;p['visualization']['annual_panel']=settings
    gid=create(p,'G',[0,1]);return p,settings,gid


class GroupedChartsTests(unittest.TestCase):
    def test_exact_cross_year_scope_zero_missing_and_outliers(self):
        p,s,gid=sample();old=calculate(p,s)[1];docs,panel,parts,members=source(p,s)
        beijing=for_scope(p,docs,panel,parts,members,group_key('北京'))
        frame=cells(beijing,'segment_count',[2020,2021,2022],[gid,'topic:2'],s,members).set_index(['year','topic_group_id'])
        self.assertEqual((frame.loc[(2020,gid),'numerator'],frame.loc[(2020,gid),'denominator'],frame.loc[(2020,gid),'value']),(3,3,3))
        self.assertEqual(frame.loc[(2021,gid),'numerator'],0);self.assertTrue(pd.isna(frame.loc[(2021,gid),'value']))
        self.assertEqual(frame.loc[(2022,gid),'value'],0);self.assertEqual(frame.loc[(2022,gid),'missing_status'],'有纳入样本')
        doc=cells(beijing,'document_mention_rate',[2020],[gid],s,members).iloc[0];self.assertEqual((doc.numerator,doc.denominator,doc.value),(2,2,1))
        sh=for_scope(p,docs,panel,parts,members,group_key('上海'));r=cells(sh,'document_mention_rate',[2021],[gid],s,members).iloc[0]
        self.assertEqual((r.numerator,r.denominator,r.value),(1,2,.5))
        total=for_scope(p,docs,panel,parts,members,ALL);r=cells(total,'document_count',[2021],[gid],s,members).iloc[0]
        self.assertEqual((r.numerator,r.denominator),(1,2));self.assertNotIn('topic:0',panel.view_id.tolist());self.assertIn('topic:2',panel.view_id.tolist());self.assertIn('topic:-1',panel.view_id.tolist())
        sub=cells(beijing,'segment_count',[2020,2021,2022],[gid],s,members);xs,ys=line_arrays(sub)
        self.assertEqual(xs,[2020,2021,2022]);self.assertTrue(math.isnan(ys[1]));self.assertEqual(ys[2],0)
        xs,ys=line_arrays(sub[sub.year.isin([2020,2022])]);self.assertEqual(xs,[2020,2021,2022]);self.assertTrue(math.isnan(ys[1]))
        docs2,segs=evidence(p,docs,parts,members,group_key('北京'),2020,gid);self.assertEqual((len(docs2),len(segs)),(2,3))
        f=p['metadata']['generated_fields'];lookup=p['rows'].set_index(f['document_id'])
        for _,part in segs.iterrows():self.assertEqual(lookup.loc[part[f['document_id']],f['original_text']][part[f['start_char']]:part[f['end_char']]],part[f['segment_text']])
        s2=copy.deepcopy(s);s2['include_outliers']=False;d,pan,part,m=source(p,s2);r=cells(for_scope(p,d,pan,part,m,group_key('上海')),'segment_share',[2021],[gid],s2,m).iloc[0]
        self.assertEqual((r.numerator,r.denominator,r.value),(1,1,1));pd.testing.assert_frame_equal(old,calculate(p,s)[1])

    def test_same_names_update_and_old_state(self):
        p,s,g1=sample();g2=create(p,'G',[2,3]);self.assertNotEqual(g1,g2)
        labels=display_labels({g1:'G',g2:'G'});self.assertNotEqual(labels[g1],labels[g2]);old=basis(p,s);sig=signature(old)
        collision_a='group-'+'abcdef12'+'0'*24;collision_b='group-'+'abcdef12'+'0'*23+'1'
        collision_labels=display_labels({collision_a:'G',collision_b:'G'});self.assertNotEqual(collision_labels[collision_a],collision_labels[collision_b])
        reopened=open_project(save_project(p));self.assertEqual(len(reopened['visualization']['topic_groups']['groups']),2);self.assertEqual(reopened['project_info']['version'],5)
        update(p,g1,'更名',[0,1]);self.assertNotEqual(sig,signature(basis(p,s)))
        d,pan,parts,m=source(p,s);self.assertEqual(pan.loc[pan.view_id.eq(g1),'view_name'].iloc[0],'更名')
        delete(p,g2);update(p,g1,'更名',[0,3]);d,pan,parts,m=source(p,s);self.assertEqual(m[g1],[0,3]);self.assertIn('topic:1',pan.view_id.tolist())
        delete(p,g1);d,pan,parts,m=source(p,s);self.assertNotIn(g1,pan.view_id.tolist());self.assertIn('topic:0',pan.view_id.tolist())
        reopened=open_project(save_project(p));self.assertFalse(reopened['visualization']['topic_groups']['groups'])

    def test_png_svg_csv_and_line_break(self):
        p,s,gid=sample();docs,panel,parts,members=source(p,s);scoped=for_scope(p,docs,panel,parts,members,group_key('北京'))
        frame=cells(scoped,'segment_share',[2020,2021,2022],[gid,'topic:2'],s,members)
        self.assertEqual(set(frame.view),{'人工主题归组'});csv=pd.read_csv(io.BytesIO(csv_bytes(frame)),keep_default_na=False)
        for i,row in csv.iterrows():
            source_row=frame.iloc[i];self.assertEqual((int(row.numerator),int(row.denominator)),(source_row.numerator,source_row.denominator))
            self.assertEqual(row.member_topic_ids,source_row.member_topic_ids);self.assertEqual(row.missing_status,source_row.missing_status)
            if pd.isna(source_row.value):self.assertEqual(row.value,'')
            else:self.assertAlmostEqual(float(row.value),float(source_row.value))
        font,_=find_chinese_font();seen=[];limits=[];orig=Axes.plot;original_xlim=Axes.set_xlim
        def spy(ax,x,y,*args,**kwargs):seen.append((list(x),list(y)));return orig(ax,x,y,*args,**kwargs)
        def capture_xlim(ax,*args,**kwargs):
            if len(args)==2:limits.append((args[0],args[1]))
            return original_xlim(ax,*args,**kwargs)
        for style in ['color','mono']:
            with patch.object(Axes,'plot',spy),patch.object(Axes,'set_xlim',capture_xlim):png,_=render_line(frame,10,6,150,font,{'style':style},'png')
            self.assertEqual(Image.open(io.BytesIO(png)).size,(1500,900));self.assertTrue(any(any(math.isnan(v) for v in ys) for _,ys in seen))
            self.assertIn((2019.5,2022.5),limits)
            svg,_=render_line(frame,10,6,150,font,{'style':style},'svg');self.assertIn(b'<svg',svg)
            heat=frame.assign(topic_id=frame.topic_group_id,topic_label=frame.topic_group_name)
            png,_=heat_render(heat,10,6,150,font,{'style':style},'png',grouped=True);self.assertEqual(Image.open(io.BytesIO(png)).size,(1500,900))
            svg,_=heat_render(heat,10,6,150,font,{'style':style},'svg',grouped=True);self.assertIn(b'<svg',svg)

    def test_page_settings_restore_and_recalculate_after_edit(self):
        p,s,gid=sample();raw=save_project(p);app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=100).run();app.radio(key='work_mode').set_value('打开已有项目').run()
        app.file_uploader(key='project_upload').set_value(('test.zip',raw,'application/zip')).run(timeout=100)
        next(r for r in app.radio if r.label=='年度统计主题视图').set_value('人工归组').run(timeout=100)
        self.assertFalse(app.exception);self.assertTrue(any(x.label=='下载归组折线图数据 CSV' for x in app.get('download_button')))
        saved=app.session_state['opened_project']['result'];before=saved['visualization']['grouped_annual_charts']['signature'];raw=save_project(saved)
        reopened=open_project(raw);self.assertEqual(reopened['project_info']['version'],5);self.assertEqual(reopened['visualization']['annual_view'],'grouped');self.assertEqual(reopened['visualization']['grouped_annual_charts'],saved['visualization']['grouped_annual_charts'])
        update(saved,gid,'已修改',[0,1]);app.run(timeout=100);self.assertFalse(app.exception)
        self.assertNotEqual(before,saved['visualization']['grouped_annual_charts']['signature']);self.assertTrue(any('重新计算' in x.value for x in app.warning))
        update(saved,gid,'已修改',[0,2]);app.run(timeout=100);self.assertFalse(app.exception)
        self.assertEqual(next(x for x in saved['visualization']['grouped_annual_charts']['basis']['groups'] if x['group_id']==gid)['members'],[0,2])
        delete(saved,gid);app.run(timeout=100);self.assertFalse(app.exception)
        self.assertFalse(saved['visualization']['grouped_annual_charts']['basis']['groups'])
        next(r for r in app.radio if r.label=='年度统计主题视图').set_value('原始主题').run(timeout=100)
        self.assertFalse(app.exception);self.assertEqual(open_project(save_project(saved))['visualization']['annual_view'],'original')

if __name__=='__main__':unittest.main()
