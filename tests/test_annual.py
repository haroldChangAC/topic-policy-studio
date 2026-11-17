import copy
import io
import json
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image
from streamlit.testing.v1 import AppTest

from studio.annual import calculate, parse_year, render, validate_settings, METRICS
from studio.corpus import read_csv, prepare
from studio.project import save_project, open_project
from studio.topic_names import rename
from studio.visualization import default_visualization, find_chinese_font
from studio.runtime import ROOT
from studio.workflow import csv_bytes


def fixture():
    source=('id,正文,年份,地区\nD1,甲乙丙,2020,北京\nD2,丁戊,2020,北京\n'
            'D3,己,2021,上海\nD4,庚,not-a-date,\nD5,,2021,上海\nD6,甲乙丙,2021,北京\n').encode()
    prep=prepare(read_csv(source),'正文','deduplicate');f=prep.fields;rows=prep.rows.copy();rows[f['document_id']]=rows[f['record_id']];segments=[]
    for idx,assignment in [(0,[0,0,1]),(1,[1,-1]),(2,[-1]),(3,[1])]:
        row=rows.iloc[idx].copy();rows.loc[rows.index[idx],f['status']]='modeled';row[f['status']]='modeled'
        for n,topic in enumerate(assignment):
            v=row.to_dict();v.update({f['segment_id']:f'S{idx}-{n}',f['segment_order']:n+1,f['start_char']:n,f['end_char']:n+1,
                 f['segment_text']:row[f['original_text']][n:n+1],f['topic_id']:topic});segments.append(v)
    segs=pd.DataFrame(segments);counts=segs[f['topic_id']].value_counts()
    topics=pd.DataFrame([{'Topic':int(t),'Count':int(counts[t]),'Name':f'{t}_原始标签','Keywords':'词'} for t in sorted(counts.index)])
    result={'rows':rows,'segments':segs,'topics':topics,'metadata':{'schema_version':2,'run_id':'annual-test','text_column':'正文','model_id':'synthetic','model_revision':'test',
             'generated_fields':f,'original_columns':prep.corpus.frame.columns.tolist(),'counts':{'input':len(rows),'segments':len(segs)}},
            'topic_edits':{'labels':{},'history':[]},'visualization':default_visualization()}
    cfg={'year_field':'年份','group_field':'地区','exclude_missing_year':True,'exclude_missing_group':True,
         'include_outliers':True,'metric':'segment_count','selected_groups':[],'selected_topics':[],'rule_version':1}
    return result,cfg


class AnnualTests(unittest.TestCase):
    def test_hand_calculation_and_outlier_denominators(self):
        p,c=fixture();docs,panel,parts=calculate(p,c);x=panel[(panel.group=='北京') & (panel.year==2020)].set_index('topic_id')
        self.assertEqual((x.loc[0,'segment_count'],x.loc[0,'segment_denominator'],x.loc[0,'segment_share']),(2,5,0.4))
        self.assertEqual((x.loc[1,'segment_count'],x.loc[1,'document_count'],x.loc[1,'document_mention_rate']),(2,2,1.0))
        self.assertEqual((x.loc[0,'document_count'],x.loc[0,'document_denominator'],x.loc[0,'document_mention_rate']),(1,2,0.5))
        self.assertEqual(x.loc[-1,'segment_count'],1);self.assertEqual(x['document_mention_rate'].sum(),2.0)
        c['include_outliers']=False;docs,panel,parts=calculate(p,c);x=panel[(panel.group=='北京') & (panel.year==2020)].set_index('topic_id')
        self.assertEqual(x.loc[0,'segment_denominator'],4);self.assertEqual(x.loc[0,'segment_share'],.5);self.assertNotIn(-1,x.index)
        sh=panel[(panel.group=='上海') & (panel.year==2021)];self.assertTrue(sh.segment_share.isna().all());self.assertTrue(sh.document_mention_rate.isna().all())
        self.assertEqual(docs.loc[docs.document_id.eq(p['rows'].iloc[2][p['metadata']['generated_fields']['document_id']]),'reason'].iloc[0],'仅有离群片段，已排除')

    def test_missing_and_no_sample_cells(self):
        p,c=fixture();docs,panel,parts=calculate(p,c)
        self.assertEqual(int(docs.year_missing.sum()),1);self.assertEqual(int(docs.group_missing.sum()),1)
        self.assertTrue(any(docs.reason=='无建模片段（有效原文）'))
        self.assertTrue(((panel.group=='北京') & (panel.year==2021)).any())
        empty=panel[(panel.group=='上海')&(panel.year==2020)];self.assertTrue(empty.segment_share.isna().all());self.assertTrue(empty.document_mention_rate.isna().all())
        c.update(exclude_missing_year=False,exclude_missing_group=False);docs,panel,parts=calculate(p,c)
        missing=panel[panel.group_missing & panel.year.isna()];self.assertEqual(missing[missing.topic_id.eq(1)].document_count.iloc[0],1)
        c['group_field']=None;docs,panel,_=calculate(p,c);self.assertEqual(set(panel['group']),{'全部语料'})
        for bad in ['2020-02-30','2020x','21','2021/13/01','2021-01-01T29:00:00','',None,True]:self.assertIsNone(parse_year(bad))
        for value in ['2020','2020-02-29','2020/02/29','2020年2月29日']:self.assertEqual(parse_year(value),2020)

    def test_rename_project_csv_and_graph(self):
        p,c=fixture();rename(p,0,'自定义名称');docs,panel,_=calculate(p,c)
        c['selected_groups']=docs.loc[docs.selected_by_metadata,'group_key'].unique().tolist();c['selected_topics']=[-1,0,1]
        p['visualization']['annual_panel']=c;validate_settings(p,c)
        restored=open_project(save_project(p));self.assertEqual(restored['visualization']['annual_panel'],c)
        _,again,_=calculate(restored,c);pd.testing.assert_frame_equal(panel,again)
        self.assertTrue(again.loc[again.topic_id.eq(0),'topic_label'].eq('自定义名称').all())
        frame=pd.read_csv(io.BytesIO(csv_bytes(again)),keep_default_na=False);self.assertEqual(len(frame),len(panel));self.assertIn('document_denominator',frame)
        font,_=find_chinese_font()
        for fmt in ['png','svg']:
            raw,missing=render(panel,'segment_share',c['selected_groups'],[0,1],8,5,150,font,fmt=fmt)
            if fmt=='png':self.assertEqual(Image.open(io.BytesIO(raw)).size,(1200,750))
            else:self.assertIn(b'<svg',raw)
        original=copy.deepcopy(p);original['visualization'].pop('annual_panel');restored=open_project(save_project(original));self.assertNotIn('annual_panel',restored['visualization'])
        c=copy.deepcopy(c);c['year_field']='不存在';p['visualization']['annual_panel']=c
        with self.assertRaisesRegex(ValueError,'字段'):save_project(p)

    def test_page_restored_project(self):
        p,c=fixture();raw=save_project(p)
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
        app.radio(key='work_mode').set_value('打开已有项目').run()
        app.file_uploader(key='project_upload').set_value(('sample.hpts.zip',raw,'application/zip')).run(timeout=90)
        self.assertFalse(app.exception)
        y=next(x for x in app.selectbox if x.label=='选择年份或日期字段');y.select('年份')
        g=next(x for x in app.selectbox if x.label=='选择分组字段（可选）');g.select(g.options.index('地区'))
        app.run(timeout=90)
        next(x for x in app.button if x.label=='计算年度主题面板').click().run(timeout=90)
        self.assertFalse(app.exception)
        saved=app.session_state['opened_project']['result'];self.assertEqual(saved['visualization']['annual_panel']['year_field'],'年份')
        reopened=open_project(save_project(saved));self.assertEqual(reopened['visualization']['annual_panel']['group_field'],'地区')
        self.assertTrue(any(x.label=='下载年度主题面板 CSV' for x in app.get('download_button')))
        g=next(x for x in app.selectbox if x.label=='选择分组字段（可选）');g.select(0).run(timeout=90)
        next(x for x in app.button if x.label=='计算年度主题面板').click().run(timeout=90)
        self.assertFalse(app.exception)
        self.assertIsNone(app.session_state['opened_project']['result']['visualization']['annual_panel']['group_field'])


if __name__=='__main__':unittest.main()
