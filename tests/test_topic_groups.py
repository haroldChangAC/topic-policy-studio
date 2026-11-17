import copy
import io
import unittest

import pandas as pd
from streamlit.testing.v1 import AppTest
from test_annual import fixture
from studio.annual import calculate
from studio.project import open_project,save_project
from studio.runtime import ROOT
from studio.topic_groups import annual,create,delete,mapping,summary,trace,update
from studio.topic_names import rename
from studio.workflow import csv_bytes


def example():
    p,settings=fixture();f=p['metadata']['generated_fields'];rows=p['rows'];segs=p['segments'];doc1=rows.iloc[0][f['document_id']];doc2=rows.iloc[1][f['document_id']]
    for doc,text in [(doc1,'甲乙'),(doc2,'丁')]:
        rows.loc[rows[f['document_id']].eq(doc),'正文']=text;rows.loc[rows[f['document_id']].eq(doc),f['original_text']]=text
        segs.loc[segs[f['document_id']].eq(doc),'正文']=text;segs.loc[segs[f['document_id']].eq(doc),f['original_text']]=text
    segs=segs[~((segs[f['document_id']].eq(doc1)&segs[f['segment_order']].eq(3)) | (segs[f['document_id']].eq(doc2)&segs[f['segment_order']].eq(2)))].copy()
    segs.loc[segs[f['document_id']].eq(doc1)&segs[f['segment_order']].eq(2),f['topic_id']]=1
    segs.loc[segs[f['document_id']].eq(doc2),f['topic_id']]=0
    p['rows']=rows[rows[f['document_id']].isin([doc1,doc2])].copy()
    segs=segs[segs[f['document_id']].isin([doc1,doc2])].copy()
    p['segments']=segs;p['metadata']['counts'].update(input=len(p['rows']),segments=len(segs))
    counts=segs[f['topic_id']].value_counts();p['topics']=p['topics'][p['topics']['Topic'].isin(counts.index)].copy();p['topics']['Count']=p['topics']['Topic'].map(counts).astype(int)
    p['visualization']['annual_panel']=settings
    return p,settings


class GroupTests(unittest.TestCase):
    def test_hand_count_distinct_documents_and_preserve_source(self):
        p,settings=example();f=p['metadata']['generated_fields'];before=p['segments'].copy(deep=True);metadata=copy.deepcopy(p['metadata']);old_panel=calculate(p,settings)[1]
        gid=create(p,'G',[0,1]);s=summary(p).set_index('view_id').loc[gid]
        self.assertEqual((s.segment_count,s.document_count),(3,2));self.assertAlmostEqual(s.segment_share,3/len(before))
        docs,panel,parts=annual(p,settings);row=panel[(panel['group']=='北京') & (panel.year==2020)&(panel.view_id==gid)].iloc[0]
        self.assertEqual((row.segment_count,row.segment_denominator,row.document_count,row.document_denominator),(3,3,2,2))
        self.assertEqual((row.segment_share,row.document_mention_rate),(1,1))
        pd.testing.assert_frame_equal(before,p['segments']);self.assertEqual(metadata,p['metadata']);pd.testing.assert_frame_equal(old_panel,calculate(p,settings)[1])
        m=mapping(p);self.assertTrue(m.keyword_note.str.contains('未重新计算').all());self.assertNotIn('weight',m.columns)
        a=trace(p);self.assertEqual(len(a),len(before));self.assertTrue({'document_id','segment_id','start_char','end_char','original_topic_id','raw::正文'}.issubset(a.columns))
        for _,item in a.iterrows():
            text=p['rows'].set_index(f['document_id']).loc[item.document_id,f['original_text']]
            self.assertEqual(text[int(item.start_char):int(item.end_char)],item.segment_text)
        restored=open_project(save_project(p));self.assertEqual(restored['visualization']['topic_groups'],p['visualization']['topic_groups'])
        pd.testing.assert_frame_equal(summary(restored),summary(p));pd.testing.assert_frame_equal(annual(restored,settings)[1],panel)
        self.assertIn('G',csv_bytes(mapping(restored)).decode('utf-8-sig'))

    def test_update_rename_delete_conflicts_and_outliers(self):
        p,_=example();gid=create(p,'政策与文本',[0,1]);self.assertFalse(update(p,gid,'政策与文本',[0,1]));self.assertEqual(len(p['visualization']['topic_groups']['history']),1)
        for invalid in ('','过长'*51,'换行\n名称'):
            with self.assertRaises(ValueError):create(p,invalid,[0,1])
        with self.assertRaisesRegex(ValueError,'离群'):create(p,'不允许',[-1,0])
        with self.assertRaisesRegex(ValueError,'其他组'):create(p,'冲突',[0,1])
        self.assertTrue(update(p,gid,'新组名',[0,1]));self.assertEqual(p['visualization']['topic_groups']['groups'][0]['name'],'新组名')
        rename(p,0,'原主题新名称');self.assertEqual(mapping(p).set_index('original_topic_id').loc[0,'original_topic_name'],'原主题新名称')
        self.assertTrue(delete(p,gid));self.assertFalse(p['visualization']['topic_groups']['groups']);self.assertTrue(mapping(p).group_id.eq('').all())
        new=create(p,'再次分组',[0,1]);self.assertNotEqual(gid,new);self.assertEqual([x['operation'] for x in p['visualization']['topic_groups']['history']],['create','update','delete','create'])
        self.assertEqual(open_project(save_project(p))['visualization']['topic_groups'],p['visualization']['topic_groups'])
        bad=copy.deepcopy(p);bad['visualization']['topic_groups']['groups'][0]['members']=[-1,1]
        with self.assertRaisesRegex(ValueError,'不一致'):save_project(bad)

    def test_adjust_members_and_legacy_project(self):
        p,settings=fixture();f=p['metadata']['generated_fields']
        for number,char in [(2,'子'),(3,'丑')]:
            doc=p['rows'].iloc[0].copy();doc[f['document_id']]=f'extra-{number}';doc[f['record_id']]=f'extra-{number}'
            doc['id']=f'D{number+5}';doc['正文']=char;doc[f['original_text']]=char
            seg=p['segments'].iloc[0].copy()
            for col in p['metadata']['original_columns']+[f['document_id'],f['record_id'],f['original_text']]:seg[col]=doc[col]
            seg[f['segment_id']]=f'extra-segment-{number}';seg[f['segment_order']]=1;seg[f['start_char']]=0;seg[f['end_char']]=1;seg[f['segment_text']]=char;seg[f['topic_id']]=number
            p['rows']=pd.concat([p['rows'],pd.DataFrame([doc])],ignore_index=True);p['segments']=pd.concat([p['segments'],pd.DataFrame([seg])],ignore_index=True)
            p['topics']=pd.concat([p['topics'],pd.DataFrame([{'Topic':number,'Count':1,'Name':f'{number}_标签','Keywords':'词'}])],ignore_index=True)
        p['metadata']['counts'].update(input=len(p['rows']),segments=len(p['segments']))
        old=open_project(save_project(p));self.assertNotIn('topic_groups',old['visualization'])
        gid=create(p,'可编辑组',[0,1]);self.assertTrue(update(p,gid,'可编辑组',[0,2]))
        self.assertEqual(mapping(p).set_index('original_topic_id').loc[1,'group_id'],'')
        create(p,'另一个组',[1,3])
        with self.assertRaisesRegex(ValueError,'其他组'):update(p,gid,'可编辑组',[0,1])
        reopened=open_project(save_project(p));self.assertEqual(len(reopened['visualization']['topic_groups']['groups']),2)
        self.assertEqual(open_project(save_project(old))['metadata'],old['metadata'])

    def test_annual_filters_missing_year_group_and_outlier(self):
        p,c=fixture();gid=create(p,'组',[0,1]);c.update(exclude_missing_year=False,exclude_missing_group=False,include_outliers=False)
        docs,base,parts=calculate(p,c);docs,grouped,_=annual(p,c)
        self.assertNotIn('topic:-1',grouped.view_id.tolist())
        miss=grouped[grouped.group_missing & grouped.year.isna() & grouped.view_id.eq(gid)].iloc[0]
        self.assertEqual((miss.segment_count,miss.document_count,miss.segment_denominator,miss.document_denominator),(1,1,1,1))
        for row in grouped.itertuples():
            basecell=base[base.group_key.eq(row.group_key)&(base.year.isna() if pd.isna(row.year) else base.year.eq(row.year))].iloc[0]
            self.assertEqual((row.segment_denominator,row.document_denominator),(basecell.segment_denominator,basecell.document_denominator))
        before=copy.deepcopy(p['visualization']['topic_groups']);reopened=open_project(save_project(p));update(reopened,gid,'修改后',[0,1]);delete(reopened,gid)
        self.assertEqual(p['visualization']['topic_groups'],before)

    def test_page_create_and_annual_view(self):
        p,settings=example();app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=100).run()
        app.radio(key='work_mode').set_value('打开已有项目').run();app.file_uploader(key='project_upload').set_value(('group.zip',save_project(p),'application/zip')).run(timeout=100)
        self.assertFalse(app.exception)
        next(x for x in app.text_input if x.label=='新上级主题组名称').set_value('页面归组')
        next(x for x in app.multiselect if x.label=='选择至少两个未归组主题').set_value([0,1])
        next(x for x in app.button if x.label=='创建主题组').click().run(timeout=100)
        self.assertFalse(app.exception);saved=app.session_state['opened_project']['result'];self.assertEqual(saved['visualization']['topic_groups']['groups'][0]['name'],'页面归组')
        next(x for x in app.radio if x.label=='年度统计主题视图').set_value('人工归组').run(timeout=100)
        self.assertFalse(app.exception);self.assertTrue(any(x.label=='下载归组年度面板 CSV' for x in app.get('download_button')))
        restored=open_project(save_project(saved));self.assertEqual(restored['visualization']['topic_groups'],saved['visualization']['topic_groups'])
        app.file_uploader(key='project_upload').set_value(('edited.zip',save_project(saved),'application/zip')).run(timeout=100)
        self.assertFalse(app.exception);self.assertEqual(app.session_state['opened_project']['result']['visualization']['topic_groups']['groups'][0]['name'],'页面归组')
        next(x for x in app.text_input if x.label=='修改主题组名称').set_value('重新打开后修改')
        next(x for x in app.button if x.label=='保存主题组修改').click().run(timeout=100)
        self.assertFalse(app.exception);self.assertEqual(app.session_state['opened_project']['result']['visualization']['topic_groups']['groups'][0]['name'],'重新打开后修改')


if __name__=='__main__':unittest.main()
