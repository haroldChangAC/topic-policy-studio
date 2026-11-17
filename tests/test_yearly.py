import copy
import io
import json
import unittest
import zipfile
from unittest.mock import patch
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from test_paper_style import sample_result
from studio.deferred_exports import create_export, fingerprint
from studio.modeling import fit_topics
from studio.project import open_project,save_project
from studio.runtime import ROOT
from studio.yearly import preview,run,tables,validate


def sample():
    r=sample_result();years=['2022','2023','无法确定','2024'];f=r['metadata']['generated_fields']
    r['rows']['year']=years;r['segments']['year']=years;r['metadata']['original_columns'].append('year')
    return r


def fake_fit(texts,encoder,**kwargs):
    if '第3篇' in texts[0]:raise ValueError('本年度样本不足')
    tid=0 if '第0篇' in texts[0] else -1
    return {'topics':[tid],'strengths':np.array([0.5]),'topic_info':pd.DataFrame([{'Topic':tid,'Count':1,'Name':f'{tid}_政策','Keywords':'政策'}]),
            'topic_vectors':[{'topic_id':tid,'values':[1.,0.]}],'keyword_weights':[{'topic_id':tid,'rank':1,'keyword':'政策','weight':0.1}],
            'parameters':{'random_state':42},'timings_seconds':{'embedding':0.,'bertopic_fit':0.},
            'topic_vector_dimension':2,'model_token_limit':128,'max_input_tokens':20}


class YearlyTests(unittest.TestCase):
    def test_missing_outliers_failed_and_roundtrip(self):
        original=sample();r=copy.deepcopy(original);p=preview(r,'year')
        self.assertEqual(p['year'].isna().sum(),1)
        self.assertNotEqual(fingerprint(r),fingerprint({**r,'yearly_analysis':{'modified':True}}))
        r['yearly_analysis']=run(r,'year',[2022,2023,2024],None,fit=fake_fit)
        runs=r['yearly_analysis']['results']
        self.assertEqual([x['status'] for x in runs],['completed','completed','failed'])
        self.assertEqual(runs[1]['outliers'],1);self.assertIn('样本不足',runs[2]['error'])
        self.assertEqual(r['yearly_analysis']['missing_document_ids'],['doc-2'])
        for k in ('rows','segments','topics'):pd.testing.assert_frame_equal(r[k],original[k])
        blob=save_project(r);restored=open_project(blob)
        self.assertEqual(restored['project_info']['version'],6);self.assertEqual(restored['yearly_analysis'],r['yearly_analysis'])
        self.assertEqual(open_project(save_project(original))['project_info']['version'],5)
        topics,parts,vectors=tables(restored);self.assertEqual(len(parts),2);self.assertEqual(len(vectors),2)
        self.assertEqual(parts['topic_id'].tolist(),[0,-1])
        archive=create_export(restored,'yearly')['data']
        with zipfile.ZipFile(io.BytesIO(archive)) as z:self.assertEqual(set(z.namelist()),{'yearly_topics.csv','yearly_segments.csv','yearly_vectors.csv','yearly_run.json'})
        broken=copy.deepcopy(restored);broken['yearly_analysis']['results'][0]['assignments'][0]['segment_id']='missing'
        with self.assertRaises(ValueError):save_project(broken)
        broken=copy.deepcopy(restored);broken['yearly_analysis']['results'][0]['topic_vectors'][0]['values']=[1.,float('nan')]
        with self.assertRaises(ValueError):validate(broken,broken['yearly_analysis'])

    def test_same_number_is_scoped_to_year(self):
        r=sample()
        def same(texts,encoder,**kwargs):
            return fake_fit([texts[0].replace('第1篇','第0篇')],encoder,**kwargs)
        r['yearly_analysis']=run(r,'year',[2022,2023],None,fit=same)
        topics,parts,_=tables(r)
        self.assertEqual(topics.topic_id.tolist(),[0,0]);self.assertEqual(parts.year.tolist(),[2022,2023])
        self.assertEqual(len(set(zip(parts.year,parts.topic_id))),2)

    def test_actual_model_guard_continues_other_years(self):
        r=sample()
        with patch('socket.socket.connect',side_effect=AssertionError('offline')):
            r['yearly_analysis']=run(r,'year',[2022,2023],None,fit=fit_topics)
        self.assertTrue(all(x['status']=='failed' and '至少需要 10' in x['error'] for x in r['yearly_analysis']['results']))

    def test_app_restore_and_run(self):
        r=sample();raw=save_project(r)
        with patch('socket.socket.connect',side_effect=AssertionError('offline')):
            app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=100).run()
            app.radio(key='work_mode').set_value('打开已有项目').run()
            app.file_uploader(key='project_upload').set_value(('example.zip',raw,'application/zip')).run(timeout=100)
            self.assertFalse(app.exception)
            scope='restored_'+app.session_state['opened_project']['digest'];prefix='yearly_'+scope
            app.selectbox(key=prefix+'_field').select('year').run(timeout=100)
            self.assertTrue(any('缺失或无法明确解析' in x.value for x in app.info))
            from threading import Lock
            with patch('studio.yearly.fit_topics',side_effect=fake_fit),patch('studio.modeling.load_encoder',return_value=object()):
                app.button(key=prefix+'_run').click().run(timeout=100)
            self.assertFalse(app.exception);self.assertIn('yearly_analysis',app.session_state['opened_project']['result'])
            app.selectbox(key='export_kind_restored').select('project').run(timeout=100)
            app.button(key='export_prepare_restored').click().run(timeout=100)
            saved=app.session_state['prepared_export_restored']['data'];self.assertEqual(open_project(saved)['project_info']['version'],6)
            app.file_uploader(key='project_upload').set_value(('new.zip',saved,'application/zip')).run(timeout=100)
            self.assertFalse(app.exception);self.assertTrue(any('当前按年结果' in x.value for x in app.caption))
