import copy
import io
import json
import unittest
import zipfile
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from studio.cross_year import calculate, defaults, evidence, graph, source_digest
from studio.deferred_exports import create_export
from studio.project import open_project, save_project
from studio.runtime import ROOT
from studio.yearly import run
from test_yearly import sample


def yearly_sample():
    r=sample();years=['2022','2022','2023','2023']
    r['rows']['year']=years;r['segments']['year']=years
    def fake(texts,encoder,**kwargs):
        return {'topics':[0,1], 'strengths':np.array([0.5,0.5]),
          'topic_info':pd.DataFrame([{'Topic':0,'Count':1,'Name':'同名','Keywords':'政策'}, {'Topic':1,'Count':1,'Name':'同名','Keywords':'教育'}]),
          'topic_vectors':[{'topic_id':0,'values':[1.,0.]},{'topic_id':1,'values':[0.,1.]}],
          'keyword_weights':[],'parameters':{'random_state':42},'timings_seconds':{},'topic_vector_dimension':2,'model_token_limit':128,'max_input_tokens':20}
    r['yearly_analysis']=run(r,'year',[2022,2023],None,fit=fake)
    return r


class CrossYearTests(unittest.TestCase):
    def test_known_vectors_threshold_mapping_and_evidence(self):
        r=yearly_sample();settings=defaults(r);settings['threshold']=0.5
        nodes,pairs,excluded,warnings=calculate(r,settings)
        self.assertFalse(excluded);self.assertFalse(warnings);self.assertEqual(len(pairs),4)
        p=pairs.set_index(['source_year','source_topic_id','target_year','target_topic_id'])
        self.assertEqual(p.loc[(2022,0,2023,0),'similarity'],1.)
        self.assertEqual(p.loc[(2022,0,2023,1),'similarity'],0.)
        self.assertEqual(len(p[p.in_graph]),2)
        self.assertEqual(p.loc[(2022,0,2023,0),'source_documents'],1)
        self.assertEqual(len(evidence(r,2022,0)),1)
        fig=graph(nodes,pairs,settings);self.assertEqual(list(fig.data[0].link.value),[1.,1.])
        self.assertEqual(sum(bool(x) for x in pairs.in_graph),2)
        settings['max_edges']=1;self.assertEqual(int(calculate(r,settings)[1].in_graph.sum()),1)

    def test_missing_zero_outlier_and_unconnected(self):
        r=yearly_sample();run2=r['yearly_analysis']['results'][1]
        run2['topic_vectors'][0]['values']=None;run2['topic_vectors'][1]['values']=[0.,0.]
        settings=defaults(r);nodes,pairs,excluded,warnings=calculate(r,settings)
        self.assertTrue(pairs.empty);self.assertEqual(len(excluded),2);self.assertIsNone(graph(nodes,pairs,settings))
        self.assertTrue(any('零向量' in x['reason'] for x in excluded))
        # Turn one real topic into an outlier and verify it is never offered.
        run2['topics'][0]['topic_id']=-1;run2['assignments'][0]['topic_id']=-1;run2['topic_vectors'][0]['topic_id']=-1
        run2['outliers']=1;run2['non_outlier_topics']=1
        self.assertNotIn([2023,-1],defaults(r)['topics'])

    def test_gap_mismatch_stale_and_roundtrip(self):
        r=yearly_sample();settings=defaults(r);r['visualization']['cross_year_links']=settings
        b=save_project(r);restored=open_project(b);self.assertEqual(restored['visualization']['cross_year_links'],settings)
        z=create_export(restored,'cross_year')['data']
        with zipfile.ZipFile(io.BytesIO(z)) as archive:
            self.assertIn('graph.html',archive.namelist());self.assertIn('graph_edges.csv',archive.namelist())
            self.assertIn(b'plotly.js',archive.read('graph.html').lower())
        changed=copy.deepcopy(restored);changed['yearly_analysis']['results'][0]['topic_vectors'][0]['values']=[0.5,0.5]
        with self.assertRaisesRegex(ValueError,'过期'):save_project(changed)
        with self.assertRaisesRegex(ValueError,'过期'):calculate(changed,settings)
        changed=copy.deepcopy(r);changed['yearly_analysis']['results'][1]['model_revision']='other'
        _,pairs,_,warnings=calculate(changed,defaults(changed));self.assertTrue(pairs.empty);self.assertTrue(warnings)
        changed=copy.deepcopy(r);second=changed['yearly_analysis']['results'][1];second['vector_dimension']=3
        for item in second['topic_vectors']:item['values'].append(0.)
        _,pairs,_,warnings=calculate(changed,defaults(changed));self.assertTrue(pairs.empty);self.assertTrue(any('维度' in x for x in warnings))
        # Select nonadjacent years by converting actual document year and its associated run.
        gap=copy.deepcopy(r);gap['rows'].loc[2:3,'year']='2025';gap['segments'].loc[2:3,'year']='2025'
        y=gap['yearly_analysis'];y['selected_years']=[2022,2025];y['results'][1]['year']=2025
        gap['visualization'].pop('cross_year_links',None)
        _,pairs,_,warnings=calculate(gap,defaults(gap));self.assertTrue(pairs.empty);self.assertTrue(any('不相邻' in x for x in warnings))

    def test_page_restored(self):
        r=yearly_sample();app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=100).run()
        app.radio(key='work_mode').set_value('打开已有项目').run()
        app.file_uploader(key='project_upload').set_value(('project.zip',save_project(r),'application/zip')).run(timeout=100)
        self.assertFalse(app.exception, list(app.exception))
        self.assertTrue(any('相邻年份主题相似关联桑基图' in x.value for x in app.subheader))
        saved=app.session_state['opened_project']['result'];self.assertIn('cross_year_links',saved['visualization'])
        self.assertEqual(open_project(save_project(saved))['visualization']['cross_year_links'],saved['visualization']['cross_year_links'])
        previous=saved['visualization']['cross_year_links']['source_sha256']
        saved['yearly_analysis']['results'][0]['topic_vectors'][0]['values']=[0.5,0.5]
        app.run(timeout=100)
        self.assertFalse(app.exception, list(app.exception))
        self.assertNotEqual(saved['visualization']['cross_year_links']['source_sha256'],previous)
        self.assertTrue(any('旧关联设置已失效' in x.value for x in app.warning))
