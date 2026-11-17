import copy
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from streamlit.testing.v1 import AppTest

from studio.corpus import prepare, read_csv
from studio.hierarchy import (DISTANCE, LINKAGE, calculate_hierarchy, ensure_hierarchy,
                              hierarchy_table, render_hierarchy, update_hierarchy, validate_hierarchy)
from studio.modeling import load_encoder
from studio.project import open_project, save_project
from studio.runtime import ROOT, configure
from studio.similarity import VECTOR_TYPE, default_similarity, similarity_data
from studio.topic_names import rename
from studio.visualization import find_chinese_font
from studio.workflow import analyze, csv_bytes

configure()


def sample():
    ids=[-1,10,20,30,40,50]
    result={'topics':pd.DataFrame({'Topic':ids,'Name':[f'{i}_原标签' for i in ids],'Keywords':['词']*len(ids)}),
            'segments':pd.DataFrame({'topic':ids}),'metadata':{'generated_fields':{'topic_id':'topic'}},'visualization':{}}
    vectors=[{'topic_id':-1,'values':[-1.,0.]},{'topic_id':10,'values':[1.,0.]},
             {'topic_id':20,'values':[1.,0.]},{'topic_id':30,'values':[0.,1.]},
             {'topic_id':40,'values':[0.,0.]}]
    result['visualization']['topic_similarity']=default_similarity(vectors,2)
    return result


class HierarchyUnitTests(unittest.TestCase):
    def test_distances_mapping_exclusions_and_rename(self):
        result=sample();state=update_hierarchy(result,[50,30,10,40,20])
        tree=state['result']; self.assertEqual(tree['leaf_topics'],[10,20,30])
        self.assertEqual([x['topic_id'] for x in tree['excluded']],[40,50])
        np.testing.assert_allclose(tree['linkage_matrix'],[[0,1,0,2],[2,3,1,3]])
        self.assertEqual(result['metadata']['topic_hierarchy']['distance'],DISTANCE)
        validate_hierarchy(result,state)
        before=copy.deepcopy(tree);rename(result,10,'中文新名称')
        self.assertEqual(calculate_hierarchy(result,[10,20,30,40,50]),before)
        _,labels,_=similarity_data(result,[10,20,30]);table=hierarchy_table(state,labels)
        self.assertEqual(table.iloc[0]['topic_id'],10);self.assertEqual(table.iloc[0]['display_name'],'中文新名称')
        self.assertEqual((table.iloc[3]['left_node'],table.iloc[3]['right_node']),(0,1))
        self.assertEqual(table.iloc[-1]['contained_topic_ids'],'10,20,30')
        self.assertEqual(table.iloc[-1]['topic_count'],3)
        csv=pd.read_csv(io.BytesIO(csv_bytes(table)),keep_default_na=False)
        self.assertEqual(csv.iloc[-1]['distance_method'],DISTANCE)
        self.assertEqual(csv.iloc[-1]['linkage_method'],LINKAGE)

    def test_outlier_insufficient_invalid_and_tampering(self):
        result=sample()
        self.assertNotIn(-1,ensure_hierarchy(result)['settings']['selected_topics'])
        with self.assertRaisesRegex(ValueError,'离群主题未启用'):update_hierarchy(result,[-1,10])
        outlier=update_hierarchy(result,[-1,10],True)
        self.assertEqual(outlier['result']['leaf_topics'],[-1,10])
        self.assertEqual(outlier['result']['linkage_matrix'][0][2],2)
        one=update_hierarchy(result,[10,40,50]);self.assertEqual(one['result']['leaf_topics'],[10]);self.assertEqual(one['result']['linkage_matrix'],[])
        none=update_hierarchy(result,[40,50]);self.assertEqual(none['result']['leaf_topics'],[])
        state=update_hierarchy(result,[10,20,30]);invalid=copy.deepcopy(state)
        invalid['result']['linkage_matrix'][0][2]=.5
        with self.assertRaisesRegex(ValueError,'距离'):validate_hierarchy(result,invalid)
        old=sample();old['visualization'].pop('topic_similarity');self.assertIsNone(ensure_hierarchy(old))

    def test_png_svg_long_labels_and_precision(self):
        font,_=find_chinese_font()
        if not font:self.skipTest('本机没有中文字体')
        result=sample();rename(result,10,'中文较长主题名称用于检查显示完整'*3)
        state=update_hierarchy(result,[10,20,30]);_,labels,_=similarity_data(result,[10,20,30])
        from scipy.cluster.hierarchy import dendrogram as real_dendrogram
        observed=[]
        def capture(matrix,*args,**kwargs):
            observed.append(np.asarray(matrix));return real_dendrogram(matrix,*args,**kwargs)
        with patch('studio.hierarchy.dendrogram',side_effect=capture):
            png,missing=render_hierarchy(state,labels,12,10,150,font)
        np.testing.assert_array_equal(observed[0],np.asarray(state['result']['linkage_matrix']))
        self.assertFalse(missing);im=Image.open(io.BytesIO(png));self.assertEqual(im.size,(1800,1500))
        self.assertAlmostEqual(im.info['dpi'][0],150,delta=.05)
        svg,_=render_hierarchy(state,labels,12,10,150,font,'svg');self.assertIn(b'<svg',svg)
        high,_=render_hierarchy(state,{10:'短名称',20:'中文主题',30:'第三主题'},8,5,600,font)
        high_img=Image.open(io.BytesIO(high));self.assertEqual(high_img.size,(4800,3000))
        self.assertAlmostEqual(high_img.info['dpi'][0],600,delta=.05)
        output=ROOT/'outputs/hierarchy_validation';output.mkdir(exist_ok=True)
        (output/'long_labels.png').write_bytes(png);(output/'long_labels.svg').write_bytes(svg)


class HierarchyOfflineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.offline=patch('socket.socket.connect',side_effect=AssertionError('network forbidden'));cls.offline.start()
        encoder=load_encoder('cpu');corpus=read_csv((ROOT/'data/examples/segmentation_zh.csv').read_bytes())
        cls.result=analyze(prepare(corpus,'正文','deduplicate'),encoder,split_mode='auto')
        cls.font,_=find_chinese_font()

    @classmethod
    def tearDownClass(cls):
        cls.offline.stop();(ROOT/'outputs/hierarchy_validation').mkdir(exist_ok=True)
        (ROOT/'outputs/hierarchy_validation/run.json').write_text(json.dumps({'offline_socket_blocked':True,
            'synthetic':True,'documents':len(cls.result['rows']),'segments':len(cls.result['segments']),
            'topics':len(cls.result['topics']),'method':LINKAGE,'distance':DISTANCE},ensure_ascii=False,indent=2)+'\n')

    def test_actual_model_distances_save_reopen_export(self):
        result=copy.deepcopy(self.result);state=result['visualization']['topic_hierarchy']
        ids=state['result']['leaf_topics'];matrix,labels,excluded=similarity_data(result,ids)
        self.assertFalse(excluded);self.assertGreaterEqual(len(ids),2)
        distance=np.clip(1-matrix.to_numpy(),0,2);distance=(distance+distance.T)/2;np.fill_diagonal(distance,0)
        expected=linkage(squareform(distance),method='average')
        np.testing.assert_allclose(state['result']['linkage_matrix'],expected,atol=1e-12)
        self.assertEqual(state['vector_type'],VECTOR_TYPE)
        self.assertEqual(result['metadata']['topic_hierarchy']['leaf_topics'],ids)
        vectors_before=copy.deepcopy(result['visualization']['topic_similarity']['vectors'])
        tree_before=copy.deepcopy(state['result']);rename(result,ids[0],'佛教与政策研究主题')
        self.assertEqual(result['visualization']['topic_similarity']['vectors'],vectors_before)
        self.assertEqual(result['visualization']['topic_hierarchy']['result'],tree_before)
        restored=open_project(save_project(result))
        self.assertEqual(restored['visualization']['topic_hierarchy'],state)
        self.assertEqual(restored['metadata']['topic_hierarchy'],result['metadata']['topic_hierarchy'])
        _,labels,_=similarity_data(restored,ids);table=hierarchy_table(state,labels)
        self.assertEqual(table.iloc[0]['display_name'],'佛教与政策研究主题')
        saved=pd.read_csv(io.BytesIO(csv_bytes(table)),keep_default_na=False,float_precision='round_trip')
        np.testing.assert_allclose(saved[saved['node_type']=='merge']['merge_distance'].to_numpy(dtype=float),expected[:,2])
        subset=update_hierarchy(restored,ids[:3]);self.assertEqual(subset['result']['leaf_topics'],ids[:3])
        self.assertEqual(len(subset['result']['linkage_matrix']),2)
        self.assertEqual(open_project(save_project(restored))['visualization']['topic_hierarchy'],subset)
        output=ROOT/'outputs/hierarchy_validation';output.mkdir(exist_ok=True)
        (output/'actual.hpts.zip').write_bytes(save_project(result))
        (output/'actual.csv').write_bytes(csv_bytes(table))
        if self.font:
            png,_=render_hierarchy(state,labels,12,10,150,self.font)
            svg,_=render_hierarchy(state,labels,12,10,150,self.font,'svg')
            (output/'actual.png').write_bytes(png);(output/'actual.svg').write_bytes(svg)

    def test_restored_legacy_and_page_selection(self):
        old=copy.deepcopy(self.result);old['visualization'].pop('topic_similarity');old['visualization'].pop('topic_hierarchy')
        old['metadata'].pop('topic_hierarchy')
        self.assertIsNone(ensure_hierarchy(open_project(save_project(old))))
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
        app.radio(key='work_mode').set_value('打开已有项目').run()
        app.file_uploader(key='project_upload').set_value(('old.zip',save_project(old),'application/zip')).run(timeout=90)
        self.assertFalse(app.exception);self.assertTrue(any('未保存主题向量' in x.value for x in app.info))
        app.file_uploader(key='project_upload').set_value(('new.zip',save_project(self.result),'application/zip')).run(timeout=90)
        self.assertFalse(app.exception)
        keys=[str(x.key) for x in app.get('download_button')]
        for prefix in ('hier_csv_','hier_png_','hier_svg_'):
            self.assertTrue(any(k.startswith(prefix) for k in keys),prefix)
        scope=next(k for k in keys if k.startswith('hier_csv_')).replace('hier_csv_','')
        current=app.multiselect(key='hier_topics_'+scope).value
        app.multiselect(key='hier_topics_'+scope).set_value(current[:3]).run(timeout=90)
        cached=app.session_state['opened_project']['result']
        self.assertEqual(cached['visualization']['topic_hierarchy']['result']['leaf_topics'],sorted(current[:3]))
        self.assertEqual(cached['metadata']['topic_hierarchy']['settings']['selected_topics'],sorted(current[:3]))
        self.assertEqual(open_project(save_project(cached))['visualization']['topic_hierarchy'],cached['visualization']['topic_hierarchy'])
        app.multiselect(key='hier_topics_'+scope).set_value(current[:1]).run(timeout=90)
        self.assertTrue(any('至少需要 2 个' in x.value for x in app.info))

    def test_old_v3_with_vectors_can_generate_and_resave(self):
        old=copy.deepcopy(self.result);old['visualization'].pop('topic_hierarchy');old['metadata'].pop('topic_hierarchy')
        reopened=open_project(save_project(old))
        vectors=copy.deepcopy(reopened['visualization']['topic_similarity']['vectors'])
        state=ensure_hierarchy(reopened)
        self.assertGreaterEqual(len(state['result']['leaf_topics']),2)
        self.assertEqual(reopened['visualization']['topic_similarity']['vectors'],vectors)
        self.assertEqual(open_project(save_project(reopened))['visualization']['topic_hierarchy'],state)

    def test_outlier_page_selection(self):
        result=copy.deepcopy(self.result);f=result['metadata']['generated_fields'];segments=result['segments'];topics=result['topics']
        old=int(segments.iloc[0][f['topic_id']]);segments.loc[segments.index[0],f['topic_id']]=-1
        topics.loc[topics['Topic'].eq(old),'Count']-=1
        row=topics.loc[topics['Topic'].eq(old)].iloc[[0]].copy();row.loc[:,'Topic']=-1;row.loc[:,'Count']=1;row.loc[:,'Name']='-1_离群'
        result['topics']=pd.concat([row,topics],ignore_index=True)
        sim=result['visualization']['topic_similarity'];source=next(v for v in sim['vectors'] if v['topic_id']==old)
        sim['vectors'].append({'topic_id':-1,'values':source['values'][:]});result['metadata']['outlier_count']=1
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
        app.radio(key='work_mode').set_value('打开已有项目').run()
        app.file_uploader(key='project_upload').set_value(('outlier.zip',save_project(result),'application/zip')).run(timeout=90)
        self.assertFalse(app.exception)
        scope=next(str(e.key) for e in app.get('download_button') if str(e.key).startswith('hier_csv_')).replace('hier_csv_','')
        self.assertNotIn(-1,app.multiselect(key='hier_topics_'+scope).value)
        app.checkbox(key='hier_outliers_'+scope).check().run(timeout=90)
        self.assertTrue(any('离群' in x for x in app.multiselect(key='hier_topics_'+scope).options))
        current=app.multiselect(key='hier_topics_'+scope).value
        app.multiselect(key='hier_topics_'+scope).set_value([-1]+current[:2]).run(timeout=90)
        cached=app.session_state['opened_project']['result']
        self.assertTrue(cached['visualization']['topic_hierarchy']['settings']['show_outliers'])
        self.assertIn(-1,cached['visualization']['topic_hierarchy']['result']['leaf_topics'])
        app.checkbox(key='hier_outliers_'+scope).uncheck().run(timeout=90)
        self.assertNotIn(-1,app.session_state['opened_project']['result']['visualization']['topic_hierarchy']['result']['leaf_topics'])


if __name__=='__main__':unittest.main(verbosity=2)
