import copy
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from bertopic import BERTopic
from PIL import Image
from streamlit.testing.v1 import AppTest

from studio.corpus import prepare, read_csv
from studio.modeling import load_encoder
from studio.project import open_project, save_project
from studio.runtime import ROOT, configure
from studio.similarity import (METHOD, VECTOR_TYPE, default_similarity, export_matrix,
                               render_similarity, similarity_data, validate_similarity)
from studio.topic_names import rename
from studio.visualization import find_chinese_font
from studio.workflow import analyze, csv_bytes

configure()


def small_result():
    ids = [-1, 0, 1, 2, 3, 4]
    result = {'topics': pd.DataFrame({'Topic':ids, 'Name':[f'{i}_原标签' for i in ids], 'Keywords':['词']*len(ids)}),
              'segments': pd.DataFrame({'topic': ids}), 'metadata':{'generated_fields':{'topic_id':'topic'}},
              'visualization':{}}
    vectors = [{'topic_id':-1,'values':[1.,1.]}, {'topic_id':0,'values':[1.,0.]},
               {'topic_id':1,'values':[1.,0.]}, {'topic_id':2,'values':[0.,1.]},
               {'topic_id':3,'values':[0.,0.]}]  # 4 intentionally missing
    result['visualization']['topic_similarity'] = default_similarity(vectors, 2)
    return result


class SimilarityUnitTests(unittest.TestCase):
    def test_known_vectors_missing_zero_rename(self):
        result = small_result()
        matrix, labels, excluded = similarity_data(result, [-1,0,1,2,3,4])
        self.assertEqual(matrix.loc[0,1], 1.0); self.assertEqual(matrix.loc[0,2], 0.0)
        np.testing.assert_allclose(matrix.to_numpy(), matrix.to_numpy().T)
        self.assertEqual({r['topic_id'] for r in excluded}, {3,4})
        self.assertEqual(matrix.index.tolist(), [-1,0,1,2])
        original = matrix.copy(); rename(result, 0, '中文主题名称')
        after, labels, _ = similarity_data(result, [0,1,2])
        pd.testing.assert_frame_equal(after, original.loc[[0,1,2],[0,1,2]])
        data = export_matrix(after, labels)
        self.assertEqual(data.loc[0, 'display_name'], '中文主题名称')
        self.assertEqual(data.loc[0, 'vector_type'], VECTOR_TYPE)
        self.assertEqual(data.loc[0, 'method'], METHOD)
        csv = pd.read_csv(io.BytesIO(csv_bytes(data)), float_precision='round_trip')
        np.testing.assert_array_equal(csv.iloc[:, 4:].to_numpy(), after.to_numpy())
        none,_,issues = similarity_data(result, [3,4]); self.assertTrue(none.empty); self.assertEqual(len(issues),2)
        singleton,_,_ = similarity_data(result, [0]); self.assertEqual(singleton.shape, (1,1))

    def test_invalid_vector_metadata(self):
        result=small_result(); obj=result['visualization']['topic_similarity']
        for bad in ([float('nan'), 0], [1], [True, 1]):
            clone=copy.deepcopy(obj); clone['vectors'][0]['values']=bad
            with self.assertRaises(ValueError):validate_similarity(result,clone)
        clone=copy.deepcopy(obj); clone['method']='invented'
        with self.assertRaises(ValueError):validate_similarity(result,clone)
        old=copy.deepcopy(result);old['visualization'].pop('topic_similarity')
        self.assertIsNone(similarity_data(old,[0])[0])

    def test_heatmap_png_svg_and_long_label(self):
        font,_ = find_chinese_font()
        if not font:self.skipTest('本机无中文字体')
        result=small_result(); rename(result, 0, '中文长主题名称用于检验显示完整'*3)
        matrix,labels,_=similarity_data(result,[-1,0,1,2])
        from matplotlib.axes import Axes
        original_imshow=Axes.imshow; observed=[]
        def spy(axis, data, *args, **kwargs):
            observed.append(np.asarray(data)); return original_imshow(axis, data, *args, **kwargs)
        with patch.object(Axes,'imshow',spy):
            png, missing=render_similarity(matrix,labels,12,10,150,font,True)
        np.testing.assert_array_equal(observed[0],matrix.to_numpy())
        self.assertFalse(missing)
        im=Image.open(io.BytesIO(png));self.assertEqual(im.size,(1800,1500))
        self.assertAlmostEqual(im.info['dpi'][0],150,delta=.05)
        svg,_=render_similarity(matrix,labels,12,10,150,font,False,'svg')
        self.assertIn(b'<svg',svg);self.assertIn(b'</svg>',svg)
        high,_=render_similarity(matrix.loc[[0,1],[0,1]],labels,8,5,600,font,True)
        high_img=Image.open(io.BytesIO(high));self.assertEqual(high_img.size,(4800,3000))
        self.assertAlmostEqual(high_img.info['dpi'][0],600,delta=.05)
        output=ROOT/'outputs/similarity_validation';output.mkdir(exist_ok=True)
        (output/'long_labels.png').write_bytes(png);(output/'long_labels.svg').write_bytes(svg)


class SimilarityOfflineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.offline=patch('socket.socket.connect', side_effect=AssertionError('network forbidden'));cls.offline.start()
        encoder=load_encoder('cpu')
        corpus=read_csv((ROOT/'data/examples/segmentation_zh.csv').read_bytes())
        cls.capture={}
        real=BERTopic.fit_transform
        def spy(model, documents, *args, **kwargs):
            output=real(model,documents,*args,**kwargs)
            cls.capture['vectors']=model.topic_embeddings_.copy()
            cls.capture['topics']=list(output[0]);cls.capture['embeddings']=kwargs['embeddings'].copy()
            return output
        with patch.object(BERTopic,'fit_transform',spy):
            cls.result=analyze(prepare(corpus,'正文','deduplicate'),encoder,split_mode='auto')
        cls.font,_=find_chinese_font()

    @classmethod
    def tearDownClass(cls):
        cls.offline.stop()
        (ROOT/'outputs/similarity_validation').mkdir(exist_ok=True)
        (ROOT/'outputs/similarity_validation/run.json').write_text(json.dumps({
            'offline_socket_blocked':True,'synthetic':True,'documents':len(cls.result['rows']),
            'segments':len(cls.result['segments']),'topics':len(cls.result['topics']),
            'dimension':cls.result['visualization']['topic_similarity']['dimension'],
            'vector_source':VECTOR_TYPE,'method':METHOD},ensure_ascii=False,indent=2)+'\n')

    def test_actual_vectors_means_and_project_roundtrip(self):
        result=copy.deepcopy(self.result); sim=result['visualization']['topic_similarity']
        vectors=np.asarray([x['values'] for x in sim['vectors']]); ids=[x['topic_id'] for x in sim['vectors']]
        np.testing.assert_array_equal(vectors,self.capture['vectors'])
        self.assertEqual(ids, sorted(set(self.capture['topics'])))
        for i,topic in enumerate(ids):
            expected=self.capture['embeddings'][np.array(self.capture['topics'])==topic].mean(axis=0)
            np.testing.assert_allclose(vectors[i],expected,atol=1e-7)
        matrix,labels,excluded=similarity_data(result,ids); self.assertFalse(excluded)
        unit=vectors/np.linalg.norm(vectors,axis=1)[:,None]
        np.testing.assert_allclose(matrix.to_numpy(),unit@unit.T,atol=1e-7)
        sim['settings'].update({'show_outliers':True,'show_values':False,'selected_topics':ids[:2]})
        rename(result, ids[0], '研究主题：佛教与政策')
        reopened=open_project(save_project(result)); self.assertEqual(reopened['visualization'],result['visualization'])
        after,labels2,_=similarity_data(reopened,ids)
        pd.testing.assert_frame_equal(matrix,after);self.assertEqual(labels2[ids[0]],'研究主题：佛教与政策')
        export=export_matrix(after,labels2)
        restored=pd.read_csv(io.BytesIO(csv_bytes(export)),float_precision='round_trip')
        np.testing.assert_array_equal(restored.iloc[:,4:].to_numpy(), after.to_numpy())
        if self.font:
            png,_=render_similarity(after,labels2,12,10,150,self.font)
            svg,_=render_similarity(after,labels2,12,10,150,self.font,fmt='svg')
            output=ROOT/'outputs/similarity_validation';output.mkdir(exist_ok=True)
            (output/'actual.png').write_bytes(png);(output/'actual.svg').write_bytes(svg)
            (output/'actual.csv').write_bytes(csv_bytes(export))
            (output/'actual.hpts.zip').write_bytes(save_project(result))

    def test_legacy_project_and_page_flow(self):
        result=copy.deepcopy(self.result); result['visualization'].pop('topic_similarity')
        result['visualization'].pop('topic_hierarchy', None); result['metadata'].pop('topic_hierarchy', None)
        old=open_project(save_project(result)); self.assertIsNone(similarity_data(old,[0])[0])
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
        app.radio(key='work_mode').set_value('打开已有项目').run()
        app.file_uploader(key='project_upload').set_value(('old.zip',save_project(result),'application/zip')).run(timeout=90)
        self.assertFalse(app.exception)
        self.assertTrue(any('未保存计算主题相似度所需的数据' in x.value for x in app.info))
        app.file_uploader(key='project_upload').set_value(('new.zip',save_project(self.result),'application/zip')).run(timeout=90)
        self.assertFalse(app.exception)
        keys=[str(x.key) for x in app.get('download_button')]
        for prefix in ('sim_csv_', 'sim_png_', 'sim_svg_'):
            self.assertTrue(any(k.startswith(prefix) for k in keys),prefix)
        scope=next(k for k in keys if k.startswith('sim_csv_')).replace('sim_csv_','')
        app.checkbox(key='sim_values_'+scope).uncheck().run(timeout=90)
        options=app.multiselect(key='sim_topics_'+scope).options
        self.assertGreaterEqual(len(options),2)
        current=app.multiselect(key='sim_topics_'+scope).value
        app.multiselect(key='sim_topics_'+scope).set_value(current[:2]).run(timeout=90)
        self.assertFalse(app.exception)
        cached=app.session_state['opened_project']['result']['visualization']['topic_similarity']['settings']
        self.assertFalse(cached['show_values']);self.assertEqual(len(cached['selected_topics']),2)
        self.assertEqual(open_project(save_project(app.session_state['opened_project']['result']))['visualization']['topic_similarity']['settings'],cached)

    def test_outlier_toggle_in_restored_project(self):
        result=copy.deepcopy(self.result);f=result['metadata']['generated_fields'];segments=result['segments'];topics=result['topics']
        old=int(segments.iloc[0][f['topic_id']]);segments.loc[segments.index[0],f['topic_id']]=-1
        topics.loc[topics['Topic'].eq(old),'Count']-=1
        row=topics.loc[topics['Topic'].eq(old)].iloc[[0]].copy()
        row.loc[:,'Topic']=-1;row.loc[:,'Count']=1;row.loc[:,'Name']='-1_离群'
        result['topics']=pd.concat([row,topics],ignore_index=True)
        sim=result['visualization']['topic_similarity'];source=next(v for v in sim['vectors'] if v['topic_id']==old)
        sim['vectors'].append({'topic_id':-1,'values':source['values'][:]});sim['settings']['selected_topics'].insert(0,-1)
        result['metadata']['outlier_count']=1
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
        app.radio(key='work_mode').set_value('打开已有项目').run()
        app.file_uploader(key='project_upload').set_value(('outlier.zip',save_project(result),'application/zip')).run(timeout=90)
        self.assertFalse(app.exception)
        scope=next(str(e.key) for e in app.get('download_button') if str(e.key).startswith('sim_csv_')).replace('sim_csv_','')
        self.assertNotIn(-1,app.multiselect(key='sim_topics_'+scope).value)
        app.checkbox(key='sim_outliers_'+scope).check().run(timeout=90)
        self.assertTrue(any('离群' in option for option in app.multiselect(key='sim_topics_'+scope).options))
        current=app.multiselect(key='sim_topics_'+scope).value
        app.multiselect(key='sim_topics_'+scope).set_value(list(dict.fromkeys([-1]+current))).run(timeout=90)
        self.assertIn(-1,app.multiselect(key='sim_topics_'+scope).value)
        app.checkbox(key='sim_outliers_'+scope).uncheck().run(timeout=90)
        self.assertNotIn(-1,app.multiselect(key='sim_topics_'+scope).value)


if __name__=='__main__':unittest.main(verbosity=2)
