import copy
import io
import json
import unittest
import zipfile
import hashlib
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from bertopic import BERTopic
from streamlit.testing.v1 import AppTest
from studio.corpus import read_csv, prepare
from studio.modeling import load_encoder
from studio.project import save_project, open_project
from studio.runtime import ROOT, configure
from studio.topic_names import rename
from studio.visualization import (WEIGHT_SOURCE, default_visualization, distribution_data,
                                  keyword_data, find_chinese_font, missing_glyphs, render_chart)
from studio.workflow import analyze

configure()

class VisualizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.offline=patch('socket.socket.connect',side_effect=AssertionError('network forbidden'))
        cls.offline.start()
        encoder=load_encoder('cpu')
        corpus=read_csv((ROOT/'data/examples/segmentation_zh.csv').read_bytes())
        observed={}
        real=BERTopic.get_topic
        def capture(model, topic, *args, **kwargs):
            response=real(model,topic,*args,**kwargs)
            if isinstance(response,list) and all(isinstance(item,tuple) and len(item)==2 for item in response):
                observed[int(topic)]=[(w,float(v)) for w,v in response]
            return response
        with patch.object(BERTopic,'get_topic',capture):
            cls.result=analyze(prepare(corpus,'正文','deduplicate'),encoder,split_mode='auto')
        cls.observed=observed
        cls.font, cls.font_name=find_chinese_font()

    @classmethod
    def tearDownClass(cls):cls.offline.stop()

    def test_real_weights_and_counts(self):
        result=self.result
        f=result['metadata']['generated_fields']; dist=distribution_data(result)
        counts=result['segments'][f['topic_id']].value_counts().to_dict()
        self.assertEqual(dist.set_index('Topic')['Count'].to_dict(),counts)
        self.assertEqual(dist['Unit'].unique().tolist(),['文本片段'])
        weights=result['visualization']['keyword_weights']
        self.assertEqual(result['visualization']['weight_source'],WEIGHT_SOURCE)
        by_topic={}
        for row in weights:by_topic.setdefault(row['topic_id'],[]).append((row['keyword'],row['weight']))
        self.assertEqual(by_topic,self.observed)
        for topic in counts:
            shown=keyword_data(result,topic)
            self.assertEqual(list(zip(shown['keyword'],shown['weight'])),[(w,v) for w,v in self.observed[topic] if w.strip()])

    def test_rename_and_outlier_filter(self):
        result=copy.deepcopy(self.result); f=result['metadata']['generated_fields']; topic=int(result['topics']['Topic'].iloc[0])
        before=copy.deepcopy(result['visualization']['keyword_weights'])
        rename(result,topic,'中文主题 / Policy')
        dist=distribution_data(result)
        self.assertEqual(dist.set_index('Topic').loc[topic,'Display_Name'],'中文主题 / Policy')
        self.assertEqual(result['visualization']['keyword_weights'],before)
        # A synthetic -1 assignment exercises display logic independent of cluster outcome.
        old=int(result['segments'].iloc[0][f['topic_id']])
        result['segments'].loc[result['segments'].index[0],f['topic_id']]=-1
        outlier=result['topics'].iloc[[0]].copy();outlier.loc[:,'Topic']=-1;outlier.loc[:,'Name']='-1_离群';outlier.loc[:,'Keywords']='离群'
        result['topics']=pd.concat([result['topics'],outlier],ignore_index=True)
        self.assertIn(-1,distribution_data(result,True)['Topic'].tolist())
        self.assertNotIn(-1,distribution_data(result,False)['Topic'].tolist())
        self.assertEqual(distribution_data(result,True).set_index('Topic').loc[-1,'Count'],1)
        if self.font:
            png,missing=render_chart(distribution_data(result,True),'distribution',8,5,150,self.font)
            self.assertTrue(png.startswith(b'\x89PNG'))
            self.assertFalse(missing)

    def test_project_roundtrip_old_versions_and_settings(self):
        result=copy.deepcopy(self.result);visual=result['visualization']
        visual['settings'].update({'show_outliers':False,'width_inches':8.0,'height_inches':5.0,'dpi':600,'keyword_topic_id':int(result['topics']['Topic'].iloc[0])})
        topic=int(result['topics']['Topic'].iloc[0]);rename(result,topic,'中文名称')
        data=save_project(result); reopened=open_project(data)
        self.assertEqual(reopened['project_info']['version'],5)
        self.assertEqual(reopened['visualization'],visual)
        self.assertEqual(distribution_data(reopened,False).to_dict('records'),distribution_data(result,False).to_dict('records'))
        self.assertEqual(keyword_data(reopened,topic).to_dict('records'),keyword_data(result,topic).to_dict('records'))
        (ROOT/'outputs/chart_test_project.hpts.zip').write_bytes(data)
        # Simulate version 1 and 2 from the same saved data; legacy files have no real weights.
        from test_project import legacy_project
        for version in (1,2):
            with zipfile.ZipFile(io.BytesIO(legacy_project(data))) as archive:
                files={name:archive.read(name) for name in archive.namelist()}
            files.pop('visualization.json')
            if version == 1: files.pop('topic_edits.json')
            info=json.loads(files['project.json']);info['version']=version
            files['project.json']=json.dumps(info,ensure_ascii=False).encode()
            manifest=json.loads(files['manifest.json'])
            manifest['files']={name:{'size':len(value),'sha256':hashlib.sha256(value).hexdigest()}
                               for name,value in files.items() if name!='manifest.json'}
            files['manifest.json']=json.dumps(manifest).encode()
            buffer=io.BytesIO()
            with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as archive:
                for name,value in files.items():archive.writestr(name,value)
            legacy=open_project(buffer.getvalue())
            self.assertEqual(legacy['project_info']['version'],version)
            if version == 2: self.assertEqual(legacy['topic_edits'],result['topic_edits'])
            self.assertIsNone(keyword_data(legacy,topic))
            self.assertFalse(distribution_data(legacy).empty)
            self.assertIsNone(open_project(save_project(legacy))['visualization']['keyword_weights'])
        # Existing actual version-1 and version-2 projects also contain no weight sidecar.
        for old in [ROOT / 'data/private/analysis.hpts.zip',ROOT / 'data/private/edited.hpts.zip']:
            if Path(old).exists():
                legacy=open_project(Path(old).read_bytes())
                self.assertIsNone(keyword_data(legacy,int(legacy['topics']['Topic'].iloc[0])))
                self.assertFalse(distribution_data(legacy).empty)
                saved=open_project(save_project(legacy))
                self.assertEqual(saved['project_info']['version'],5)
                self.assertIsNone(saved['visualization']['keyword_weights'])

    def test_png_svg_size_dpi_font_and_borders(self):
        self.assertTrue(self.font, '本机缺少中文字体')
        self.assertFalse(missing_glyphs(self.font,['主题分布文本片段关键词权重','环境政策','教育研究']))
        result=copy.deepcopy(self.result);topic=int(result['topics']['Topic'].iloc[0]);rename(result,topic,'中文研究主题')
        charts=[('distribution',distribution_data(result),None,''),('keywords',keyword_data(result,topic),topic,'中文研究主题')]
        output=ROOT/'outputs/chart_validation';output.mkdir(parents=True,exist_ok=True)
        for kind,frame,tid,label in charts:
            data,missing=render_chart(frame,kind,8,5,600,self.font,topic_id=tid,topic_label=label)
            self.assertFalse(missing)
            image=Image.open(io.BytesIO(data));self.assertEqual(image.size,(4800,3000))
            self.assertAlmostEqual(image.info['dpi'][0],600,delta=.1)
            rgb=np.asarray(image.convert('RGB'))
            self.assertTrue((rgb[0,:,:] > 245).all() and (rgb[-1,:,:] > 245).all())
            self.assertTrue((rgb[:,0,:] > 245).all() and (rgb[:,-1,:] > 245).all())
            self.assertGreater(np.count_nonzero(np.any(rgb<100,axis=2)),100)
            (output/f'{kind}_600dpi.png').write_bytes(data)
            svg,_=render_chart(frame,kind,8,5,150,self.font,topic_id=tid,topic_label=label,fmt='svg')
            self.assertIn(b'<svg',svg); (output/f'{kind}.svg').write_bytes(svg)
        self.assertEqual(render_chart.__name__,'render_chart')
        with self.assertRaisesRegex(ValueError,'字体'):
            render_chart(charts[0][1],'distribution',8,5,150,'/nonexistent/font.ttf')
        with self.assertRaisesRegex(ValueError,'像素'):
            render_chart(charts[0][1],'distribution',16,12,600,self.font)

    def test_old_project_page_and_new_project_page(self):
        legacy=ROOT / 'data/private/analysis.hpts.zip'
        if Path(legacy).exists():
            app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
            app.radio(key='work_mode').set_value('打开已有项目').run()
            app.file_uploader(key='project_upload').set_value(('old.zip',Path(legacy).read_bytes(),'application/zip')).run(timeout=90)
            self.assertFalse(app.exception)
            self.assertTrue(any('未保存真实关键词权重' in x.value for x in app.warning))
            self.assertTrue(any(str(x.key).startswith('distribution_png_') for x in app.get('download_button')))
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
        app.radio(key='work_mode').set_value('打开已有项目').run()
        app.file_uploader(key='project_upload').set_value(('new.zip',save_project(self.result),'application/zip')).run(timeout=90)
        self.assertFalse(app.exception)
        keys=[str(x.key) for x in app.get('download_button')]
        for prefix in ('distribution_png_','keywords_png_','distribution_csv_','keywords_csv_','distribution_svg_','keywords_svg_'):
            self.assertTrue(any(k.startswith(prefix) for k in keys),prefix)
        # save_project includes a fresh timestamp; use the actual scope keys, not a regenerated ZIP.
        scope_key=next(k for k in keys if k.startswith('distribution_png_')).replace('distribution_png_','')
        app.checkbox(key='chart_outliers_'+scope_key).uncheck().run(timeout=90)
        app.selectbox(key='chart_dpi_'+scope_key).select(300).run(timeout=90)
        cached=app.session_state['opened_project']['result']
        self.assertFalse(cached['visualization']['settings']['show_outliers'])
        self.assertEqual(cached['visualization']['settings']['dpi'],300)
        self.assertEqual(open_project(save_project(cached))['visualization']['settings']['dpi'],300)

if __name__=='__main__':unittest.main(verbosity=2)
