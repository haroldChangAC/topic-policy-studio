import copy
import io
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
from streamlit.testing.v1 import AppTest

from studio.hierarchy import ensure_hierarchy, render_hierarchy
from studio.plot_style import default_appearance, resolve_appearance, topic_label
from studio.project import open_project, save_project
from studio.runtime import ROOT, configure
from studio.similarity import default_similarity, render_similarity, similarity_data
from studio.visualization import (WEIGHT_SOURCE, default_visualization, distribution_data,
                                  find_chinese_font, keyword_data, render_chart)

configure()


def sample_result():
    f={k:'studio_'+k for k in ('record_id','document_id','original_text','status','segment_id','segment_order','start_char','end_char','segment_text','topic_id')}
    rows=[];segments=[]
    for i in range(4):
        text=f'第{i}篇中文原文，研究主题与政策。';did=f'doc-{i}'
        rows.append({'text':text,f['record_id']:did,f['document_id']:did,f['original_text']:text,f['status']:'modeled'})
        segments.append({**rows[-1],f['segment_id']:f'segment-{i}',f['segment_order']:1,f['start_char']:0,
                         f['end_char']:len(text),f['segment_text']:text,f['topic_id']:i})
    topics=pd.DataFrame({'Topic':[0,1,2,3],'Count':[1,1,1,1],
                         'Name':['0_中文政策研究的长标签用于检查排版','1_佛教文化','2_教育研究','3_地区发展'],
                         'Keywords':['政策、研究','佛教、文化','教育、研究','地区、发展']})
    weights=[{'topic_id':i,'rank':j+1,'keyword':word,'weight':float(0.4/(j+1))}
             for i,words in enumerate([['政策','研究'],['佛教','文化'],['教育','研究'],['地区','发展']]) for j,word in enumerate(words)]
    visual=default_visualization(weights)
    visual['topic_similarity']=default_similarity([{'topic_id':0,'values':[1.,0.,0.]},
        {'topic_id':1,'values':[0.,1.,0.]},{'topic_id':2,'values':[0.,0.,1.]},{'topic_id':3,'values':[1.,1.,0.]}],3)
    result={'rows':pd.DataFrame(rows),'segments':pd.DataFrame(segments),'topics':topics,
            'metadata':{'generated_fields':f,'original_columns':['text'],'counts':{'input':4,'segments':4},
                        'schema_version':2,'run_id':'paper-test','model_id':'test-model','model_revision':'test',
                        'versions':{'bertopic':'0.17.4'}},'visualization':visual}
    ensure_hierarchy(result)
    return result


class PaperStyleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.font,cls.family=find_chinese_font()

    def test_two_styles_four_charts_and_data_unchanged(self):
        self.assertTrue(self.font)
        result=sample_result();before=copy.deepcopy(result)
        dist=distribution_data(result);words=keyword_data(result,0)
        matrix,labels,_=similarity_data(result,[0,1,2,3]);state=result['visualization']['topic_hierarchy']
        for style in ('color','mono'):
            cfg=default_appearance();cfg.update({'style':style,'show_keyword_values':True})
            tasks=[lambda fmt:render_chart(dist,'distribution',10,7,150,self.font,fmt=fmt,appearance=cfg),
                   lambda fmt:render_chart(words,'keywords',10,7,150,self.font,topic_id=0,topic_label=labels[0],fmt=fmt,appearance=cfg),
                   lambda fmt:render_similarity(matrix,labels,10,7,150,self.font,True,fmt,appearance=cfg),
                   lambda fmt:render_hierarchy(state,labels,10,7,150,self.font,fmt,appearance=cfg)]
            for render in tasks:
                png,missing=render('png');self.assertFalse(missing)
                img=Image.open(io.BytesIO(png));self.assertEqual(img.size,(1500,1050));self.assertAlmostEqual(img.info['dpi'][0],150,delta=.1)
                rgb=np.asarray(img.convert('RGB'))
                self.assertTrue((rgb[0,:,:]>245).all() and (rgb[-1,:,:]>245).all())
                self.assertTrue((rgb[:,0,:]>245).all() and (rgb[:,-1,:]>245).all())
                if style=='mono':
                    self.assertTrue(np.array_equal(rgb[:,:,0],rgb[:,:,1]) and np.array_equal(rgb[:,:,1],rgb[:,:,2]))
                svg,_=render('svg');self.assertIn(b'<svg',svg);self.assertIn(b'</svg>',svg)
        pd.testing.assert_frame_equal(result['rows'],before['rows']);pd.testing.assert_frame_equal(result['segments'],before['segments'])
        pd.testing.assert_frame_equal(result['topics'],before['topics']);self.assertEqual(result['visualization'],before['visualization'])

    def test_title_value_labels_and_integer_axis(self):
        result=sample_result();dist=distribution_data(result);words=keyword_data(result,0)
        cfg=default_appearance();cfg['titles']['distribution']='论文专用标题';cfg['show_distribution_values']=True
        from matplotlib.ticker import MaxNLocator
        from studio.plot_style import check_y_label_spacing
        captured=[]
        def record(axis):
            captured.append((axis.xaxis.get_major_locator(),axis.get_xticks().copy()))
            return check_y_label_spacing(axis)
        with patch('studio.visualization.check_y_label_spacing',side_effect=record):
            svg,_=render_chart(dist,'distribution',10,7,150,self.font,fmt='svg',appearance=cfg)
        self.assertIsInstance(captured[0][0],MaxNLocator)
        self.assertTrue(all(float(tick).is_integer() for tick in captured[0][1]))
        self.assertIn('论文专用标题',svg.decode());self.assertIn('文本片段数',svg.decode())
        self.assertIn('<!-- 1 -->',svg.decode())
        cfg['show_title']=False
        svg,_=render_chart(dist,'distribution',10,7,150,self.font,fmt='svg',appearance=cfg)
        self.assertNotIn('论文专用标题',svg.decode());self.assertIn('文本片段数',svg.decode())
        cfg['show_keyword_values']=True
        svg,_=render_chart(words,'keywords',10,7,150,self.font,topic_id=0,topic_label='0_长名称',fmt='svg',appearance=cfg)
        self.assertIn('c-TF-IDF',svg.decode());self.assertIn('<!-- 0.4 -->',svg.decode())
        self.assertEqual(topic_label(0,'0_环境政策'),'0 · 环境政策')
        self.assertEqual(topic_label(-1,'-1_离群'),'-1（离群） · 离群')
        with self.assertRaises(ValueError):resolve_appearance({'font_size':99})

    def test_project_roundtrip_and_page_controls(self):
        result=sample_result();base=copy.deepcopy(result)
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
        app.radio(key='work_mode').set_value('打开已有项目').run()
        app.file_uploader(key='project_upload').set_value(('paper.zip',save_project(result),'application/zip')).run(timeout=90)
        self.assertFalse(app.exception)
        keys=[str(e.key) for e in app.get('download_button')]
        scope=next(k for k in keys if k.startswith('distribution_png_')).replace('distribution_png_','')
        app.selectbox(key='paper_style_'+scope).select('论文黑白').run(timeout=90)
        app.number_input(key='paper_font_'+scope).set_value(12).run(timeout=90)
        app.checkbox(key='paper_title_'+scope).uncheck().run(timeout=90)
        app.checkbox(key='paper_keyword_values_'+scope).check().run(timeout=90)
        app.text_input(key='paper_title_distribution_'+scope).set_value('测试论文标题').run(timeout=90)
        self.assertFalse(app.exception)
        cached=app.session_state['opened_project']['result'];cfg=cached['visualization']['settings']['appearance']
        self.assertEqual((cfg['style'],cfg['font_size'],cfg['show_title'],cfg['show_keyword_values']),('mono',12,False,True))
        reopened=open_project(save_project(cached));self.assertEqual(reopened['visualization']['settings']['appearance'],cfg)
        pd.testing.assert_frame_equal(cached['segments'],base['segments']);pd.testing.assert_frame_equal(cached['topics'],base['topics'])
        self.assertEqual(cached['visualization']['topic_similarity'],base['visualization']['topic_similarity'])
        self.assertEqual(cached['visualization']['topic_hierarchy'],base['visualization']['topic_hierarchy'])
        self.assertEqual(cached['visualization']['keyword_weights'],base['visualization']['keyword_weights'])


if __name__=='__main__':unittest.main(verbosity=2)
