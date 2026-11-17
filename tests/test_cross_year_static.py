import copy
import io
import unittest
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from studio.cross_year import calculate, defaults, graph
from studio.cross_year_static import default_export, render, selected_data, validate_export
from studio.deferred_exports import create_export, fingerprint
from studio.project import open_project,save_project
from test_cross_year import yearly_sample


class StaticSankeyTests(unittest.TestCase):
    def test_static_matches_interactive_and_dpi(self):
        result=yearly_sample();settings=defaults(result);settings['threshold']=.5
        nodes,frame,_,_=calculate(result,settings);active,edges=selected_data(nodes,frame)
        interactive=graph(nodes,frame,settings);self.assertEqual([e['weight'] for e in edges],list(interactive.data[0].link.value))
        self.assertEqual([e['weight'] for e in edges],frame.loc[frame.in_graph,'line_weight'].astype(float).tolist())
        for style in ('color','mono'):
            settings['style']=style
            png,meta=render(nodes,frame,settings,default_export(),'png')
            img=Image.open(io.BytesIO(png));self.assertEqual(img.size,(2100,1350));self.assertAlmostEqual(img.info['dpi'][0],150,delta=.1)
            self.assertEqual(meta['edge_weights'],[e['weight'] for e in edges]);self.assertEqual(len(meta['node_ids']),len(active))
            scales=[thickness/weight for thickness,weight in zip(meta['ribbon_thickness_inches'],meta['edge_weights'])]
            self.assertTrue(np.allclose(scales,scales[0]))
            if style=='mono':
                rgb=np.asarray(img.convert('RGB'));self.assertTrue(np.array_equal(rgb[:,:,0],rgb[:,:,1]) and np.array_equal(rgb[:,:,1],rgb[:,:,2]))
            svg,_=render(nodes,frame,settings,default_export(),'svg');root=ET.fromstring(svg);self.assertTrue(root.tag.endswith('svg'))
            self.assertIn('相邻自然年份主题相似关联',svg.decode())
        size600={'width':8.,'height':6.,'dpi':600,'font_size':7}
        png,_=render(nodes,frame,settings,size600,'png');img=Image.open(io.BytesIO(png))
        self.assertEqual(img.size,(4800,3600));self.assertAlmostEqual(img.info['dpi'][0],600,delta=.1)

    def test_empty_dense_and_long_labels_fail_clearly(self):
        r=yearly_sample();s=defaults(r);nodes,frame,_,_=calculate(r,s)
        empty=frame.copy();empty['in_graph']=False
        with self.assertRaisesRegex(ValueError,'没有符合'):render(nodes,empty,s,default_export(),'png')
        dense=pd_concat([frame[frame.in_graph]]*41)
        with self.assertRaisesRegex(ValueError,'80'):render(nodes,dense,s,default_export(),'png')
        moderate=copy.deepcopy(nodes);moderate[0]['label']='长期公共政策与社会发展研究主题说明'
        data,_=render(moderate,frame,s,default_export(),'svg');self.assertIn('公共政策',data.decode())
        long=copy.deepcopy(nodes);long[0]['label']='超长中文研究主题'*50
        with self.assertRaisesRegex(ValueError,'超出|重叠'):render(long,frame,s,default_export(),'png')
        with self.assertRaisesRegex(ValueError,'3600'):validate_export({'width':16.,'height':14.,'dpi':600,'font_size':9})

    def test_deferred_and_project_settings_roundtrip(self):
        result=yearly_sample();result['visualization']['cross_year_links']=defaults(result)
        result['visualization']['cross_year_export']=default_export();base=fingerprint(result)
        a=create_export(result,'cross_png');b=create_export(result,'cross_svg')
        self.assertTrue(a['data'].startswith(b'\x89PNG'));self.assertIn(b'<svg',b['data'])
        opened=open_project(save_project(result));self.assertEqual(opened['visualization']['cross_year_export'],default_export())
        result['visualization']['cross_year_export']['font_size']=8
        self.assertNotEqual(base,fingerprint(result))


def pd_concat(frames):
    import pandas as pd
    return pd.concat(frames,ignore_index=True)
