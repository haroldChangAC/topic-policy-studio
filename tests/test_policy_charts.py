import copy
import io
import json
import unittest
import zipfile

import pandas as pd
from PIL import Image
from streamlit.testing.v1 import AppTest

from studio.corpus import read_txt
from studio.policy import (active_version, add_indicator, append_policies, confirm_indicators, record_encoding,
                           revise_policy, score)
from studio.policy_charts import (METHOD, SURFACE_METHOD, chart_settings, drawing_data, render_radar,
                                  render_surface, surface_data, validate_chart_settings)
from studio.policy_project import open_policy_project, save_policy_project
from studio.runtime import ROOT
from studio.visualization import find_chinese_font
from studio.workflow import csv_bytes
from test_policy import make_project, make_scheme


def sample_three():
    project=make_project();children=make_scheme(project)
    pid=project['policies'][0]['policy_id']
    for indicator,value in zip(children,['yes','yes','yes','no','no','no']):
        record_encoding(project,pid,indicator,value)
    return project


def sample_nine():
    project=make_project(); children=[]
    for index in range(9):
        parent=add_indicator(project,1,f'第{index+1}项中文政策指标LongLabel','指标定义')
        children.append([add_indicator(project,2,f'子项{index+1}-{part}','编码说明',parent) for part in (1,2)])
    confirm_indicators(project)
    for policy_index,policy in enumerate(project['policies']):
        for index,pair in enumerate(children):
            value=(index+policy_index)%3
            for offset,indicator in enumerate(pair):
                record_encoding(project,policy['policy_id'],indicator,'yes' if offset<value else 'no')
    return project,children


class PolicyChartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.font,_=find_chinese_font()

    def test_three_indicator_actual_scores_and_no_surface(self):
        project=sample_three();pid=project['policies'][0]['policy_id'];frame,blocked=drawing_data(project,[pid])
        self.assertFalse(blocked);self.assertEqual(frame['score'].tolist(),[1,.5,0])
        self.assertEqual(frame['numerator'].tolist(),[2,1,0]);self.assertEqual(frame['denominator'].tolist(),[2,2,2])
        self.assertTrue(frame['calculation_method'].eq(METHOD).all())
        settings=chart_settings(project);png,missing=render_radar(frame,settings,self.font)
        self.assertFalse(missing);self.assertEqual(Image.open(io.BytesIO(png)).size,(1200,900))
        with self.assertRaisesRegex(ValueError,'九个'):surface_data(frame,settings['surface_order'])

    def test_nine_mapping_reorder_roundtrip_and_export(self):
        project,_=sample_nine();pid=project['policies'][0]['policy_id'];frame,blocked=drawing_data(project,[pid]);self.assertFalse(blocked)
        settings=chart_settings(project);original=copy.deepcopy(score(project));ids=settings['surface_order'][:]
        matrix=surface_data(frame,ids);self.assertEqual(list(zip(matrix.matrix_row,matrix.matrix_column)),[(r,c) for r in (1,2,3) for c in (1,2,3)])
        self.assertEqual(matrix['score'].tolist(),[0,.5,1,0,.5,1,0,.5,1])
        reordered=list(reversed(ids));settings['surface_order']=reordered
        updated=surface_data(frame,reordered);self.assertEqual(updated['indicator_id'].tolist(),reordered)
        self.assertEqual(updated.iloc[0]['score'],matrix.iloc[-1]['score']);self.assertEqual(score(project),original)
        reread=pd.read_csv(io.BytesIO(csv_bytes(updated)))
        self.assertEqual(reread['indicator_id'].tolist(),reordered);self.assertEqual(reread['score'].tolist(),updated['score'].tolist())
        self.assertTrue(reread['surface_method'].eq(SURFACE_METHOD).all())
        settings['radar_policy_ids']=[pid];settings['surface_policy_id']=pid;validate_chart_settings(project,project['chart_settings'])
        raw=save_policy_project(project)
        with zipfile.ZipFile(io.BytesIO(raw)) as z:self.assertEqual(json.loads(z.read('project.json'))['version'],3)
        restored=open_policy_project(raw);self.assertEqual(restored,project)
        self.assertEqual(surface_data(drawing_data(restored,[pid])[0],restored['chart_settings']['by_version'][restored['active_version']]['surface_order'])['indicator_id'].tolist(),reordered)

    def test_incomplete_revisions_and_version_guards(self):
        project,children=sample_nine();pid=project['policies'][0]['policy_id'];second=project['policies'][1]['policy_id']
        old=project['active_version'];settings=chart_settings(project)
        for state,expected in [('pending','待核查'),('not_applicable','不适用')]:
            p=copy.deepcopy(project);record_encoding(p,pid,children[0][0],state)
            frame,blocked=drawing_data(p,[pid,second]);self.assertEqual(len(blocked),1);self.assertIn(expected,blocked[0]['reason'])
            with self.assertRaisesRegex(ValueError,'完成'):render_radar(frame,settings,self.font)
            with self.assertRaisesRegex(ValueError,'完成'):surface_data(frame[frame.policy_id==pid],settings['surface_order'])
        missing=copy.deepcopy(project);missing['encodings']=[e for e in missing['encodings'] if not(e['policy_id']==pid and e['indicator_id']==children[0][0])]
        self.assertIn('缺失编码',drawing_data(missing,[pid])[1][0]['reason'])
        stale=copy.deepcopy(project);revise_policy(stale,pid,stale['policies'][0]['original_text']+'新增')
        self.assertIn('待重新确认',drawing_data(stale,[pid])[1][0]['reason'])
        changed=copy.deepcopy(project);changed['draft'][0]['name']='已修改草稿'
        with self.assertRaisesRegex(ValueError,'草稿'):drawing_data(changed,[pid])
        confirm_indicators(changed)
        with self.assertRaisesRegex(ValueError,'版本'):drawing_data(changed,[pid],old)
        self.assertTrue(drawing_data(changed,[pid])[1])
        frame,_=drawing_data(project,[pid,second]);bad=frame.copy();bad.loc[bad.policy_id.eq(second),'indicator_version']='other'
        with self.assertRaisesRegex(ValueError,'版本'):render_radar(bad,settings,self.font)
        bad=frame.copy();bad.loc[bad.policy_id.eq(second),'calculation_method']='其他口径'
        with self.assertRaisesRegex(ValueError,'口径'):render_radar(bad,settings,self.font)

    def test_color_mono_png_svg_and_size(self):
        project,_=sample_nine();ids=[p['policy_id'] for p in project['policies']];frame,blocked=drawing_data(project,ids)
        self.assertFalse(blocked);settings=chart_settings(project);settings.update({'width_inches':10.0,'height_inches':8.0,'font_size':8})
        matrix=surface_data(frame[frame.policy_id==ids[0]],settings['surface_order'])
        for style in ('color','mono'):
            settings['style']=style
            for draw,data in ((render_radar,frame),(render_surface,matrix)):
                png,missing=draw(data,settings,self.font);self.assertFalse(missing)
                image=Image.open(io.BytesIO(png));self.assertEqual(image.size,(1500,1200));self.assertAlmostEqual(image.info['dpi'][0],150,delta=.1)
                self.assertIn(METHOD,image.info['Description']);self.assertIn(ids[0],image.info['Description'])
                svg,_=draw(data,settings,self.font,'svg');self.assertIn(b'<svg',svg);self.assertIn(b'</svg>',svg)
                self.assertIn(METHOD.encode(),svg);self.assertIn(ids[0].encode(),svg)
        small=sample_three();pid=small['policies'][0]['policy_id'];settings=chart_settings(small);settings.update({'dpi':600,'font_size':8})
        data,_=drawing_data(small,[pid]);png,_=render_radar(data,settings,self.font)
        image=Image.open(io.BytesIO(png));self.assertEqual(image.size,(4800,3600));self.assertAlmostEqual(image.info['dpi'][0],600,delta=.1)
        settings.update({'width_inches':4.0,'height_inches':3.0,'dpi':600})
        with self.assertRaisesRegex(ValueError,'画布'):render_surface(matrix,settings,self.font)
        settings.update({'width_inches':8.0,'height_inches':6.0})
        png,_=render_surface(matrix,settings,self.font)
        self.assertEqual(Image.open(io.BytesIO(png)).size,(4800,3600))

    def test_old_projects_and_app_selection_roundtrip(self):
        project,_=sample_nine();legacy=save_policy_project(project)
        with zipfile.ZipFile(io.BytesIO(legacy)) as z:self.assertEqual(json.loads(z.read('project.json'))['version'],1)
        legacy_open=open_policy_project(legacy);self.assertNotIn('chart_settings',legacy_open)
        append_policies(legacy_open,read_txt('另一份政策'.encode()),'text',{},'one.txt')
        v2=save_policy_project(legacy_open)
        with zipfile.ZipFile(io.BytesIO(v2)) as z:self.assertEqual(json.loads(z.read('project.json'))['version'],2)
        self.assertNotIn('chart_settings',open_policy_project(v2))
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
        app.radio(key='work_mode').set_value('基础政策评价').run();app.radio(key='pmc_mode').set_value('打开政策评价项目').run()
        app.file_uploader(key='pmc_project_upload').set_value(('old.zip',legacy,'application/zip')).run(timeout=90)
        self.assertFalse(app.exception);p=app.session_state['pmc_project'];vid=p['active_version']
        k=next(str(e.key) for e in app.multiselect if str(e.key).startswith('pmc_radar_policies_'));scope=k.replace('pmc_radar_policies_','')
        pids=[x['policy_id'] for x in p['policies']]
        app.multiselect(key=k).set_value(pids).run(timeout=90)
        app.selectbox(key='pmc_surface_policy_'+scope).select(pids[0]).run(timeout=90)
        names={str(e.key) for e in app.get('download_button')}
        self.assertTrue({'pmc_radar_png_'+scope,'pmc_radar_svg_'+scope,'pmc_radar_csv_'+scope,
                         'pmc_surface_png_'+scope,'pmc_surface_svg_'+scope,'pmc_surface_csv_'+scope}.issubset(names))
        ids=p['chart_settings']['by_version'][vid]['surface_order'][:]
        app.selectbox(key=f'pmc_matrix_{scope}_0').select(ids[1]).run(timeout=90)
        self.assertTrue(any('重复指标' in x.value for x in app.warning))
        self.assertEqual(p['chart_settings']['by_version'][vid]['surface_order'],ids)
        app.selectbox(key=f'pmc_matrix_{scope}_1').select(ids[0]).run(timeout=90)
        self.assertFalse(app.exception);self.assertEqual(p['chart_settings']['by_version'][vid]['surface_order'][:2],[ids[1],ids[0]])
        restored=open_policy_project(save_policy_project(p));self.assertEqual(restored['chart_settings'],p['chart_settings'])


if __name__=='__main__':unittest.main(verbosity=2)
