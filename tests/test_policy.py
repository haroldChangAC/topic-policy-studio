import copy
import io
import json
import unittest
import zipfile
from unittest.mock import patch

import pandas as pd
from openpyxl import load_workbook
from streamlit.testing.v1 import AppTest

from studio.corpus import read_csv, read_txt, read_xlsx
from studio.policy import (active_version, add_indicator, confirm_indicators, create_project, delete_indicator,
                           edit_indicator, record_encoding, revise_policy, score)
from studio.policy_project import excel_policy, open_policy_project, save_policy_project, tables
from studio.project import save_project
from studio.runtime import ROOT
from studio.workflow import csv_bytes
from test_paper_style import sample_result


def make_project():
    raw='标题,正文,地区,policy_id\r\n政策甲,"原文一。\r\n政策证据。",北京,原始编号01\r\n政策乙,另一份原文,首尔,原始编号02\r\n'.encode('utf-8-sig')
    return create_project(read_csv(raw),'正文',{'title':'标题','region':'地区'},'test.csv')


def make_scheme(project):
    children=[]
    for name in ('A','B','C'):
        parent=add_indicator(project,1,name,'一级说明')
        children.extend(add_indicator(project,2,name+str(i),'二级说明',parent) for i in (1,2))
    confirm_indicators(project)
    return children


class PolicyTests(unittest.TestCase):
    def test_hand_calculation_multiple_and_incomplete(self):
        p=make_project(); children=make_scheme(p); a,b=[x['policy_id'] for x in p['policies']]
        for iid,value in zip(children,['yes','yes','yes','no','no','no']):record_encoding(p,a,iid,value)
        primary, totals=score(p)
        self.assertEqual([x['score'] for x in primary if x['policy_id']==a],[1,.5,0])
        self.assertEqual(totals[0]['raw_pmc'],1.5);self.assertEqual(totals[0]['theoretical_max'],3)
        self.assertEqual(totals[0]['normalized_10'],5)
        self.assertIsNone(totals[1]['raw_pmc']);self.assertIn('缺失编码',totals[1]['reason'])
        record_encoding(p,b,children[0],'pending');record_encoding(p,b,children[1],'not_applicable')
        _,totals=score(p);self.assertIsNone(totals[1]['raw_pmc'])
        self.assertIn('待核查',totals[1]['reason']);self.assertIn('不适用',totals[1]['reason'])
        self.assertIn('缺失编码',totals[1]['reason'])
        for iid in children:record_encoding(p,b,iid,'no')
        self.assertEqual(score(p)[1][1]['raw_pmc'],0.0)
        empty=make_project();self.assertIsNone(score(empty)[1][0]['raw_pmc'])
        with self.assertRaisesRegex(ValueError,'至少'):confirm_indicators(empty)
        parent=add_indicator(empty,1,'空一级','')
        with self.assertRaisesRegex(ValueError,'二级'):confirm_indicators(empty)
        add_indicator(empty,2,'有二级','说明',parent)
        with self.assertRaisesRegex(ValueError,'定义'):confirm_indicators(empty)

    def test_evidence_versioning_and_history(self):
        p=make_project();children=make_scheme(p);pid=p['policies'][0]['policy_id'];text=p['policies'][0]['original_text']
        start=text.index('政策证据');end=start+len('政策证据')
        with self.assertRaisesRegex(ValueError,'不一致'):record_encoding(p,pid,children[0],'yes','错误',start,end)
        record_encoding(p,pid,children[0],'yes','政策证据',start,end,coder='')
        self.assertFalse(record_encoding(p,pid,children[0],'yes','政策证据',start,end,coder=''))
        record_encoding(p,pid,children[0],'no',rationale='人工判断')
        self.assertEqual(len(p['encodings']),2)
        old=active_version(p)['version_id'];original=copy.deepcopy(p['versions'][0])
        edit_indicator(p,children[0],'修改指标','更新定义')
        self.assertTrue(confirm_indicators(p));self.assertEqual(p['versions'][0],original)
        self.assertNotEqual(active_version(p)['version_id'],old);self.assertEqual(len(p['encodings']),2)
        self.assertIsNone(score(p)[1][0]['raw_pmc'])
        self.assertIn('缺失编码',score(p)[1][0]['reason'])
        record_encoding(p,pid,children[0],'yes','政策证据',start,end)
        revise_policy(p,pid,text+'修订','补充')
        self.assertEqual(p['policies'][0]['original_text'],text)
        self.assertIn('待重新确认',score(p)[1][0]['reason'])
        old_version=p['versions'][-1]
        delete_indicator(p,children[0]);confirm_indicators(p)
        self.assertIn(children[0],[i['id'] for i in old_version['indicators']])
        with self.assertRaisesRegex(ValueError,'位置'):record_encoding(p,pid,children[1],'yes','政策',0,1)

    def test_csv_xlsx_txt_and_exact_roundtrip(self):
        txt=read_txt('首行\r\n  第二行。'.encode());p=create_project(txt,'text',{},'政策.txt')
        self.assertEqual(p['policies'][0]['original_text'],'首行\r\n  第二行。')
        self.assertEqual(p['policies'][0]['title'],'政策.txt')
        raw=(ROOT/'data/examples/import_formats.xlsx').read_bytes(); x=read_xlsx(raw,'政策')
        xproj=create_project(x,'正文',{},'政策.xlsx');self.assertGreater(len(xproj['policies']),1)
        self.assertEqual(xproj['source']['details']['sheet'],'政策')
        formulas=read_xlsx(raw,'公式示例')
        with self.assertRaisesRegex(ValueError,'公式'):create_project(formulas,'正文',{},'bad.xlsx')
        p=make_project();children=make_scheme(p);pid=p['policies'][0]['policy_id']
        start=p['policies'][0]['original_text'].index('政策证据')
        record_encoding(p,pid,children[0],'yes','政策证据',start,start+4,coder='研究者')
        b=save_policy_project(p);loaded=open_policy_project(b);self.assertEqual(loaded,p)
        out=tables(loaded);csv=pd.read_csv(io.BytesIO(csv_bytes(out['policies'])),keep_default_na=False,dtype=str)
        self.assertEqual(csv.loc[0,'original_text'],p['policies'][0]['original_text'])
        self.assertEqual(csv.loc[0,'raw::policy_id'],'原始编号01')
        book=load_workbook(io.BytesIO(excel_policy(loaded)),data_only=True)
        self.assertTrue({'policies','indicators','encodings','primary_scores','policy_scores','text_revisions','calculation_rule'}.issubset(book.sheetnames))
        heads=[x.value for x in book['policies'][1]];self.assertEqual(book['policies'].cell(2,heads.index('original_text')+1).value,p['policies'][0]['original_text'])

    def test_corrupt_and_theme_project_is_distinct(self):
        p=make_project();raw=save_policy_project(p)
        with self.assertRaises(ValueError):open_policy_project(b'not zip')
        with zipfile.ZipFile(io.BytesIO(raw)) as z:contents={n:z.read(n) for n in z.namelist()}
        contents['../bad']=b'bad';buf=io.BytesIO()
        with zipfile.ZipFile(buf,'w') as z:
            for k,v in contents.items():z.writestr(k,v)
        with self.assertRaisesRegex(ValueError,'路径'):open_policy_project(buf.getvalue())
        with self.assertRaises(ValueError):open_policy_project(save_project(sample_result()))

    def test_page_import_example_encode_compute_and_reopen(self):
        with patch('socket.socket.connect',side_effect=AssertionError('network')):
            app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
            app.radio(key='work_mode').set_value('基础政策评价').run()
            app.file_uploader(key='pmc_upload').set_value(('policies.csv','标题,正文,地区\n政策甲,原文证据,北京\n政策乙,另一政策,首尔\n'.encode(),'text/csv')).run()
            app.button(key='pmc_create').click().run()
            self.assertFalse(app.exception);app.button(key='pmc_example').click().run()
            app.button(key='pmc_confirm').click().run()
            p=app.session_state['pmc_project'];self.assertIsNotNone(p['active_version'])
            app.selectbox(key='pmc_value').select('1：符合')
            app.text_area(key='pmc_evidence').set_value('原文')
            app.checkbox(key='pmc_precise').check()
            app.number_input(key='pmc_start').set_value(0);app.number_input(key='pmc_end').set_value(2)
            app.button(key='pmc_save_manual').click().run()
            self.assertFalse(app.exception);self.assertFalse(app.error)
            self.assertEqual(len(p['encodings']),1);self.assertEqual(p['encodings'][0]['evidence'],'原文')
            other=next(i['id'] for i in active_version(p)['indicators'] if i['level']==2 and i['id']!=p['encodings'][0]['indicator_id'])
            app.selectbox(key='pmc_code_indicator').select(other).run()
            self.assertEqual(app.selectbox(key='pmc_value').value,'请选择')
            self.assertEqual(app.text_area(key='pmc_evidence').value,'')
            app.button(key='pmc_calculate').click().run()
            self.assertTrue(any('不能计算完整总分' in w.value for w in app.warning))
            self.assertIn('pmc_download_project',[e.key for e in app.get('download_button')])
            exported=save_policy_project(p)
            app.radio(key='pmc_mode').set_value('打开政策评价项目').run()
            app.file_uploader(key='pmc_project_upload').set_value(('project.zip',exported,'application/zip')).run()
            self.assertFalse(app.exception);self.assertEqual(app.session_state['pmc_project']['encodings'],p['encodings'])

    def test_page_long_policy_split_using_local_model(self):
        with patch('socket.socket.connect',side_effect=AssertionError('network')):
            app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
            app.radio(key='work_mode').set_value('基础政策评价').run()
            text='第一段政策论述，保护环境和推动产业转型。\n'*150
            app.file_uploader(key='pmc_upload').set_value(('long.txt',text.encode(),'text/plain')).run()
            app.button(key='pmc_create').click().run()
            app.button(key='pmc_split').click().run(timeout=90)
            self.assertFalse(app.exception);self.assertFalse(app.error)
            split=app.session_state['pmc_split_result']
            self.assertGreater(len(split['rows']),1)
            self.assertEqual(''.join(x['text'] for x in split['rows']),text)
            self.assertTrue(all(x['tokens']<=split['limit'] and text[x['start']:x['end']]==x['text'] for x in split['rows']))
            self.assertEqual(len(app.session_state['pmc_project']['policies']),1)


if __name__=='__main__':unittest.main(verbosity=2)
