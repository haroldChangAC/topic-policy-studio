import copy
import io
import json
import unittest
import zipfile
from unittest.mock import patch

import pandas as pd
from openpyxl import Workbook, load_workbook
from streamlit.testing.v1 import AppTest

from studio.corpus import read_csv, read_txt, read_xlsx
from studio.policy import (active_version, append_policies, coding_matrix, confirm_indicators, create_project,
                           edit_indicator, plan_append, record_encoding, score)
from studio.policy_project import excel_policy, open_policy_project, save_policy_project, tables
from studio.runtime import ROOT
from studio.workflow import csv_bytes
from test_policy import make_project, make_scheme


def ready_project():
    project=make_project();children=make_scheme(project)
    for policy_index,policy in enumerate(project['policies']):
        for index,indicator in enumerate(children):
            status='yes' if (index+policy_index)%3 else 'no'
            if policy_index==0 and index==0:
                record_encoding(project,policy['policy_id'],indicator,status,'原文',0,2,coder='研究者')
            else:
                record_encoding(project,policy['policy_id'],indicator,status)
    return project,children


def sample_csv():
    return ('policy_id,标题,正文,地区,额外字段\n'
            '原始编号01,新增政策,新增中文政策正文。,广州,001\n'
            '不同编号,重复政策,"原文一。\r\n政策证据。",北京,002\n'
            '空白,没有正文,   ,首尔,003\n').encode('utf-8-sig')


class PolicyAppendTests(unittest.TestCase):
    def test_append_third_preserves_existing_and_complete_score(self):
        project,children=ready_project(); original=copy.deepcopy(project); before=score(project)
        old=save_policy_project(project)
        with zipfile.ZipFile(io.BytesIO(old)) as z:self.assertEqual(json.loads(z.read('project.json'))['version'],1)
        project=open_policy_project(old); corpus=read_csv(sample_csv())
        preview=plan_append(project,corpus,'正文',{'title':'标题','region':'地区'},'新增.csv','policy_id','exclude')
        self.assertEqual((preview['valid_count'],preview['empty_count'],preview['duplicate_count'],len(preview['accepted'])),(2,1,1,1))
        self.assertEqual(preview['external_id_conflicts'],1)
        self.assertEqual(project,original)
        append_policies(project,corpus,'正文',{'title':'标题','region':'地区'},'新增.csv','policy_id','exclude')
        self.assertEqual(project['policies'][:2],original['policies'])
        self.assertEqual(project['versions'],original['versions']);self.assertEqual(project['encodings'],original['encodings'])
        self.assertEqual(score(project)[1][:2],before[1]);self.assertEqual(score(project)[0][:6],before[0][:6])
        third=project['policies'][2];states=[x for x in coding_matrix(project) if x['policy_id']==third['policy_id']]
        self.assertEqual(len(states),6);self.assertTrue(all(x['status']=='待编码' and x['binary_value'] is None for x in states))
        self.assertIsNone(score(project)[1][2]['raw_pmc']);self.assertEqual(third['source_original_id'],'原始编号01')
        self.assertEqual(third['source_id_conflict_with'],project['policies'][0]['policy_id'])
        export=tables(project);row=export['policy_scores'].iloc[2]
        self.assertEqual((row['coding_status'],row['uncoded_count']),('未完成',6))
        pending=export['encodings'];pending=pending[pending.policy_id==third['policy_id']]
        self.assertEqual(len(pending),6);self.assertTrue(pending['event_id'].eq('').all())
        self.assertTrue(pending['status_label'].eq('待编码').all())
        self.assertTrue(pending['record_type'].eq('待编码状态（非人工记录）').all())
        matrix=pd.read_csv(io.BytesIO(csv_bytes(export['coding_status'])),keep_default_na=False)
        self.assertEqual(len(matrix[(matrix.policy_id==third['policy_id']) & (matrix.status=='待编码')]),6)
        raw=save_policy_project(project)
        with zipfile.ZipFile(io.BytesIO(raw)) as z:self.assertEqual(json.loads(z.read('project.json'))['version'],2)
        reopened=open_policy_project(raw);self.assertEqual(reopened,project)
        for indicator,value in zip(children,['yes','yes','yes','no','no','no']):record_encoding(reopened,third['policy_id'],indicator,value)
        self.assertEqual(score(reopened)[1][2]['raw_pmc'],1.5)
        self.assertEqual(score(reopened)[1][:2],before[1])
        self.assertEqual(tables(reopened)['policy_scores'].iloc[2]['coding_status'],'已完成')
        self.assertEqual(corpus.raw,sample_csv())

    def test_duplicate_choice_and_same_source_id_collision(self):
        project,_=ready_project();raw=sample_csv();corpus=read_csv(raw)
        keep=plan_append(project,corpus,'正文',{'title':'标题'},'新增.csv','policy_id','keep')
        self.assertEqual(len(keep['accepted']),2);self.assertEqual(keep['duplicate_count'],1)
        append_policies(project,corpus,'正文',{'title':'标题'},'新增.csv','policy_id','keep')
        second=plan_append(project,corpus,'正文',{'title':'标题'},'新增.csv','policy_id','keep')
        self.assertEqual(second['system_id_collisions'],2)
        self.assertTrue(all(':append-000001' in x['policy_id'] for x in second['rows'][:2]))
        append_policies(project,corpus,'正文',{'title':'标题'},'新增.csv','policy_id','keep')
        self.assertEqual(len({p['policy_id'] for p in project['policies']}),6)
        self.assertEqual(project['policies'][4]['source_record_id'],project['policies'][2]['source_record_id'])
        self.assertEqual(open_policy_project(save_policy_project(project)),project)

    def test_xlsx_txt_multiple_sources_long_text_and_exports(self):
        project,_=ready_project();book=Workbook();sheet=book.active;sheet.title='政策'
        long='多段中文政策，须保留空白与标点。\n'*600
        sheet.append(['标题','正文','发布机构','编号']);sheet.append(['长政策',long,'示例机构','Z01']);sheet.append(['另一政策','另有正文','机构二','Z02'])
        data=io.BytesIO();book.save(data);xlsx=data.getvalue();corpus=read_xlsx(xlsx,'政策')
        self.assertEqual(plan_append(project,corpus,'正文',{'title':'标题'},'样例.xlsx','编号')['rows'][0]['source_row'],2)
        append_policies(project,corpus,'正文',{'title':'标题','agency':'发布机构'},'样例.xlsx','编号','keep')
        self.assertEqual(project['policies'][2]['original_text'],long)
        txt='  整份政策文本。\r\n第二段内容。\n';raw=txt.encode('utf-8-sig')
        reopened=open_policy_project(save_policy_project(project))
        append_policies(reopened,read_txt(raw),'text',{},'附加.txt',None,'keep')
        reopened2=open_policy_project(save_policy_project(reopened))
        self.assertEqual(reopened2['policies'][-1]['original_text'],txt)
        self.assertEqual(reopened2['policies'][-1]['title'],'附加.txt')
        self.assertEqual(len(reopened2['sources']),2)
        frames=tables(reopened2);p=frames['policies'];self.assertEqual(p.iloc[2]['raw::发布机构'],'示例机构')
        self.assertEqual(p.iloc[-1]['raw::text'],txt);self.assertEqual(len(frames['import_sources']),3)
        restored_csv=pd.read_csv(io.BytesIO(csv_bytes(p)),dtype=str,keep_default_na=False)
        self.assertEqual(restored_csv.iloc[2]['original_text'],long);self.assertEqual(restored_csv.iloc[-1]['raw::text'],txt)
        x=load_workbook(io.BytesIO(excel_policy(reopened2)),data_only=True)
        self.assertIn('coding_status',x.sheetnames);self.assertIn('import_sources',x.sheetnames)
        heads=[x.value for x in x['policies'][1]];self.assertEqual(x['policies'].cell(4,heads.index('original_text')+1).value,long)

    def test_empty_invalid_and_unconfirmed(self):
        project=make_project();before=copy.deepcopy(project)
        with self.assertRaises(ValueError):read_txt(b'')
        with self.assertRaises(ValueError):read_csv(b'')
        empty=read_csv('标题,正文\n空文,   \n'.encode())
        with self.assertRaisesRegex(ValueError,'没有可追加'):append_policies(project,empty,'正文',{'title':'标题'})
        self.assertEqual(project,before)
        workbook=Workbook();sheet=workbook.active;sheet.append(['标题','正文']);sheet.append(['无缓存公式','=A2'])
        buffer=io.BytesIO();workbook.save(buffer)
        with self.assertRaisesRegex(ValueError,'公式'):
            append_policies(project,read_xlsx(buffer.getvalue(),sheet.title),'正文',{'title':'标题'})
        self.assertEqual(project,before)
        append_policies(project,read_txt('新政策正文'.encode()),'text',{},'新政策.txt')
        self.assertIsNone(project['active_version']);self.assertEqual(coding_matrix(project),[])
        self.assertTrue(all(x['raw_pmc'] is None and '尚未确认' in x['reason'] for x in score(project)[1]))
        self.assertEqual(open_policy_project(save_policy_project(project)),project)

    def test_duplicates_inside_batch_and_external_id_matching_system_id(self):
        project,_=ready_project(); existing=project['policies'][0]['policy_id']
        raw=('政策ID,正文\n'+existing+',独特的新政策\nother,独特的新政策\n').encode('utf-8-sig')
        corpus=read_csv(raw)
        plan=plan_append(project,corpus,'正文',{},'批次.csv','政策ID','exclude')
        self.assertEqual((plan['valid_count'],plan['duplicate_count'],plan['external_id_conflicts'],len(plan['accepted'])),(2,1,1,1))
        self.assertEqual(plan['rows'][0]['source_id_conflict_with'],existing)
        append_policies(project,corpus,'正文',{},'批次.csv','政策ID','exclude')
        self.assertEqual(len(project['policies']),3)
        self.assertEqual(project['policies'][-1]['source_original_id'],existing)
        self.assertEqual(open_policy_project(save_policy_project(project)),project)

    def test_custom_id_columns_can_be_mapped_across_sources(self):
        first=read_csv('标题,正文,旧档案号\n政策一,第一份文本,X-17\n'.encode())
        project=create_project(first,'正文',{'title':'标题'})
        incoming=read_csv('名称,文本,新来源编号\n政策二,另一份文本,X-17\n'.encode())
        plan=plan_append(project,incoming,'文本',{'title':'名称'},'新.csv','新来源编号','keep','旧档案号')
        self.assertEqual(plan['external_id_conflicts'],1)
        append_policies(project,incoming,'文本',{'title':'名称'},'新.csv','新来源编号','keep','旧档案号')
        self.assertEqual(project['sources'][0]['existing_id_column'],'旧档案号')
        self.assertEqual(project['policies'][-1]['source_id_conflict_with'],project['policies'][0]['policy_id'])
        self.assertEqual(open_policy_project(save_policy_project(project)),project)

    def test_page_open_append_code_and_reopen(self):
        project,children=ready_project();old=save_policy_project(project)
        with patch('socket.socket.connect',side_effect=AssertionError('network')):
            app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
            app.radio(key='work_mode').set_value('基础政策评价').run()
            app.radio(key='pmc_mode').set_value('打开政策评价项目').run()
            app.file_uploader(key='pmc_project_upload').set_value(('old.pmc.zip',old,'application/zip')).run()
            app.file_uploader(key='pmc_append_upload').set_value(('new.csv',sample_csv(),'text/csv')).run()
            self.assertTrue(any('相同' in w.value for w in app.warning))
            app.radio(key='pmc_append_duplicates').set_value('排除相同原文').run()
            app.button(key='pmc_append_confirm').click().run()
            self.assertFalse(app.exception);self.assertFalse(app.error)
            result=app.session_state['pmc_project'];self.assertEqual(len(result['policies']),3)
            third=result['policies'][-1]['policy_id'];app.selectbox(key='pmc_code_policy').select(third).run()
            self.assertEqual(app.selectbox(key='pmc_value').value,'请选择')
            app.selectbox(key='pmc_value').select('1：符合')
            app.button(key='pmc_save_manual').click().run()
            self.assertEqual(result['encodings'][-1]['policy_id'],third)
            saved=save_policy_project(result);reopened=open_policy_project(saved)
            self.assertEqual(reopened,result);self.assertEqual(reopened['policies'][:2],project['policies'])

    def test_page_switch_project_resets_append_selection(self):
        first,_=ready_project();second,_=ready_project()
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
        app.radio(key='work_mode').set_value('基础政策评价').run()
        app.radio(key='pmc_mode').set_value('打开政策评价项目').run()
        app.file_uploader(key='pmc_project_upload').set_value(('one.zip',save_policy_project(first),'application/zip')).run()
        app.file_uploader(key='pmc_append_upload').set_value(('new.csv',sample_csv(),'text/csv')).run()
        app.selectbox(key='pmc_append_id').select('额外字段').run()
        app.radio(key='pmc_append_duplicates').set_value('排除相同原文').run()
        app.file_uploader(key='pmc_project_upload').set_value(('two.zip',save_policy_project(second),'application/zip')).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox(key='pmc_append_id').value,'policy_id')
        self.assertEqual(app.radio(key='pmc_append_duplicates').value,'保留所有有效政策（默认）')

    def test_new_indicator_version_after_append_does_not_mix_scores(self):
        project,children=ready_project()
        append_policies(project,read_txt('新增政策文本'.encode()),'text',{},'新.txt')
        third=project['policies'][-1]['policy_id']
        for indicator in children:record_encoding(project,third,indicator,'yes')
        old_version=project['active_version'];old_results=score(project,old_version)
        edit_indicator(project,children[0],'修订后指标','新的编码说明')
        self.assertTrue(confirm_indicators(project))
        self.assertNotEqual(project['active_version'],old_version)
        self.assertEqual(score(project,old_version),old_results)
        self.assertTrue(all(row['status']=='待编码' for row in coding_matrix(project)))
        self.assertTrue(all(row['raw_pmc'] is None for row in score(project)[1]))
        reopened=open_policy_project(save_policy_project(project))
        self.assertEqual(score(reopened,old_version),old_results)
        self.assertEqual(reopened['active_version'],project['active_version'])


if __name__=='__main__':unittest.main(verbosity=2)
