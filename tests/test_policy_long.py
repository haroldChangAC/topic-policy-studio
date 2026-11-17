import copy
import csv
import hashlib
import io
import json
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from openpyxl import load_workbook
from streamlit.testing.v1 import AppTest

from studio.corpus import read_csv
from studio.policy import add_indicator, confirm_indicators, create_project, edit_indicator, record_encoding, revise_policy, score
from studio.policy_assist import make_prompt, request_ollama, review
from studio.policy_long import (PARAMETERS, PROMPT_BYTE_BUDGET, cancel_remaining, candidates, coverage,
                                create_job, is_current, process_segment, split_ranges)
from studio.policy_project import excel_policy, open_policy_project, save_policy_project, tables
from studio.runtime import ROOT


FIXTURE_SHA='783bad3b7af15b32739743025e3491d183532152b0b01f455df41fafd8b68305'


def fixture():
    raw=(ROOT/'tests/fixtures/policy_long_eval_2026-11-07.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==FIXTURE_SHA
    return json.loads(raw)


def make_long_project():
    item=fixture();out=io.StringIO(newline='');writer=csv.writer(out);writer.writerow(['标题','正文']);writer.writerow(['合成政策',item['text']])
    p=create_project(read_csv(out.getvalue().encode()),'正文',{'title':'标题'})
    parent=add_indicator(p,1,'示例一级','合成测试')
    iid=add_indicator(p,2,item['indicator']['name'],item['indicator']['definition'],parent);confirm_indicators(p)
    return p,iid,item


def mock_response(value,evidence):
    return json.dumps({'status':value,'rationale':'合成模拟结果，须人工核对。','evidence':evidence},ensure_ascii=False)


class PolicyLongTests(unittest.TestCase):
    def test_split_offsets_boundary_overlap_budget_and_natural_breaks(self):
        p,iid,item=make_long_project();indicator=p['versions'][0]['indicators'][1];text=item['text']
        spans,cfg=split_ranges(text,indicator,40);self.assertGreaterEqual(len(spans),4)
        self.assertEqual(spans[0]['end'],item['no_overlap_first_boundary'])
        crossing=item['cross_boundary_position'];self.assertLess(crossing,spans[0]['end']);self.assertGreater(crossing+len(item['evidence']),spans[0]['end'])
        self.assertTrue(any(s['start']<=crossing and crossing+len(item['evidence'])<=s['end'] for s in spans))
        self.assertEqual(spans[0]['start'],0);self.assertEqual(spans[-1]['end'],len(text))
        for i,s in enumerate(spans):
            self.assertLessEqual(len(make_prompt(indicator,text,s['start'],s['end']).encode()),PROMPT_BYTE_BUDGET)
            if i:self.assertEqual(s['overlap_before'],spans[i-1]['end']-s['start'])
            self.assertEqual(text[s['start']:s['end']],item['text'][s['start']:s['end']])
        self.assertEqual(cfg['request_parameters'],PARAMETERS);self.assertIn('非 Qwen tokenizer',cfg['estimation'])
        natural=('第一段内容。\n\n'+'第二段内容。\n\n')*160
        parts,_=split_ranges(natural,indicator,0)
        self.assertTrue(all(natural[s['end']-1]=='\n' for s in parts[:-1]))
        with self.assertRaises(ValueError):split_ranges(' ',indicator,40)

    def test_fail_cancel_retry_deduplicate_and_review(self):
        p,iid,item=make_long_project();pid=p['policies'][0]['policy_id'];before=score(p)
        job=create_job(p,pid,iid,'mock-local',40)
        responder=lambda *a:mock_response('yes',item['evidence'])
        process_segment(p,job,1,requester=responder)
        first=p['ai_suggestions'][0]
        with self.assertRaisesRegex(ValueError,'分段未完成'):
            review(p,first['suggestion_id'],'modify','yes',item['evidence'],12,20,'仅检查部分',coder='合成测试')
        self.assertFalse(p['encodings'])
        failed=process_segment(p,job,2,requester=lambda *a:(_ for _ in ()).throw(ConnectionError('中断')))
        self.assertEqual(failed['result'],'failed');self.assertEqual(job['segments'][1]['status'],'failed')
        cancel_remaining(p,job);self.assertFalse(coverage(job)['all_returned'])
        self.assertLess(coverage(job)['returned_characters'],len(item['text']))
        raw=save_policy_project(p)
        with zipfile.ZipFile(io.BytesIO(raw)) as z:self.assertEqual(json.loads(z.read('project.json'))['version'],6)
        p=open_policy_project(raw);job=p['long_jobs'][0]
        for segment in job['segments']:
            if segment['status']!='returned':process_segment(p,job,segment['order'],requester=responder)
        self.assertTrue(coverage(job)['all_returned']);self.assertEqual(coverage(job)['returned_characters'],len(item['text']))
        self.assertEqual(len(p['ai_suggestions']),len(job['segments'])+1)
        with self.assertRaisesRegex(ValueError,'不重复'):process_segment(p,job,1,requester=responder)
        found,unresolved=candidates(p,job)
        self.assertEqual([[x['start'],x['end']] for x in found],item['expected_positions'])
        duplicate=next(c for c in found if c['start']==item['overlap_duplicate_position'])
        self.assertEqual(duplicate['segment_orders'],[1,2]);self.assertEqual(len(duplicate['suggestion_ids']),2)
        crossing=next(c for c in found if c['start']==item['cross_boundary_position'])
        self.assertEqual(crossing['segment_orders'],[2]);self.assertTrue(unresolved)
        self.assertEqual(score(p),before);self.assertFalse(p['encodings'])
        chosen=found[0];review(p,chosen['suggestion_ids'][0],'accept',start=chosen['start'],end=chosen['end'],coder='合成测试')
        self.assertEqual(score(p)[1][0]['raw_pmc'],1);self.assertEqual(p['encodings'][0]['evidence'],item['evidence'])
        reopened=open_policy_project(save_policy_project(p));self.assertEqual(reopened,p)
        frames=tables(reopened);self.assertEqual(len(frames['long_evidence']),len(item['expected_positions']))
        workbook=load_workbook(io.BytesIO(excel_policy(reopened)),read_only=True)
        self.assertIn('long_segments',workbook.sheetnames);self.assertIn('long_evidence',workbook.sheetnames)

    def test_nonmatching_and_stale_policy_or_version(self):
        p,iid,item=make_long_project();pid=p['policies'][0]['policy_id'];job=create_job(p,pid,iid,'mock-local')
        process_segment(p,job,1,requester=lambda *a:mock_response('yes','虚构或改写的证据'))
        found,unresolved=candidates(p,job);self.assertFalse(found);self.assertIn('未精确匹配',unresolved[0]['reason'])
        old=copy.deepcopy(p);revise_policy(p,pid,item['text']+'修订')
        self.assertFalse(is_current(p,job))
        with self.assertRaisesRegex(ValueError,'已改变'):process_segment(p,job,2,requester=lambda *a:'{}')
        with self.assertRaisesRegex(ValueError,'变化'):review(p,p['ai_suggestions'][0]['suggestion_id'],'accept')
        self.assertEqual(open_policy_project(save_policy_project(p)),p)
        edit_indicator(old,iid,'修改后指标','修改后定义');confirm_indicators(old)
        self.assertFalse(is_current(old,old['long_jobs'][0]))
        with self.assertRaisesRegex(ValueError,'已改变'):process_segment(old,old['long_jobs'][0],2,requester=lambda *a:'{}')

    def test_existing_formal_encoding_is_not_overwritten(self):
        p,iid,item=make_long_project();pid=p['policies'][0]['policy_id']
        record_encoding(p,pid,iid,'no',rationale='先前人工判断');before=copy.deepcopy(p['encodings']);old=score(p)
        job=create_job(p,pid,iid,'mock-local')
        for segment in job['segments']:
            process_segment(p,job,segment['order'],requester=lambda *a:mock_response('yes',item['evidence']))
        self.assertEqual(p['encodings'],before);self.assertEqual(score(p),old)
        candidate=candidates(p,job)[0][0]
        with self.assertRaisesRegex(ValueError,'已有正式编码'):
            review(p,candidate['suggestion_ids'][0],'accept',start=candidate['start'],end=candidate['end'])
        self.assertEqual(p['encodings'],before)
        p['long_jobs'][0]['segments'][0]['end']+=1
        with self.assertRaisesRegex(ValueError,'范围|规则'):save_policy_project(p)

    def test_real_request_parameters_are_sent(self):
        class Reply:
            def __enter__(self):return self
            def __exit__(self,*a):return False
            def read(self,_n):return json.dumps({'done':True,'model':'mock-local','response':mock_response('pending','')}).encode()
        class Opener:
            def open(self,req,timeout):self.req=req;return Reply()
        opener=Opener()
        with patch('studio.policy_assist.urllib.request.build_opener',return_value=opener):
            request_ollama('http://127.0.0.1:11434','mock-local','内容',options=PARAMETERS)
        self.assertEqual(json.loads(opener.req.data)['options'],PARAMETERS)

    def test_page_create_process_and_review_mock(self):
        p,iid,item=make_long_project();raw=save_policy_project(p)
        with patch('studio.policy_assist.request_ollama',return_value=mock_response('yes',item['evidence'])):
            app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
            app.radio(key='work_mode').set_value('基础政策评价').run()
            app.radio(key='pmc_mode').set_value('打开政策评价项目').run()
            app.file_uploader(key='pmc_project_upload').set_value(('long.pmc.zip',raw,'application/zip')).run(timeout=90)
            self.assertFalse(app.exception);app.text_input(key='long_model').set_value('mock-local').run()
            app.button(key='long_create').click().run(timeout=90)
            job=app.session_state['pmc_project']['long_jobs'][0]
            app.number_input(key='long_batch_amount').set_value(5).run()
            app.button(key='long_batch').click().run(timeout=90)
            self.assertFalse(app.exception);self.assertTrue(coverage(job)['all_returned'])
            actual=app.session_state['pmc_project'];self.assertFalse(actual['encodings'])
            key=next(str(x.key) for x in app.button if str(x.key).startswith('long_review_submit_'))
            app.button(key=key).click().run(timeout=90)
            self.assertFalse(app.exception);self.assertEqual(actual['encodings'][0]['value'],'yes')
            self.assertEqual(open_policy_project(save_policy_project(actual)),actual)


if __name__=='__main__':unittest.main(verbosity=2)
