import copy
import contextlib
import io
import json
import unittest
import zipfile
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from studio.policy import append_policies, confirm_indicators, edit_indicator, record_encoding, revise_policy, score
from studio.corpus import read_txt
from studio.policy_assist import (TEMPLATE_VERSION, generate, request_ollama, review, review_batch, status,
                                  validate_assist)
from studio.policy_project import open_policy_project, save_policy_project, tables
from studio.runtime import ROOT
from test_policy import make_project, make_scheme
from test_policy_transfer import theme_project
from test_project import change_zip
from studio.policy_transfer import apply_transfer, plan_transfer
from studio.policy_charts import chart_settings
import hashlib


def response(value='yes',evidence='政策证据',rationale='原文依据'):
    return json.dumps({'status':value,'rationale':rationale,'evidence':evidence},ensure_ascii=False)


def fake(raw):
    return lambda _url,_model,_prompt: raw


class PolicyAssistTests(unittest.TestCase):
    def setUp(self):
        self.p=make_project();self.children=make_scheme(self.p);self.pid=self.p['policies'][0]['policy_id']
        self.text=self.p['policies'][0]['original_text']+' 原文再见。';revise_policy(self.p,self.pid,self.text)
        self.iid=self.children[0]

    def make(self, raw, iid=None, start=0, end=None):
        return generate(self.p,self.pid,iid or self.iid,'mock-local',start,len(self.text) if end is None else end,requester=fake(raw))

    def test_generation_does_not_change_formal_codes_and_locations(self):
        before=copy.deepcopy(self.p['encodings']);scores=score(self.p)
        good=self.make(response());self.assertEqual(good['match_status'],'unique')
        self.assertEqual(self.text[good['matches'][0]['start']:good['matches'][0]['end']],'政策证据')
        missing=self.make(response(evidence='不存在的引用'));self.assertEqual(missing['match_status'],'missing')
        multiple=self.make(response(evidence='文'));self.assertEqual(multiple['match_status'],'multiple')
        malformed=self.make('不是 JSON');self.assertEqual(malformed['result'],'failed')
        broken=generate(self.p,self.pid,self.iid,'mock-local',0,len(self.text),requester=lambda *a:(_ for _ in ()).throw(ConnectionError('private detail')))
        self.assertEqual(broken['result'],'failed');self.assertNotIn('private detail',broken['error'])
        partial=self.make(response('no',''),start=0,end=3)
        self.assertFalse(partial['full_text_sent']);self.assertEqual((partial['model_status'],partial['suggested_status']),('no','pending'))
        self.assertEqual(self.p['encodings'],before);self.assertEqual(score(self.p),scores);validate_assist(self.p)
        with self.assertRaisesRegex(ValueError,'唯一'):review(self.p,missing['suggestion_id'],'accept')
        with self.assertRaisesRegex(ValueError,'唯一'):review(self.p,multiple['suggestion_id'],'accept')
        position=multiple['matches'][1];review(self.p,multiple['suggestion_id'],'accept',start=position['start'],end=position['end'])
        self.assertEqual(self.p['encodings'][-1]['start'],position['start'])

    def test_accept_modify_reject_and_existing_protection(self):
        original=self.make(response());record_encoding(self.p,self.pid,self.iid,'no',rationale='原人工判断')
        old=copy.deepcopy(self.p['encodings'])
        with self.assertRaisesRegex(ValueError,'已有正式编码'):review(self.p,original['suggestion_id'],'accept')
        self.assertEqual(self.p['encodings'],old)
        result=review(self.p,original['suggestion_id'],'accept',replace=True,coder='研究者')
        self.assertEqual(result['previous_event_id'],old[0]['event_id']);self.assertEqual(self.p['encodings'][-1]['value'],'yes')
        self.assertEqual(self.p['encodings'][-1]['evidence'],'政策证据')
        bad=self.make(response(evidence='错证据'),iid=self.children[1]);start=self.text.index('政策证据')
        review(self.p,bad['suggestion_id'],'modify','yes','政策证据',start,start+4,'人工校正','研究者')
        self.assertEqual(self.p['encodings'][-1]['rationale'],'人工校正')
        rejected=self.make(response(),iid=self.children[2]);n=len(self.p['encodings'])
        review(self.p,rejected['suggestion_id'],'reject',coder='研究者');self.assertEqual(len(self.p['encodings']),n)
        not_applicable=self.make(response('pending',''),iid=self.children[3])
        review(self.p,not_applicable['suggestion_id'],'modify','not_applicable',rationale='人工判断不适用')
        self.assertEqual([r['operation'] for r in self.p['ai_reviews']],['accept','modify','reject','modify'])
        validate_assist(self.p);self.assertEqual(open_policy_project(save_policy_project(self.p)),self.p)

    def test_partial_no_requires_manual_modification_and_batch_atomic(self):
        no=self.make(response('no',''),end=3);review(self.p,no['suggestion_id'],'accept')
        self.assertEqual(self.p['encodings'][-1]['value'],'pending');self.assertIsNone(score(self.p)[1][0]['raw_pmc'])
        another=self.make(response(),iid=self.children[1]);ambiguous=self.make(response(evidence='文'),iid=self.children[2])
        before=copy.deepcopy(self.p)
        with self.assertRaisesRegex(ValueError,'唯一'):review_batch(self.p,[another['suggestion_id'],ambiguous['suggestion_id']])
        self.assertEqual(self.p,before)
        pending=self.make(response('pending','',rationale='需人工查看'),iid=self.children[3])
        review_batch(self.p,[another['suggestion_id'],pending['suggestion_id']],coder='研究者')
        self.assertEqual([e['value'] for e in self.p['encodings']],['pending','yes','pending']);validate_assist(self.p)
        first=self.make(response(),iid=self.children[4]);second=self.make(response(),iid=self.children[4])
        with self.assertRaisesRegex(ValueError,'只能选择一条'):review_batch(self.p,[first['suggestion_id'],second['suggestion_id']])

    def test_stale_versions_roundtrip_and_secret_exclusion(self):
        good=self.make(response());self.assertEqual(good['template_version'],TEMPLATE_VERSION)
        raw=save_policy_project(self.p)
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            self.assertEqual(json.loads(z.read('project.json'))['version'],5)
            self.assertIn('ai_assist.json',z.namelist());self.assertNotIn(b'api_secret_ABC',b''.join(z.read(n) for n in z.namelist()))
        restored=open_policy_project(raw);self.assertEqual(restored,self.p)
        self.assertEqual(len(tables(restored)['ai_suggestions']),1)
        revise_policy(restored,self.pid,self.text+'补充');self.assertIn('待重新核查',status(restored,good))
        with self.assertRaisesRegex(ValueError,'变化'):review(restored,good['suggestion_id'],'accept')
        self.assertEqual(open_policy_project(save_policy_project(restored)),restored)
        with self.assertRaisesRegex(ValueError,'本机'):
            request_ollama('http://user:api_secret_ABC@127.0.0.1:11434','model','提示')
        with self.assertRaisesRegex(ValueError,'本机'):
            request_ollama('https://external.example:443','model','提示')
        other=open_policy_project(raw);edit_indicator(other,self.children[0],'修改指标','新说明');confirm_indicators(other)
        self.assertIn('待重新核查',status(other,good))
        with self.assertRaisesRegex(ValueError,'变化'):review(other,good['suggestion_id'],'accept')

    def test_local_http_request_and_failure_do_not_record_credentials(self):
        class Reply:
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def read(self,_size):return json.dumps({'done':True,'done_reason':'stop','model':'model-local:latest','response':response()},ensure_ascii=False).encode()
        class Opener:
            def open(self,request,timeout):
                self.request=request;self.timeout=timeout;return Reply()
        opener=Opener()
        with patch('studio.policy_assist.urllib.request.build_opener',return_value=opener):
            self.assertEqual(request_ollama('http://127.0.0.1:11434','model-local','内容'),(response(),'model-local:latest'))
        request=json.loads(opener.request.data);self.assertEqual(opener.request.full_url,'http://127.0.0.1:11434/api/generate')
        self.assertEqual((request['stream'],request['format'],request['options']['temperature']),(False,'json',0))
        class Cutoff(Reply):
            def read(self,_size):return json.dumps({'done':True,'done_reason':'length','model':'model-local','response':response()}).encode()
        class CutoffOpener:
            def open(self,*args,**kwargs):return Cutoff()
        with patch('studio.policy_assist.urllib.request.build_opener',return_value=CutoffOpener()):
            with self.assertRaisesRegex(RuntimeError,'不完整'):request_ollama('http://127.0.0.1:11434','model-local','内容')
        secret='api_secret_ABC'
        output=io.StringIO()
        with contextlib.redirect_stderr(output), contextlib.redirect_stdout(output):
            generate(self.p,self.pid,self.iid,'mock-local',0,len(self.text),base_url='http://ignored:11434',
                     requester=lambda *a:(_ for _ in ()).throw(ConnectionError(secret)))
        self.assertNotIn(secret,output.getvalue())
        raw=save_policy_project(self.p)
        with zipfile.ZipFile(io.BytesIO(raw)) as z:self.assertNotIn(secret.encode(),b''.join(z.read(n) for n in z.namelist()))
        self.assertNotIn(secret,json.dumps(tables(self.p)['ai_suggestions'].to_dict(),ensure_ascii=False))

    def test_tampered_review_link_is_rejected(self):
        s=self.make(response());review(self.p,s['suggestion_id'],'accept')
        def corrupt(files):
            data=json.loads(files['ai_assist.json']);data['reviews'][0]['final_event_id']='invented'
            files['ai_assist.json']=json.dumps(data,ensure_ascii=False).encode()
        with self.assertRaisesRegex(ValueError,'正式编码'):open_policy_project(change_zip(save_policy_project(self.p),corrupt,True))

    def test_policy_versions_1_through_4_upgrade_when_suggestions_saved(self):
        one=make_project();make_scheme(one);two=copy.deepcopy(one)
        append_policies(two,read_txt('第三份政策正文'.encode()),'text',{},'third.txt')
        three=copy.deepcopy(one);chart_settings(three)
        theme,raw=theme_project();four=apply_transfer(plan_transfer(theme,hashlib.sha256(raw).hexdigest(),['doc-0'],{'title':'标题'}));make_scheme(four)
        for expected,p in enumerate((one,two,three,four),1):
            old=save_policy_project(p)
            with zipfile.ZipFile(io.BytesIO(old)) as z:self.assertEqual(json.loads(z.read('project.json'))['version'],expected)
            p=open_policy_project(old);pid=p['policies'][0]['policy_id'];iid=next(i['id'] for i in p['versions'][-1]['indicators'] if i['level']==2)
            text=p['policies'][0]['original_text'];generate(p,pid,iid,'mock-local',0,len(text),requester=fake(response('pending','')))
            new=save_policy_project(p)
            with zipfile.ZipFile(io.BytesIO(new)) as z:self.assertEqual(json.loads(z.read('project.json'))['version'],5)
            self.assertEqual(open_policy_project(new),p)

    def test_page_generate_and_human_review_with_mock(self):
        with patch('studio.policy_assist.request_ollama',return_value=response()):
            app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
            app.radio(key='work_mode').set_value('基础政策评价').run()
            app.radio(key='pmc_mode').set_value('打开政策评价项目').run()
            app.file_uploader(key='pmc_project_upload').set_value(('sample.pmc.zip',save_policy_project(self.p),'application/zip')).run(timeout=90)
            self.assertFalse(app.exception);app.text_input(key='ai_model').set_value('mock-local').run()
            app.button(key='ai_generate').click().run(timeout=90)
            actual=app.session_state['pmc_project'];self.assertEqual(len(actual['ai_suggestions']),1);self.assertFalse(actual['encodings'])
            sid=actual['ai_suggestions'][0]['suggestion_id']
            app.button(key='ai_review_submit_'+sid).click().run(timeout=90)
            self.assertFalse(app.exception);self.assertEqual(actual['encodings'][0]['value'],'yes')
            self.assertEqual(actual['ai_reviews'][0]['operation'],'accept')
            self.assertEqual(open_policy_project(save_policy_project(actual)),actual)


if __name__=='__main__':unittest.main(verbosity=2)
