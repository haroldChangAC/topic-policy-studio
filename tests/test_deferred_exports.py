"""Large result exports are explicit, reusable and invalidated on changes."""
import copy
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest
from studio.corpus import read_csv
from studio.deferred_exports import create_export, fingerprint
from studio.project import open_project, save_project
from studio.runtime import ROOT
from studio.topic_groups import create, update
from studio.topic_names import materialize, rename
from studio.workflow import csv_bytes, persist
from test_paper_style import sample_result


def example():
    result=sample_result();create(result,'人工组',[0,1]);return result


class DeferredExportTests(unittest.TestCase):
    def test_fingerprint_tracks_all_result_fields(self):
        original=example();base=fingerprint(original)
        variants=[]
        changed=copy.deepcopy(original);changed['rows'].loc[0,'text']='修改原文';variants.append(changed)
        changed=copy.deepcopy(original);changed['segments'].loc[0,'studio_topic_id']=2;variants.append(changed)
        changed=copy.deepcopy(original);rename(changed,0,'改名');variants.append(changed)
        changed=copy.deepcopy(original);gid=changed['visualization']['topic_groups']['groups'][0]['group_id'];update(changed,gid,'改组名',[0,1]);variants.append(changed)
        changed=copy.deepcopy(original);changed['visualization']['settings']['dpi']=300;variants.append(changed)
        changed=copy.deepcopy(original);changed['metadata']['model_id']='other';variants.append(changed)
        self.assertEqual(base,fingerprint(original))
        for item in variants:self.assertNotEqual(base,fingerprint(item))
        original['topics']['extra_score']=[1e-16]*len(original['topics'])
        fine=copy.deepcopy(original);fine['topics'].loc[0,'extra_score']=2e-16
        self.assertNotEqual(fingerprint(original),fingerprint(fine))

    def test_filename_digest_covers_entire_file(self):
        first=b'A'*(4*1024*1024);second=first[:1048576]+b'B'+first[1048577:]
        with patch('studio.topic_names.materialize',return_value=({'segments':pd.DataFrame()},{})),\
             patch('studio.workflow.csv_bytes',side_effect=[first,second]):
            a=create_export({},'segments');b=create_export({},'segments')
        self.assertEqual(len(a['data']),len(b['data']));self.assertNotEqual(a['filename'],b['filename'])
        self.assertNotEqual(a['content_sha256'],b['content_sha256'])

    def test_minimal_persist_does_not_generate_exports(self):
        result=sample_result();corpus=read_csv(b'text\nabc\n')
        with tempfile.TemporaryDirectory() as tmp,\
             patch('studio.workflow.csv_bytes',side_effect=AssertionError('eager CSV')),\
             patch('studio.workflow.excel_bytes',side_effect=AssertionError('eager Excel')):
            dest=persist(result,corpus,Path(tmp),include_exports=False)
            self.assertEqual({x.name for x in dest.iterdir()},{'source.csv','run.json'})
            self.assertEqual((dest/'source.csv').read_bytes(),corpus.raw)

    def test_ui_prepare_reuse_invalidate_switch_release_and_failure(self):
        original=example();raw=save_project(original)
        changed=copy.deepcopy(original);rename(changed,0,'另一个项目');raw2=save_project(changed)
        with patch('socket.socket.connect',side_effect=AssertionError('network forbidden')):
            app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=100).run()
            app.radio(key='work_mode').set_value('打开已有项目').run()
            with patch('studio.deferred_exports.create_export',wraps=create_export) as create_spy:
                app.file_uploader(key='project_upload').set_value(('first.zip',raw,'application/zip')).run(timeout=100)
                self.assertFalse(app.exception);self.assertEqual(create_spy.call_count,0)
                self.assertNotIn('prepared_export_restored',app.session_state)
                self.assertNotIn('export_download_restored',[e.key for e in app.get('download_button')])
                app.selectbox(key='export_kind_restored').select('segments').run(timeout=100)
                app.button(key='export_prepare_restored').click().run(timeout=100)
                self.assertFalse(app.exception);self.assertEqual(create_spy.call_count,1)
                cached=app.session_state['prepared_export_restored'];saved=app.session_state['opened_project']['result']
                self.assertEqual(cached['data'],csv_bytes(materialize(saved)[0]['segments']))
                self.assertIn('export_download_restored',[e.key for e in app.get('download_button')])
                app.selectbox(key='restored_topic').select(1).run(timeout=100)
                self.assertEqual(create_spy.call_count,1);self.assertIn('prepared_export_restored',app.session_state)
                app.button(key='export_prepare_restored').click().run(timeout=100)
                self.assertEqual(create_spy.call_count,1)
                # A real theme-editor action invalidates the prepared bytes.
                app.selectbox(key='edit_topic_restored').select(0).run(timeout=100)
                app.text_input(key='name_input_restored_0').set_value('新名称')
                next(b for b in app.button if b.label=='保存名称').click().run(timeout=100)
                self.assertNotIn('prepared_export_restored',app.session_state)
                self.assertTrue(any('已失效' in x.value for x in app.warning))
                app.button(key='export_prepare_restored').click().run(timeout=100)
                self.assertEqual(create_spy.call_count,2)
                self.assertIn('新名称'.encode(),app.session_state['prepared_export_restored']['data'])
                gid=saved['visualization']['topic_groups']['groups'][0]['group_id']
                update(saved,gid,'改组',[0,1]);app.run(timeout=100)
                self.assertNotIn('prepared_export_restored',app.session_state)
                app.selectbox(key='export_kind_restored').select('group_trace').run(timeout=100)
                app.button(key='export_prepare_restored').click().run(timeout=100)
                self.assertEqual(create_spy.call_count,3)
                self.assertIn('改组'.encode(),app.session_state['prepared_export_restored']['data'])
                scope='restored_'+app.session_state['opened_project']['digest']
                app.checkbox(key='chart_outliers_'+scope).uncheck().run(timeout=100)
                self.assertNotIn('prepared_export_restored',app.session_state)
                app.selectbox(key='export_kind_restored').select('project').run(timeout=100)
                app.button(key='export_prepare_restored').click().run(timeout=100)
                self.assertEqual(create_spy.call_count,4)
                reopened=open_project(app.session_state['prepared_export_restored']['data'])
                self.assertEqual(reopened['visualization']['topic_groups']['groups'][0]['name'],'改组')
                self.assertEqual(reopened['topic_edits']['labels']['0'],'新名称')
                # Replacing the uploaded project clears even a previously prepared file.
                app.file_uploader(key='project_upload').set_value(('second.zip',raw2,'application/zip')).run(timeout=100)
                self.assertNotIn('prepared_export_restored',app.session_state)
                self.assertNotIn('export_download_restored',[e.key for e in app.get('download_button')])
            with patch('studio.deferred_exports.create_export',side_effect=ValueError('模拟失败')):
                app.button(key='export_prepare_restored').click().run(timeout=100)
                self.assertTrue(any('模拟失败' in e.value for e in app.error))
                self.assertNotIn('prepared_export_restored',app.session_state)
            app.button(key='export_prepare_restored').click().run(timeout=100)
            self.assertFalse(app.error);self.assertIn('prepared_export_restored',app.session_state)
            app.button(key='export_release_restored').click().run(timeout=100)
            self.assertNotIn('prepared_export_restored',app.session_state)
            self.assertNotIn('export_download_restored',[e.key for e in app.get('download_button')])


if __name__=='__main__':unittest.main()
