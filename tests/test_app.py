from html import escape
import json
import unittest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from studio.runtime import ROOT


class AppWorkflowTests(unittest.TestCase):
    def test_upload_preview_run_filter_download_and_invalidate(self):
        with patch('socket.socket.connect', side_effect=AssertionError('Offline UI test attempted network')):
            app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
            self.assertFalse(app.exception)
            app.file_uploader(key='csv_upload').set_value(('edge.csv',(ROOT/'data/examples/import_edge_cases.csv').read_bytes(),'text/csv')).run()
            self.assertFalse(app.exception)
            self.assertEqual(app.selectbox(key='text_column').value,'正文')
            self.assertEqual(app.radio(key='duplicate_policy').value,'保留全部（默认）')
            self.assertEqual([m.value for m in app.metric][:4],['55','3','3','52'])
            app.button(key='run_analysis').click().run(timeout=90)
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            result=app.session_state['analysis_result']['result']
            self.assertEqual(result['metadata']['counts']['selected'],52)
            download_keys=[element.key for element in app.get('download_button')]
            self.assertTrue({'records_download','metadata_download'}.issubset(download_keys))
            self.assertNotIn('export_download_new',download_keys)
            app.button(key='export_prepare_new').click().run(timeout=90)
            self.assertIn('export_download_new',[e.key for e in app.get('download_button')])
            selector=app.selectbox(key='topic_'+result['metadata']['run_id'])
            topic=int(result['topics']['Topic'].iloc[0])
            selector.select(topic).run()
            f=result['metadata']['generated_fields']
            shown=next(element.value for element in app.dataframe if f['topic_id'] in element.value.columns)
            self.assertTrue(shown[f['topic_id']].eq(topic).all())
            self.assertGreater(len(shown),0)
            app.radio(key='duplicate_policy').set_value('去重，仅首条参与建模').run()
            self.assertEqual([m.value for m in app.metric][:4],['55','3','3','49'])
            self.assertNotIn('records_download',[e.key for e in app.get('download_button')])
            app.selectbox(key='text_column').select('年份').run()
            self.assertTrue(app.button(key='run_analysis').disabled)
            self.assertNotIn('records_download',[e.key for e in app.get('download_button')])
            app.file_uploader(key='csv_upload').set_value(('bad.csv',b'a,a\n1,2','text/csv')).run()
            self.assertFalse(app.exception)
            self.assertTrue(app.error)
            report={'passed':True,'real_file_uploader':True,'real_bertopic_fit':True,'offline_socket_connect_blocked':True,
                    'default_policy':'keep','preview_counts':[55,3,3,52],'deduplicate_selected':49,
                    'topic_filter':True,'export_buttons_present':True,'stale_result_invalidated':True,
                    'malformed_upload_handled':True,'browser_click_download_tested':False}
            (ROOT/'outputs/csv_ui_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')

    def test_auto_split_trace_and_switch_back_to_stop(self):
        with patch('socket.socket.connect', side_effect=AssertionError('Offline UI test attempted network')):
            app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
            app.file_uploader(key='csv_upload').set_value(('long.csv',(ROOT/'data/examples/segmentation_zh.csv').read_bytes(),'text/csv')).run(timeout=90)
            self.assertFalse(app.exception)
            self.assertEqual(app.radio(key='split_mode').value,'超长时停止（默认）')
            self.assertTrue(app.button(key='run_analysis').disabled)
            metrics={m.label:m.value for m in app.metric}
            self.assertEqual(metrics['原始有效文档数'],'57')
            self.assertEqual(metrics['超长文档数'],'4')
            self.assertEqual(metrics['参与建模的片段数'],'0')
            app.radio(key='split_mode').set_value('自动切分超长文本').run(timeout=90)
            self.assertFalse(app.button(key='run_analysis').disabled)
            app.button(key='run_analysis').click().run(timeout=90)
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            result=app.session_state['analysis_result']['result']
            f=result['metadata']['generated_fields'];segs=result['segments'];docs=result['rows']
            self.assertGreater(len(segs),len(docs))
            app.selectbox(key='export_kind_new').select('segments').run(timeout=90)
            app.button(key='export_prepare_new').click().run(timeout=90)
            self.assertIn('export_download_new',[x.key for x in app.get('download_button')])
            selected=segs[segs[f['segment_order']]>1].iloc[0]
            key='segment_'+result['metadata']['run_id']+'_全部片段'
            app.selectbox(key=key).select(selected[f['segment_id']]).run()
            displayed = [x.value for x in app.html if 'data-studio-source' in x.value]
            displayed = [html.replace("&#13;", "\r") for html in displayed]
            self.assertTrue(any(f"<span>{escape(selected[f['segment_text']])}</span>" in html for html in displayed))
            self.assertTrue(any(f"<span>{escape(selected[f['original_text']])}</span>" in html for html in displayed))
            app.radio(key='split_mode').set_value('超长时停止（默认）').run()
            self.assertTrue(app.button(key='run_analysis').disabled)
            self.assertNotIn('export_download_new',[x.key for x in app.get('download_button')])
            app.radio(key='split_mode').set_value('自动切分超长文本').run()
            app.radio(key='duplicate_policy').set_value('去重，仅首条参与建模').run()
            metrics={m.label:m.value for m in app.metric}
            self.assertEqual(metrics['原始有效文档数'],'53')
            self.assertEqual(metrics['超长文档数'],'3')
            (ROOT/'outputs/segmentation_ui_validation.json').write_text(json.dumps({
                'passed':True,'real_file_upload':True,'real_auto_split_fit':True,
                'stop_mode_blocks_long_documents':True,'raw_text_trace_view':True,
                'mode_change_invalidates_result':True,'duplicate_policy_preserved':True,
                'offline_socket_connect_blocked':True,'manual_browser_click_tested':False,
                'counts':result['metadata']['counts']},ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    unittest.main(verbosity=2)
