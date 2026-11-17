import copy
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from bertopic import BERTopic
from streamlit.testing.v1 import AppTest

from studio.corpus import prepare, read_csv
from studio.keywords import decode_word_file, make_settings, parse_word_list, preview
from studio.modeling import fit_topics, load_encoder
from studio.project import open_project, save_project
from studio.runtime import ROOT, configure
from studio.topic_names import rename
from studio.visualization import keyword_data
from studio.workflow import analyze, csv_bytes, excel_bytes

configure()


def sample_corpus():
    return read_csv((ROOT/'data/examples/keyword_terms_zh.csv').read_bytes())


class KeywordUnitTests(unittest.TestCase):
    def test_lists_preview_and_isolation(self):
        self.assertEqual(parse_word_list(' 政策 \n\n佛教\n政策\n'), ['政策', '佛教'])
        self.assertEqual(decode_word_file(b'\xef\xbb\xbfhello\n'), 'hello\n')
        for bad in [b'\xff', b'\x81']:
            with self.assertRaisesRegex(ValueError, 'UTF-8'): decode_word_file(bad)
        with self.assertRaisesRegex(ValueError, '词内不能含空白'): parse_word_list('词 条')
        text = '阿弥陀佛，称名念佛；外国人政策 政策 佛教 繁體。'
        old = preview(text, make_settings())
        configured = make_settings('chinese', True, ['政策'], ['称名念佛', '外国人政策'])
        words = preview(text, configured)
        self.assertIn('称名念佛', words); self.assertIn('外国人政策', words)
        self.assertNotIn('政策', words); self.assertIn('佛教', words)
        self.assertNotIn('，', words)
        self.assertEqual(old, preview(text, make_settings()))
        self.assertEqual(words, preview(text, configured))
        self.assertNotIn('称名念佛', preview(text, make_settings('chinese')))
        self.assertIn('政策', preview(text, make_settings('chinese', False, ['政策'])))


class KeywordOfflineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.network = patch('socket.socket.connect', side_effect=AssertionError('network forbidden'))
        cls.network.start()
        cls.encoder = load_encoder('cpu')
        cls.corpus = sample_corpus()
        cls.settings = make_settings('chinese', True, ['政策'], ['称名念佛', '外国人政策', '往生净土'])
        cls.observed = {}
        real = BERTopic.get_topic
        def spy(model, topic, *args, **kwargs):
            value = real(model, topic, *args, **kwargs)
            if isinstance(value, list) and all(isinstance(x, tuple) and len(x) == 2 for x in value):
                cls.observed[int(topic)] = [(w, float(v)) for w, v in value]
            return value
        fits = []
        def capture(*args, **kwargs):
            value = fit_topics(*args, **kwargs); fits.append(value); return value
        with patch('studio.workflow.fit_topics', side_effect=capture):
            cls.baseline = analyze(prepare(cls.corpus, '正文'), cls.encoder)
            with patch.object(BERTopic, 'get_topic', spy):
                cls.configured = analyze(prepare(cls.corpus, '正文'), cls.encoder, keyword_settings=cls.settings)
        cls.fits = fits

    @classmethod
    def tearDownClass(cls):
        cls.network.stop()
        result = {'offline_socket_blocked': True, 'sample': 'synthetic Buddhist and policy texts',
                  'documents': len(cls.corpus.frame), 'segments': len(cls.configured['segments']),
                  'baseline_keywords': cls.baseline['topics'][['Topic','Keywords']].to_dict('records'),
                  'configured_keywords': cls.configured['topics'][['Topic','Keywords']].to_dict('records'),
                  'settings': cls.settings, 'same_embeddings': bool(np.array_equal(cls.fits[0]['embeddings'], cls.fits[1]['embeddings'])),
                  'same_assignments': cls.fits[0]['topics'] == cls.fits[1]['topics']}
        (ROOT/'outputs/keyword_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')

    def test_actual_comparison_and_model_weights(self):
        a,b = self.fits
        np.testing.assert_array_equal(a['embeddings'], b['embeddings'])
        self.assertEqual(a['topics'], b['topics'])
        old = {w for pairs in a['topic_representations'].values() for w,_ in pairs if w}
        new = {w for pairs in b['topic_representations'].values() for w,_ in pairs if w}
        self.assertIn('政策', old); self.assertNotIn('政策', new)
        self.assertIn('外国人政策', new); self.assertIn('称名念佛', new)
        actual = {}
        for row in self.configured['visualization']['keyword_weights']:
            actual.setdefault(row['topic_id'], []).append((row['keyword'], row['weight']))
        self.assertEqual(actual, self.observed)
        self.assertNotEqual(self.baseline['topics']['Keywords'].tolist(), self.configured['topics']['Keywords'].tolist())

    def test_source_offsets_settings_and_project(self):
        result = copy.deepcopy(self.configured); f = result['metadata']['generated_fields']
        self.assertEqual(result['metadata']['keyword_processing'], self.settings)
        fields = [f[x] for x in ('document_id','segment_id','segment_order','start_char','end_char','segment_text','topic_id')]
        pd.testing.assert_frame_equal(self.baseline['segments'][fields], result['segments'][fields])
        for _, part in result['segments'].iterrows():
            self.assertEqual(part[f['segment_text']], part[f['original_text']][part[f['start_char']]:part[f['end_char']]])
            self.assertEqual(part['来源'], '合成测试')
        topic = int(result['topics']['Topic'].iloc[0]); rename(result, topic, '新名称')
        reopened = open_project(save_project(result))
        self.assertEqual(reopened['metadata']['keyword_processing'], self.settings)
        self.assertEqual(reopened['topic_edits'], result['topic_edits'])
        self.assertEqual(reopened['visualization'], result['visualization'])
        self.assertEqual(keyword_data(reopened, topic).to_dict('records'), keyword_data(result, topic).to_dict('records'))
        self.assertTrue(csv_bytes(reopened['segments'])); self.assertTrue(excel_bytes(reopened))
        legacy = copy.deepcopy(result); legacy['metadata'].pop('keyword_processing')
        legacy_zip = save_project(legacy)
        self.assertNotIn('keyword_processing', open_project(legacy_zip)['metadata'])
        app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=90).run()
        app.radio(key='work_mode').set_value('打开已有项目').run(timeout=90)
        app.file_uploader(key='project_upload').set_value(('old.hpts.zip', legacy_zip, 'application/zip')).run(timeout=90)
        self.assertFalse(app.exception)
        self.assertTrue(any('历史项目未保存' in item.value for item in app.info))

    def test_all_filtered_fails_without_modifying_source(self):
        docs = ['政策' + '。'*n for n in range(1,13)]
        with self.assertRaisesRegex(ValueError, '整个语料.*没有有效词'):
            fit_topics(docs, self.encoder, keyword_settings=make_settings('chinese', True, ['政策']))
        self.assertEqual(docs[0], '政策。')

    def test_one_topic_without_words_is_reported(self):
        # Stub model outputs to cover BERTopic's documented empty placeholder case.
        docs = self.corpus.frame['正文'].tolist()
        info = pd.DataFrame({'Topic':[0,1], 'Count':[24,24], 'Name':['0_佛教','1_'], 'Representation':[['佛教'],['']]})
        with patch.object(BERTopic, 'fit_transform', return_value=([0,1]*24, np.ones(48))), \
             patch.object(BERTopic, 'get_topic_info', return_value=info), \
             patch.object(BERTopic, 'get_topic', side_effect=lambda topic: [('佛教', 0.5)] if topic == 0 else [('', 0.00001)]):
            result = analyze(prepare(self.corpus, '正文'), self.encoder, keyword_settings=self.settings)
        self.assertEqual(result['metadata']['empty_keyword_topics'], [1])
        self.assertEqual(result['topics'].set_index('Topic').loc[1, 'Keywords'], '')
        self.assertEqual(result['topics'].set_index('Topic').loc[1, 'Name'], '1_无有效关键词')
        self.assertEqual(len(result['segments']), len(docs))
        self.assertTrue(keyword_data(result, 1).empty)

    def test_page_preview_settings_and_real_analysis(self):
        app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=90).run()
        app.file_uploader(key='csv_upload').set_value(('sample.csv', self.corpus.raw, 'text/csv')).run(timeout=90)
        app.radio(key='keyword_mode').set_value('中文分词').run(timeout=90)
        app.checkbox(key='stop_enabled').check().run(timeout=90)
        app.text_area(key='stop_text').set_value('政策').run(timeout=90)
        app.file_uploader(key='stop_file').set_value(('stop.txt', '以及\n'.encode(), 'text/plain')).run(timeout=90)
        app.text_area(key='dictionary_text').set_value('称名念佛\n外国人政策').run(timeout=90)
        app.file_uploader(key='dictionary_file').set_value(('terms.txt', '往生净土\n'.encode(), 'text/plain')).run(timeout=90)
        app.text_area(key='keyword_preview').set_value('称名念佛 外国人政策 政策').run(timeout=90)
        self.assertFalse(app.exception); self.assertFalse(app.error)
        app.button(key='run_analysis').click().run(timeout=90)
        self.assertFalse(app.exception); self.assertFalse(app.error)
        settings = app.session_state['analysis_result']['result']['metadata']['keyword_processing']
        self.assertEqual(settings['stopwords_applied'], ['政策','以及'])
        self.assertEqual(settings['user_dictionary'], ['称名念佛','外国人政策','往生净土'])
        self.assertTrue(app.button(key='export_prepare_new'))
        app.file_uploader(key='csv_upload').set_value(('other.csv', self.corpus.raw+b'\n', 'text/csv')).run(timeout=90)
        self.assertFalse(app.exception)
        self.assertEqual(app.text_area(key='dictionary_text').value, '')
        self.assertEqual(app.text_area(key='stop_text').value, '')
        self.assertFalse(app.checkbox(key='stop_enabled').value)
        app.text_area(key='dictionary_text').set_value('人口流失').run(timeout=90)
        app.file_uploader(key='csv_upload').set_value(None).run(timeout=90)
        app.file_uploader(key='csv_upload').set_value(('other.csv', self.corpus.raw+b'\n', 'text/csv')).run(timeout=90)
        self.assertEqual(app.text_area(key='dictionary_text').value, '')


if __name__ == '__main__': unittest.main(verbosity=2)
