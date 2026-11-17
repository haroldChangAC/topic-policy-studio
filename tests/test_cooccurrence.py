import copy
import io
import json
import unittest
from unittest.mock import patch

import pandas as pd
from PIL import Image
from streamlit.testing.v1 import AppTest

from studio.cooccurrence import (calculate, evidence, render_network, saved_tokenizer_settings,
                                 tables, validate_cooccurrence)
from studio.keywords import make_settings
from studio.plot_style import default_appearance
from studio.project import open_project, save_project
from studio.runtime import ROOT
from studio.visualization import default_visualization, find_chinese_font
from studio.workflow import csv_bytes
from test_paper_style import sample_result
from test_project import change_zip


def example():
    result = sample_result()
    f = result['metadata']['generated_fields']
    texts = ['政策 政策 佛教 佛教', '政策 佛教', '政策 经济', '单独']
    ids = [0, 0, 1, -1]
    for i, text in enumerate(texts):
        for table in ('rows', 'segments'):
            result[table].loc[i, 'text'] = text
            result[table].loc[i, f['original_text']] = text
        result['segments'].loc[i, f['segment_text']] = text
        result['segments'].loc[i, f['end_char']] = len(text)
        result['segments'].loc[i, f['topic_id']] = ids[i]
    result['topics'] = pd.DataFrame({'Topic':[-1,0,1], 'Count':[1,2,1],
                                     'Name':['-1_离群','0_佛教','1_政策'], 'Keywords':['单独','佛教、政策','政策']})
    result['visualization'] = default_visualization()
    return result


def options(**kwargs):
    return {'topic_id':None, 'include_outliers':True, 'max_keywords':20,
            'min_frequency':1, 'min_cooccurrence':1, 'max_edges':40, **kwargs}


class CooccurrenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = make_settings('chinese', True, ['经济'], ['政策','佛教'])
        cls.font, _ = find_chinese_font()

    def test_known_counts_filters_and_evidence(self):
        result = example()
        obj = calculate(result, options(), self.settings, 'user_selected')
        nodes, edges, _ = tables(obj)
        counts = nodes.set_index('keyword')['segment_frequency'].to_dict()
        self.assertEqual(counts['政策'], 3)
        self.assertEqual(counts['佛教'], 2)
        self.assertNotIn('经济', counts)
        edge = edges[(edges.source == '佛教') & (edges.target == '政策')].iloc[0]
        self.assertEqual(edge.cooccurrence_segments, 2)
        self.assertFalse((edges.source == edges.target).any())
        matches = evidence(result, obj, '佛教', '政策')
        self.assertEqual(matches['text'].tolist(), ['政策 政策 佛教 佛教','政策 佛教'])
        scoped = calculate(result, options(topic_id=0, include_outliers=False), self.settings, 'user_selected')
        self.assertEqual(scoped['segment_count'], 2)
        self.assertEqual(tables(scoped)[1].iloc[0]['cooccurrence_segments'], 2)
        not_outlier = calculate(result, options(include_outliers=False), self.settings, 'user_selected')
        self.assertEqual(not_outlier['segment_count'], 3)
        outlier = calculate(result, options(topic_id=-1), self.settings, 'user_selected')
        self.assertEqual(outlier['segment_count'], 1)
        self.assertTrue(tables(outlier)[1].empty)
        empty = copy.deepcopy(result)
        empty['segments'].loc[0, empty['metadata']['generated_fields']['segment_text']] = ''
        self.assertEqual(calculate(empty, options(), self.settings, 'user_selected')['segment_tokens'][0]['tokens'], [])
        separate = copy.deepcopy(result)
        field = separate['metadata']['generated_fields']['segment_text']
        separate['segments'][field] = ['甲词 甲词', '乙词 乙词', '', '']
        distinct = calculate(separate, options(), make_settings('chinese', False, [], ['甲词','乙词']), 'user_selected')
        separate_nodes, separate_edges, _ = tables(distinct)
        self.assertEqual(separate_nodes.set_index('keyword')['segment_frequency'].to_dict(), {'甲词':1, '乙词':1})
        self.assertTrue(separate_edges.empty)
        self.assertFalse(separate_nodes['in_graph'].any())

    def test_dictionary_and_threshold_exports(self):
        result = example(); f = result['metadata']['generated_fields']
        result['segments'].loc[0, f['segment_text']] = '外国人政策 外国人政策 佛教'
        configured = make_settings('chinese', True, ['佛教'], ['外国人政策'])
        obj = calculate(result, options(max_keywords=2, min_cooccurrence=3), configured, 'user_selected')
        self.assertIn('外国人政策', obj['segment_tokens'][0]['tokens'])
        self.assertNotIn('佛教', obj['segment_tokens'][0]['tokens'])
        nodes, edges, _ = tables(obj)
        self.assertLessEqual(nodes['in_graph'].sum(), 2)
        self.assertFalse(edges['in_graph'].any())
        parsed = pd.read_csv(io.BytesIO(csv_bytes(nodes)))
        self.assertEqual(parsed['in_graph'].sum(), nodes['in_graph'].sum())
        self.assertEqual(len(parsed), len(nodes))
        empty_edges = calculate(example(), options(topic_id=1), self.settings, 'user_selected')
        with self.assertRaisesRegex(ValueError, '不足'):
            render_network(*tables(empty_edges)[:2], 8, 5, 150, self.font, default_appearance())

    def test_roundtrip_and_plot_data(self):
        result = example(); settings = self.settings
        result['metadata']['keyword_processing'] = settings
        before = copy.deepcopy(result)
        self.assertEqual(saved_tokenizer_settings(result['metadata']), settings)
        self.assertIsNone(saved_tokenizer_settings({}))
        different = copy.deepcopy(result['metadata']); different['keyword_processing']['component_version'] = 'older'
        self.assertIsNone(saved_tokenizer_settings(different))
        obj = calculate(result, options(max_keywords=2, max_edges=1), settings, 'project_saved')
        result['visualization']['cooccurrence'] = obj
        pd.testing.assert_frame_equal(before['rows'], result['rows'])
        pd.testing.assert_frame_equal(before['segments'], result['segments'])
        pd.testing.assert_frame_equal(before['topics'], result['topics'])
        self.assertEqual(before['metadata'], result['metadata'])
        validate_cooccurrence(result, obj)
        restored = open_project(save_project(result))
        self.assertEqual(restored['visualization']['cooccurrence'], obj)
        nodes, edges, _ = tables(obj)
        self.assertEqual(int(nodes['in_graph'].sum()), 2)
        self.assertEqual(int(edges['in_graph'].sum()), 1)
        for style in ('color','mono'):
            cfg = default_appearance(); cfg['style'] = style
            from matplotlib.axes import Axes
            original_plot = Axes.plot
            lines = []
            def capture(axis, *args, **kwargs):
                lines.append((args, kwargs))
                return original_plot(axis, *args, **kwargs)
            with patch.object(Axes, 'plot', autospec=True, side_effect=capture):
                png, missing = render_network(nodes, edges, 8, 5, 150, self.font, cfg)
            self.assertEqual(len(lines), int(edges['in_graph'].sum()))
            self.assertFalse(missing)
            self.assertEqual(Image.open(io.BytesIO(png)).size, (1200,750))
            svg, _ = render_network(nodes, edges, 8, 5, 150, self.font, cfg, 'svg')
            self.assertIn(b'<svg', svg)
        corrupt = copy.deepcopy(obj); corrupt['edges'][0]['cooccurrence_segments'] += 1
        with self.assertRaisesRegex(ValueError, '计数'):
            validate_cooccurrence(result, corrupt)

    def test_old_project_page_explicit_settings_and_restore(self):
        result = example()
        app = AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
        app.radio(key='work_mode').set_value('打开已有项目').run()
        app.file_uploader(key='project_upload').set_value(('old.zip',save_project(result),'application/zip')).run(timeout=90)
        self.assertFalse(app.exception)
        keys = [str(e.key) for e in app.get('button')]
        scope = next(k for k in keys if k.startswith('co_compute_')).replace('co_compute_','')
        self.assertTrue(app.button(key='co_compute_'+scope).disabled)
        self.assertTrue(any('未保存完整分词设置' in w.value for w in app.warning))
        app.selectbox(key='co_mode_'+scope).select('中文分词').run(timeout=90)
        app.text_area(key='co_dictionary_'+scope).set_value('政策\n佛教').run(timeout=90)
        app.checkbox(key='co_confirm_'+scope).check().run(timeout=90)
        app.button(key='co_compute_'+scope).click().run(timeout=90)
        self.assertFalse(app.exception)
        self.assertFalse(any('上次计算' in w.value for w in app.warning))
        cached = app.session_state['opened_project']['result']
        obj = cached['visualization']['cooccurrence']
        self.assertEqual(obj['settings_source'], 'user_selected')
        downloads = {e.key for e in app.get('download_button')}
        self.assertTrue({'co_nodes_'+scope,'co_edges_'+scope,'co_png_'+scope,'co_svg_'+scope}.issubset(downloads))
        reopened = open_project(save_project(cached))
        self.assertEqual(reopened['visualization']['cooccurrence'], obj)
        app.number_input(key='co_min_pair_'+scope).set_value(3).run(timeout=90)
        self.assertTrue(any('上次计算' in w.value for w in app.warning))
        self.assertEqual(cached['visualization']['cooccurrence']['settings']['min_cooccurrence'], 1)

    def test_legacy_versions_open_and_save_network(self):
        from test_project import legacy_project
        raw = legacy_project(save_project(example()))
        for version in (1, 2, 3):
            def make_legacy(files):
                info = json.loads(files['project.json']); info['version'] = version
                files['project.json'] = json.dumps(info).encode()
                manifest = json.loads(files['manifest.json'])
                for name in (['visualization.json'] if version < 3 else []) + (['topic_edits.json'] if version == 1 else []):
                    files.pop(name); manifest['files'].pop(name)
                files['manifest.json'] = json.dumps(manifest).encode()
            legacy = change_zip(raw, make_legacy, True)
            old = open_project(legacy)
            self.assertIsNone(saved_tokenizer_settings(old['metadata']))
            old['visualization']['cooccurrence'] = calculate(old, options(), self.settings, 'user_selected')
            restored = open_project(save_project(old))
            self.assertEqual(restored['project_info']['version'], 5)
            self.assertEqual(restored['visualization']['cooccurrence']['segment_count'], 4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
