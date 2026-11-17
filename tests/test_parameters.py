import copy
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from studio.diagnostics import parameter_record, summarize
from studio.modeling import default_clustering_settings, validate_clustering_settings
from studio.project import open_project, save_project
from studio.runtime import ROOT
from studio.topic_names import materialize
from test_paper_style import sample_result


class ParameterDiagnosticsTests(unittest.TestCase):
    def test_defaults_and_validation(self):
        defaults = default_clustering_settings()
        self.assertEqual(defaults, {'n_neighbors': 8, 'min_dist': 0.0, 'min_cluster_size': 4, 'min_samples': 2})
        self.assertEqual(validate_clustering_settings(None, 10), defaults)
        valid = dict(defaults, n_neighbors=9, min_cluster_size=10, min_samples=9, min_dist=1.0)
        self.assertEqual(validate_clustering_settings(valid, 10), valid)
        for key, value in [('n_neighbors', 10), ('n_neighbors', 1), ('min_dist', -0.1),
                           ('min_dist', float('nan')), ('min_cluster_size', 11),
                           ('min_cluster_size', 1), ('min_samples', 0), ('min_samples', 10)]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_clustering_settings(dict(defaults, **{key: value}), 10)
        with self.assertRaises(ValueError):
            validate_clustering_settings({'n_neighbors': 8}, 10)

    def test_diagnostics_and_legacy_parameters(self):
        result, _ = materialize(sample_result())
        f = result['metadata']['generated_fields']
        result['segments'].loc[0, f['topic_id']] = -1
        outlier = result['topics'].iloc[[0]].copy()
        outlier['Topic'] = -1
        result['topics'] = __import__('pandas').concat([result['topics'], outlier], ignore_index=True)
        stats, topics = summarize(result)
        self.assertEqual((stats['original_documents'], stats['valid_documents'], stats['segments']), (4, 4, 4))
        self.assertEqual((stats['topics'], stats['outliers'], stats['outlier_share']), (3, 1, .25))
        self.assertEqual(topics['Count'].sum(), 4)
        self.assertAlmostEqual(topics['Segment_Share'].sum(), 1)
        self.assertEqual(parameter_record({})['UMAP 邻居数'], '历史记录未保存')

    def test_parameters_and_diagnostics_roundtrip(self):
        result = sample_result()
        requested = dict(default_clustering_settings(), min_cluster_size=3)
        result['metadata']['clustering_settings_requested'] = requested
        result['metadata']['parameters'] = {'random_state': 42, 'umap': {'n_neighbors': 8, 'min_dist': 0.0},
                                            'hdbscan': {'min_cluster_size': 3, 'min_samples': 2}}
        restored = open_project(save_project(result))
        self.assertEqual(parameter_record(restored['metadata']), parameter_record(result['metadata']))
        self.assertEqual(summarize(materialize(restored)[0])[0], summarize(materialize(result)[0])[0])
        old = copy.deepcopy(result)
        del old['metadata']['clustering_settings_requested']
        del old['metadata']['parameters']['umap']['n_neighbors']
        reopened = open_project(save_project(old))
        self.assertEqual(parameter_record(reopened['metadata'])['UMAP 邻居数'], '历史记录未保存')

    def test_page_invalid_and_new_analysis(self):
        with patch('socket.socket.connect', side_effect=AssertionError('network')):
            app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=90).run()
            app.file_uploader(key='csv_upload').set_value(('edge.csv', (ROOT/'data/examples/import_edge_cases.csv').read_bytes(), 'text/csv')).run(timeout=90)
            self.assertEqual(app.number_input(key='cluster_min_size').value, 4)
            self.assertEqual(app.number_input(key='cluster_neighbors').value, 8)
            app.number_input(key='cluster_neighbors').set_value(52).run(timeout=90)
            self.assertTrue(app.button(key='run_analysis').disabled)
            self.assertTrue(any('邻居数' in x.value for x in app.error))
            app.number_input(key='cluster_neighbors').set_value(8).run(timeout=90)
            app.number_input(key='cluster_min_size').set_value(5).run(timeout=90)
            app.button(key='run_analysis').click().run(timeout=90)
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            result = app.session_state['analysis_result']['result']
            self.assertEqual(result['metadata']['parameters']['hdbscan']['min_cluster_size'], 5)
            self.assertEqual(result['metadata']['clustering_settings_requested']['min_cluster_size'], 5)
            metrics = {m.label: m.value for m in app.metric}
            self.assertEqual(metrics['建模片段数'], str(len(result['segments'])))
            self.assertIn('离群片段占比', metrics)
            app.number_input(key='cluster_min_samples').set_value(3).run(timeout=90)
            self.assertNotIn('analysis_result', app.session_state)


if __name__ == '__main__':
    unittest.main(verbosity=2)
