import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from openpyxl import load_workbook

from studio.corpus import prepare, read_csv
from studio.modeling import fit_topics, load_encoder
from studio.runtime import ROOT, configure
from studio.workflow import analyze, excel_bytes, persist

configure()
FIXTURE = ROOT / 'data/examples/import_edge_cases.csv'


class CorpusTests(unittest.TestCase):
    def test_counts_raw_fields_and_stable_ids(self):
        corpus = read_csv(FIXTURE.read_bytes())
        keep = prepare(corpus, '正文')
        dedup = prepare(corpus, '正文', 'deduplicate')
        again = prepare(read_csv(FIXTURE.read_bytes()), '正文')
        self.assertEqual(keep.counts, dict(input=55, empty=3, duplicate_extra=3, duplicate_involved=6, selected=52))
        self.assertEqual(dedup.counts['selected'], 49)
        self.assertEqual(keep.fields['record_id'], '_studio_record_id')
        for item in (keep, dedup, again):
            pd.testing.assert_frame_equal(item.rows[corpus.frame.columns], corpus.frame)
            self.assertEqual(item.rows[item.fields['original_text']].tolist(), corpus.frame['正文'].tolist())
            self.assertTrue(item.rows[item.fields['record_id']].is_unique)
            self.assertEqual(keep.rows[keep.fields['record_id']].tolist(), item.rows[item.fields['record_id']].tolist())
        self.assertEqual(corpus.frame.iloc[0]['source_id'], '0001')
        self.assertEqual(corpus.frame.iloc[3]['备注'], 'NA')
        self.assertTrue(corpus.frame.iloc[-1]['正文'].startswith('  '))

    def test_other_column_and_blank_records(self):
        corpus = read_csv(b'a,b\r\n001,NA\r\n\r\n002,"x\ny"\r\n')
        self.assertEqual(len(corpus.frame), 3)
        prepared = prepare(corpus, 'b')
        self.assertEqual(prepared.counts['empty'], 1)
        self.assertEqual(corpus.frame['b'].tolist(), ['NA', '', 'x\ny'])
        self.assertEqual(prepare(corpus, 'a').rows['studio_record_id'].tolist(), prepared.rows['studio_record_id'].tolist())

    def test_encodings(self):
        text = '编号,正文\n001,中文内容\n'
        for encoding in ('utf-8-sig', 'gb18030', 'utf-16'):
            frame = read_csv(text.encode(encoding), encoding).frame
            self.assertEqual(frame.iloc[0].tolist(), ['001', '中文内容'])
        with self.assertRaises(ValueError):
            read_csv(text.encode('gb18030'))

    def test_reject_malformed_input(self):
        for raw in (b'', b'a,a\n1,2', b'a,\n1,2', b'a,b\n1,2,3', b'a,b\n"unfinished', b'x'*(10*1024**2+1)):
            with self.subTest(raw=raw[:30]), self.assertRaises(ValueError):
                read_csv(raw)
        self.assertEqual(prepare(read_csv(b'text\n'), 'text').counts['input'], 0)
        all_empty = prepare(read_csv(b'text\n""\n" "\n'), 'text')
        self.assertEqual(all_empty.counts['selected'], 0)

    def test_excel_preserves_literals_and_rejects_loss(self):
        frame = pd.DataFrame({'id':['0001'], 'text':['=1+1'], 'other':['NA'], 'newlines':['a\r\nb\rc\n']})
        book = load_workbook(io.BytesIO(excel_bytes({'rows':frame,'topics':pd.DataFrame(), 'metadata':{}})))
        self.assertEqual(book['records']['B2'].data_type, 's')
        self.assertEqual(book['records']['B2'].value, '=1+1')
        self.assertEqual(book['records']['A2'].value, '0001')
        self.assertEqual(book['records']['D2'].value, 'a\r\nb\rc\n')
        for value in ('x'*32768, 'a\x01b'):
            with self.assertRaises(ValueError):
                excel_bytes({'rows':pd.DataFrame({'text':[value]}),'topics':pd.DataFrame(),'metadata':{}})


class OfflineIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.network = patch('socket.socket.connect', side_effect=AssertionError('Network forbidden during offline test'))
        cls.network.start()
        cls.encoder = load_encoder('cpu')
        cls.corpus = read_csv(FIXTURE.read_bytes())
        cls.reports = []

    @classmethod
    def tearDownClass(cls):
        cls.network.stop()
        destination=ROOT/'outputs/csv_validation.json'
        destination.write_text(json.dumps({'offline_socket_connect_blocked':True,'runs':cls.reports},ensure_ascii=False,indent=2)+'\n')

    def test_real_keep_and_deduplicate(self):
        for policy, expected in [('keep',52),('deduplicate',49)]:
            with self.subTest(policy=policy):
                prepared = prepare(self.corpus,'正文',policy)
                result = analyze(prepared,self.encoder)
                f = prepared.fields
                rows = result['rows']
                selected = rows[f['status']].eq('modeled')
                self.assertEqual(len(rows), 55)
                self.assertEqual(selected.sum(),expected)
                segments = result['segments']
                self.assertNotIn(f['topic_id'], rows.columns)
                self.assertEqual(len(segments), expected)
                self.assertTrue(segments[f['topic_id']].notna().all())
                self.assertEqual(segments[f['document_id']].tolist(), rows.loc[selected, f['record_id']].tolist())
                self.assertEqual(result['topics']['Count'].sum(),expected)
                pd.testing.assert_frame_equal(rows[self.corpus.frame.columns], self.corpus.frame)
                self.assertEqual(rows[f['original_text']].tolist(), self.corpus.frame['正文'].tolist())
                for _, row in segments.iterrows():
                    keyword=result['topics'].set_index('Topic').loc[row[f['topic_id']], 'Keywords']
                    self.assertEqual(row[f['topic_keywords']],keyword)
                if policy == 'deduplicate':
                    duplicates=rows[rows[f['status']].eq('excluded_duplicate')]
                    indexed=rows.set_index(f['record_id'])
                    for _, row in duplicates.iterrows():
                        self.assertEqual(row[f['original_text']],indexed.loc[row[f['duplicate_of']],f['original_text']])
                path=persist(result,self.corpus,ROOT/'outputs/integration_runs')
                self.assertEqual((path/'source.csv').read_bytes(),FIXTURE.read_bytes())
                for exported in [pd.read_csv(path/'records.csv',dtype=str,keep_default_na=False),
                                 pd.read_excel(path/'results.xlsx',sheet_name='records',dtype=str,keep_default_na=False)]:
                    self.assertEqual(len(exported),55)
                    pd.testing.assert_frame_equal(exported[self.corpus.frame.columns],self.corpus.frame,check_dtype=False)
                    self.assertEqual(exported[f['record_id']].tolist(), rows[f['record_id']].tolist())
                    self.assertEqual(exported[f['original_text']].tolist(), self.corpus.frame['正文'].tolist())
                    self.assertNotIn(f['topic_id'], exported.columns)
                for exported in [pd.read_csv(path/'segments.csv', dtype=str, keep_default_na=False),
                                 pd.read_excel(path/'results.xlsx', sheet_name='segments', dtype=str, keep_default_na=False)]:
                    self.assertEqual(exported[f['topic_id']].tolist(),segments[f['topic_id']].astype('string').tolist())
                    self.assertEqual(exported[f['segment_text']].tolist(),rows.loc[selected,f['original_text']].tolist())
                meta=json.loads((path/'run.json').read_text())
                self.assertEqual(meta['duplicate_policy'],policy)
                self.reports.append({'policy':policy,'input_rows':55,'modeled_rows':expected,
                                     'export_rows':55,'source_bytes_and_fields_preserved':True,
                                     'topic_assignment_alignment':True,'metadata':meta})

    def test_guards_small_empty_and_long(self):
        with self.assertRaisesRegex(ValueError,'至少需要'):
            fit_topics(['短文本']*3,self.encoder)
        with self.assertRaisesRegex(ValueError,'空文本'):
            fit_topics(['']+self.corpus.frame['正文'].iloc[:12].tolist(),self.encoder)
        docs=self.corpus.frame['正文'].iloc[:12].tolist()
        docs[0]='中文长文本'*200
        with self.assertRaisesRegex(ValueError,'上限'):
            fit_topics(docs,self.encoder)
        with self.assertRaisesRegex(ValueError,'关键词'):
            fit_topics(['！'*n for n in range(1,13)],self.encoder)

    def test_export_failure_is_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            result={'metadata':{'run_id':'failed'},'rows':pd.DataFrame({'bad':['x'*32768]}),'topics':pd.DataFrame()}
            with patch('studio.workflow.csv_bytes', side_effect=OSError('simulated write failure')):
                with self.assertRaises(OSError):
                    persist(result,self.corpus,Path(directory))
            self.assertEqual(list(Path(directory).iterdir()),[])
            path=persist(result,self.corpus,Path(directory))
            self.assertTrue((path/'records.csv').exists())
            self.assertTrue((path/'excel_error.txt').exists())
            self.assertFalse((path/'results.xlsx').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
