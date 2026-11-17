import io
import json
import random
import unittest
from unittest.mock import patch

import pandas as pd
import numpy as np
from studio.corpus import read_csv, prepare
from studio.modeling import load_encoder
from studio.segmentation import TokenBudget, split_text, plan_segments
from studio.workflow import analyze, persist
from studio.runtime import ROOT


class SegmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.offline = patch('socket.socket.connect', side_effect=AssertionError('Network forbidden'))
        cls.offline.start()
        cls.encoder = load_encoder()
        cls.budget = TokenBudget(cls.encoder)
        cls.reports=[]

    @classmethod
    def tearDownClass(cls):
        cls.offline.stop()
        (ROOT/'outputs/segmentation_validation.json').write_text(json.dumps(
            {'budget':cls.budget.metadata(),'offline_socket_connect_blocked':True,'runs':cls.reports},ensure_ascii=False,indent=2)+'\n')

    def assert_coverage(self,text):
        spans=split_text(text,self.budget)
        pieces=[text[a:b] for a,b in spans]
        self.assertEqual(''.join(pieces),text)
        self.assertEqual(spans[0][0],0)
        self.assertEqual(spans[-1][1],len(text))
        self.assertEqual(sum(b-a for a,b in spans),len(text))
        for i,(a,b) in enumerate(spans):
            if i: self.assertEqual(a,spans[i-1][1])
            self.assertLessEqual(self.budget.count(text[a:b]),self.budget.limit)
            self.assertTrue(text[a:b].strip())
        self.budget.verify_model_inputs(pieces)
        return spans

    def test_short_long_chinese_unpunctuated_and_paragraphs(self):
        texts=['  短文本。\r\n', '唐诗意象与文学历史。'*200,
               '无标点环境保护政策研究'*300,
               '\t\r\n首段。\r\n\r\n'+('第二段主题研究。'*100)+'\n\n第三段文字。  ',
               ('中文🙂 e\u0301 English 日本語 한국어。\n'*100)]
        for text in texts:
            with self.subTest(length=len(text)):
                spans=self.assert_coverage(text)
                self.assertEqual(len(spans)==1,self.budget.count(text)<=self.budget.limit)
        self.assertEqual(split_text('',self.budget),[])
        self.assertEqual(split_text(' \r\n\t',self.budget),[])

    def test_exact_and_near_limit_including_special_tokens(self):
        found={self.budget.count('中'*n):'中'*n for n in range(1,256)}
        for length in (self.budget.limit-1,self.budget.limit,self.budget.limit+1):
            text=found[length]
            spans=self.assert_coverage(text)
            self.assertEqual(len(spans)==1,length<=self.budget.limit)
            self.assertEqual(len(self.encoder.tokenizer.encode(text,add_special_tokens=False))+self.budget.special_tokens,length)

    def test_prefers_paragraph_then_sentence_boundary(self):
        unit='中'*75
        for separator in ('\r\n\r\n','。'):
            text=unit+separator+unit
            spans=self.assert_coverage(text)
            self.assertEqual(spans,[(0,len(unit+separator)),(len(unit+separator),len(text))])

    def test_varied_unicode_and_whitespace_coverage(self):
        rng=random.Random(42)
        for _ in range(15):
            text='  '+''.join(rng.choice(['文字研究','。','\n\n','\r\n','🙂','研究\t','test ','无标点']) for _ in range(220))+'  '
            self.assert_coverage(text)

    def test_stop_mode_duplicates_short_structure_and_stability(self):
        corpus=read_csv((ROOT/'data/examples/segmentation_zh.csv').read_bytes())
        for policy in ('keep','deduplicate'):
            prepared=prepare(corpus,'正文',policy);f=prepared.fields
            stopped=plan_segments(prepared,self.encoder,'stop')
            self.assertTrue(stopped.blocked)
            self.assertEqual(stopped.counts['segments'],0)
            with self.assertRaisesRegex(ValueError,'上限'):
                analyze(prepared,self.encoder,split_mode='stop')
            plan=plan_segments(prepared,self.encoder,'auto')
            again=plan_segments(prepared,self.encoder,'auto')
            self.assertEqual(plan.segments[f['segment_id']].tolist(),again.segments[f['segment_id']].tolist())
            self.assertTrue(plan.segments[f['segment_id']].is_unique)
            selected_ids=set(prepared.rows.loc[prepared.selected,f['record_id']])
            self.assertEqual(set(plan.segments[f['document_id']]),selected_ids)
            short=plan.documents[(plan.documents[f['status']]=='pending') & ~plan.documents[f['is_long']]]
            self.assertTrue(short[f['segment_count']].eq(1).all())
        keep=plan_segments(prepare(corpus,'正文'),self.encoder,'auto')
        dedup=plan_segments(prepare(corpus,'正文','deduplicate'),self.encoder,'auto')
        self.assertTrue(set(dedup.segments[f['segment_id']]).issubset(set(keep.segments[f['segment_id']])))

    def test_real_split_fit_and_csv_excel_roundtrip(self):
        raw=(ROOT/'data/examples/segmentation_zh.csv').read_bytes()
        corpus=read_csv(raw)
        for policy in ('keep','deduplicate'):
            prepared=prepare(corpus,'正文',policy);f=prepared.fields
            result=analyze(prepared,self.encoder,split_mode='auto')
            docs,segs=result['rows'],result['segments']
            self.assertNotIn(f['topic_id'],docs.columns)
            self.assertEqual(len(docs),60)
            self.assertGreater(len(segs),len(docs))
            self.assertTrue(segs[f['token_count']].le(self.budget.limit).all())
            self.assertTrue(segs[f['topic_id']].notna().all())
            self.assertEqual(result['topics']['Count'].sum(),len(segs))
            pd.testing.assert_frame_equal(docs[corpus.frame.columns],corpus.frame)
            for doc_id,parts in segs.groupby(f['document_id'],sort=False):
                parts=parts.sort_values(f['segment_order'])
                original=docs.set_index(f['document_id']).loc[doc_id]
                self.assertEqual(''.join(parts[f['segment_text']]),original[f['original_text']])
                self.assertEqual(parts[f['segment_order']].tolist(),list(range(1,len(parts)+1)))
                for _,part in parts.iterrows():
                    self.assertEqual(part[f['segment_text']],original[f['original_text']][int(part[f['start_char']]):int(part[f['end_char']])])
                    self.assertEqual(part[corpus.frame.columns].tolist(),original[corpus.frame.columns].tolist())
            path=persist(result,corpus,ROOT/'outputs/segmentation_runs')
            self.assertEqual((path/'source.csv').read_bytes(),raw)
            for filename,sheet,expected in [('records.csv','records',docs),('segments.csv','segments',segs)]:
                for exported in [pd.read_csv(path/filename,dtype=str,keep_default_na=False),
                                 pd.read_excel(path/'results.xlsx',sheet_name=sheet,dtype=str,keep_default_na=False)]:
                    exact_columns = [column for column in expected.columns if column != f['membership_strength']]
                    pd.testing.assert_frame_equal(exported[exact_columns],expected[exact_columns].astype('string').fillna(''),check_dtype=False)
                    if f['membership_strength'] in expected:
                        np.testing.assert_allclose(pd.to_numeric(exported[f['membership_strength']],errors='coerce'),
                                                   expected[f['membership_strength']].to_numpy(dtype=float,na_value=np.nan),
                                                   rtol=1e-14,atol=1e-15,equal_nan=True)
            metadata=json.loads((path/'run.json').read_text())
            self.assertEqual(metadata['analysis_unit'],'text_segment')
            self.assertEqual(metadata['document_topic_aggregation'],'none')
            self.reports.append({'policy':policy,'counts':metadata['counts'],'max_segment_tokens':int(segs[f['token_count']].max()),
                                 'full_coverage_and_roundtrip':True,'source_sha256':corpus.sha256,'run_id':metadata['run_id']})


if __name__=='__main__':
    unittest.main(verbosity=2)
