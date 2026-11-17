"""Regression coverage for shared immutable document fields in v5 projects."""
import io
import json
import unittest
from unittest.mock import patch

import pandas as pd
from openpyxl import load_workbook
from scripts.memory_trial import build
from studio.annual import calculate
from studio.grouped_charts import source,evidence
from studio.annual_heatmap import ALL
from studio.policy_transfer import documents
from studio.project import open_project,save_project
from studio.topic_groups import mapping,summary,trace
from studio.workflow import csv_bytes,excel_bytes


class SharedDocumentFieldsTests(unittest.TestCase):
    def test_restore_shares_strings_and_retains_values(self):
        original=build(3,3000,120,1200);raw=save_project(original);loaded=open_project(raw)
        f=loaded['metadata']['generated_fields'];segs=loaded['segments'];docs=loaded['rows']
        for field in ('正文','备注',f['original_text']):
            self.assertEqual(segs[field].dtype,object)
            for doc in docs[f['document_id']]:
                values=segs.loc[segs[f['document_id']].eq(doc),field].tolist()
                self.assertEqual(len({id(x) for x in values}),1)
        detached=segs.copy(deep=True)
        original_second=detached.iloc[1]['正文']
        detached.loc[detached.index[0],'正文']='单行修改'
        self.assertEqual(detached.iloc[1]['正文'],original_second)
        self.assertEqual(docs.iloc[0]['正文'],original_second)
        for table in ('rows','segments','topics'):
            before=json.loads(original[table].to_json(orient='split',force_ascii=False,double_precision=15))
            after=json.loads(loaded[table].to_json(orient='split',force_ascii=False,double_precision=15))
            self.assertEqual(before['columns'],after['columns']);self.assertEqual(before['data'],after['data'])
        self.assertEqual(csv_bytes(original['segments']),csv_bytes(loaded['segments']))
        settings=loaded['visualization']['annual_panel']
        pd.testing.assert_frame_equal(calculate(original,settings)[1],calculate(loaded,settings)[1])
        d,p,parts,members=source(loaded,settings);gid=next(x for x in members if x.startswith('group-'))
        matched,selected=evidence(loaded,d,parts,members,ALL,2020,gid)
        lookup=docs.set_index(f['document_id'])
        for part in selected.to_dict('records'):
            text=lookup.loc[part[f['document_id']],f['original_text']]
            self.assertEqual(text[part[f['start_char']]:part[f['end_char']]],part[f['segment_text']])
        transferred=documents(loaded)
        self.assertEqual([d['original_text'] for d in transferred],docs[f['original_text']].tolist())
        book=load_workbook(io.BytesIO(excel_bytes(loaded)),read_only=True)
        sheet=book['segments'];rows=sheet.values;headers=list(next(rows));first=next(rows)
        self.assertEqual(first[headers.index('正文')],segs.iloc[0]['正文']);book.close()
        again=open_project(save_project(loaded))
        self.assertEqual(again['visualization']['topic_groups'],loaded['visualization']['topic_groups'])
        self.assertEqual(again['segments'][f['segment_text']].tolist(),segs[f['segment_text']].tolist())

    def test_group_summary_does_not_construct_full_trace(self):
        result=build(3,3000,120,1200);full_trace=trace(result);expected={}
        for vid,frame in full_trace.groupby('view_id'):
            expected[vid]=(len(frame),frame.document_id.nunique())
        with patch('studio.topic_groups.trace',side_effect=AssertionError('summary must not expand original fields')):
            compact=summary(result)
        self.assertEqual({r.view_id:(r.segment_count,r.document_count) for r in compact.itertuples()},expected)


if __name__=='__main__':unittest.main()
