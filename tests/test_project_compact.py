"""Portable compact projects retain full material and reject invalid references."""
import copy
import io
import json
import unittest
import zipfile
from unittest.mock import patch

import pandas as pd
from openpyxl import load_workbook

from studio.project import open_project, save_project
from studio.workflow import csv_bytes, excel_bytes
from test_paper_style import sample_result
from test_project import change_zip, legacy_project


def sample():
    r=sample_result(); f=r['metadata']['generated_fields']
    text='  政策\r\n第二行\t「引号」😀 & < >。\n末行  '
    r['rows'].loc[0,['text',f['original_text']]]=[text,text]
    r['segments'].loc[0,['text',f['original_text'],f['segment_text']]]=[text,text,text]
    r['segments'].loc[0,f['end_char']]=len(text)
    r['metadata']['original_columns'].append('地区/备注')
    for frame in (r['rows'],r['segments']):
        frame['地区/备注']=['首尔\n北区','A, B','特殊 "符号"','']
    # Identical text is kept as an excluded duplicate with a distinct stable ID.
    duplicate=r['rows'].iloc[0].copy();duplicate[f['record_id']]='excluded-duplicate'
    duplicate[f['document_id']]='excluded-duplicate';duplicate[f['status']]='excluded_duplicate'
    r['rows']=pd.concat([r['rows'],pd.DataFrame([duplicate])],ignore_index=True)
    r['metadata']['counts']['input']=5
    return r


class CompactProjectTests(unittest.TestCase):
    def test_roundtrip_duplicates_fields_and_exports(self):
        original=sample();f=original['metadata']['generated_fields'];raw=save_project(original)
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            info=json.loads(z.read('project.json'));stored=json.loads(z.read('segments.json'))
            self.assertEqual(info['version'],5)
            self.assertFalse(set(original['metadata']['original_columns']) & set(stored['columns']))
            self.assertNotIn(f['original_text'],stored['columns']);self.assertNotIn(f['segment_text'],stored['columns'])
            self.assertEqual(stored['column_order'],original['segments'].columns.tolist())
        reopened=open_project(raw)
        for name in ('rows','segments','topics'):
            self.assertEqual(json.loads(original[name].to_json(orient='split',force_ascii=False,double_precision=15))['data'],
                             json.loads(reopened[name].to_json(orient='split',force_ascii=False,double_precision=15))['data'])
            self.assertEqual(original[name].columns.tolist(),reopened[name].columns.tolist())
        self.assertEqual(len(reopened['rows']),5);self.assertEqual(len(reopened['segments']),4)
        self.assertEqual(reopened['rows'].iloc[0]['text'],reopened['rows'].iloc[4]['text'])
        self.assertEqual(reopened['rows'].iloc[4][f['status']],'excluded_duplicate')
        self.assertNotIn('excluded-duplicate',reopened['segments'][f['document_id']].tolist())
        csv=pd.read_csv(io.BytesIO(csv_bytes(reopened['segments'])),dtype=str,keep_default_na=False)
        self.assertEqual(csv.iloc[0][f['segment_text']],original['segments'].iloc[0][f['segment_text']])
        wb=load_workbook(io.BytesIO(excel_bytes(reopened)),read_only=True)
        table=list(wb['segments'].values);headers=list(table[0])
        self.assertEqual(table[1][headers.index(f['segment_text'])],original['segments'].iloc[0][f['segment_text']])
        self.assertEqual(table[1][headers.index('地区/备注')],'首尔\n北区')

    def test_legacy_versions_restore_and_save_new(self):
        raw=save_project(sample())
        for version in (1,2,3,4):
            legacy=legacy_project(raw, 4 if version==4 else 3)
            if version<3:
                def downgrade(files):
                    info=json.loads(files['project.json']);info['version']=version
                    files['project.json']=json.dumps(info).encode()
                    manifest=json.loads(files['manifest.json'])
                    files.pop('visualization.json');manifest['files'].pop('visualization.json')
                    if version==1:files.pop('topic_edits.json');manifest['files'].pop('topic_edits.json')
                    files['manifest.json']=json.dumps(manifest).encode()
                legacy=change_zip(legacy,downgrade,True)
            old=open_project(legacy);self.assertEqual(old['project_info']['version'],version)
            new=open_project(save_project(old));self.assertEqual(new['project_info']['version'],5)
            for name in ('rows','segments','topics'):
                pd.testing.assert_frame_equal(old[name],new[name])

    def test_reject_invalid_reference_position_order_and_schema(self):
        original=sample();f=original['metadata']['generated_fields'];raw=save_project(original)
        def mutate(field,value):
            def change(files):
                payload=json.loads(files['segments.json']);payload['data'][0][payload['columns'].index(field)]=value
                files['segments.json']=json.dumps(payload,ensure_ascii=False).encode()
            return change_zip(raw,change,True)
        for field,value in ((f['document_id'],'unknown'),(f['document_id'],'excluded-duplicate'),
                            (f['start_char'],-1),(f['start_char'],1),
                            (f['end_char'],999999),(f['segment_order'],2),
                            (f['start_char'],True),(f['segment_id'],'segment-1'),(f['topic_id'],999)):
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):open_project(mutate(field,value))
        def bad_order(files):
            payload=json.loads(files['segments.json']);payload['column_order'].pop()
            files['segments.json']=json.dumps(payload,ensure_ascii=False).encode()
        with self.assertRaisesRegex(ValueError,'字段或顺序'):open_project(change_zip(raw,bad_order,True))
        bad=copy.deepcopy(original);bad['segments'].loc[0,f['original_text']]='篡改'
        with self.assertRaisesRegex(ValueError,'不一致'):save_project(bad)
        with patch('studio.project.MAX_RECONSTRUCTED',1),self.assertRaisesRegex(ValueError,'恢复后的片段字段'):
            open_project(raw)
        with patch('studio.project.MAX_RECONSTRUCTED',1),self.assertRaisesRegex(ValueError,'无法保存'):
            save_project(original)


if __name__=='__main__':unittest.main()
