import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from openpyxl import Workbook
from streamlit.testing.v1 import AppTest
from studio.corpus import read_csv, read_txt, read_xlsx, xlsx_sheets, prepare, missing_text_formulae
from studio.modeling import load_encoder
from studio.project import open_project, save_project
from studio.runtime import ROOT, configure
from studio.segmentation import plan_segments
from studio.workflow import analyze, csv_bytes, excel_bytes, persist

configure()
XLSX=ROOT/'data/examples/import_formats.xlsx'
TXT=ROOT/'data/examples/import_long.txt'

class FormatTests(unittest.TestCase):
    def test_xlsx_sheets_rows_and_formula_policy(self):
        raw=XLSX.read_bytes()
        self.assertEqual(xlsx_sheets(raw),['摘要','政策','公式示例','空工作表'])
        a=read_xlsx(raw,'摘要');b=read_xlsx(raw,'政策')
        self.assertEqual((len(a.frame),len(b.frame)),(50,16))
        self.assertEqual(a.frame.iloc[0]['编号'],'0001')
        self.assertEqual(a.frame.iloc[0]['年份'],'2020')
        self.assertEqual(a.frame.iloc[0]['地区'],'北京')
        self.assertEqual(a.frame.iloc[1]['来源'],'=1+1')
        self.assertEqual(a.source_details['formula_cells'][0]['coordinate'],'E3')
        self.assertIn('\r\n第二行原文。',a.frame.iloc[0]['正文'])
        self.assertEqual(a.frame.iloc[0]['正文'],a.frame.iloc[48]['正文'])
        ka=prepare(a,'正文');kb=prepare(b,'正文')
        self.assertEqual((ka.counts['input'],ka.counts['empty'],ka.counts['duplicate_extra'],ka.counts['selected']),(50,1,1,49))
        self.assertEqual(prepare(a,'正文','deduplicate').counts['selected'],48)
        self.assertEqual(ka.rows.iloc[0][ka.fields['source_record']],2)
        self.assertNotEqual(ka.rows.iloc[0][ka.fields['record_id']],kb.rows.iloc[0][kb.fields['record_id']])
        self.assertEqual(ka.rows[ka.fields['record_id']].tolist(),prepare(a,'正文','deduplicate').rows[ka.fields['record_id']].tolist())
        c=read_xlsx(raw,'公式示例');missing=missing_text_formulae(c,'正文')
        self.assertEqual(len(c.source_details['formula_cells']),3)
        self.assertEqual(missing[0]['coordinate'],'B2')
        self.assertEqual(c.frame.iloc[0]['正文'],'="无缓存的公式正文"')
        self.assertEqual(c.frame.iloc[1]['正文'],'有缓存的公式正文')
        pc=prepare(c,'正文');self.assertEqual(pc.counts['formula_uncached'],1)
        self.assertEqual(pc.rows.iloc[0][pc.fields['status']],'blocked_formula_unavailable')
        with self.assertRaisesRegex(ValueError,'公式没有可用缓存'):
            analyze(pc,None)
        with self.assertRaisesRegex(ValueError,'空'):
            read_xlsx(raw,'空工作表')
        with self.assertRaisesRegex(ValueError,'不存在'):
            read_xlsx(raw,'不存在')

    def test_invalid_and_empty(self):
        for bad in [b'',(ROOT/'tests/fixtures/corrupt.xlsx').read_bytes()]:
            with self.assertRaises(ValueError):xlsx_sheets(bad)
        for bad in [b'',b' \r\n', (ROOT/'tests/fixtures/invalid_utf8.txt').read_bytes()]:
            with self.assertRaises(ValueError):read_txt(bad)
        with self.assertRaises(ValueError):read_csv(b'')
        w=Workbook();w.active.append(['正文','正文']);w.active.append(['a','b']);buff=io.BytesIO();w.save(buff)
        with self.assertRaisesRegex(ValueError,'不重复'):read_xlsx(buff.getvalue(),w.active.title)
        w=Workbook();w.active.append(['正文']);buff=io.BytesIO();w.save(buff)
        with self.assertRaisesRegex(ValueError,'没有数据行'):read_xlsx(buff.getvalue(),w.active.title)
        w=Workbook();w.active.append(['年份','数量']);w.active.append([2024,10]);buff=io.BytesIO();w.save(buff)
        self.assertEqual(read_xlsx(buff.getvalue(),w.active.title).source_details['detected_text_columns'],[])

    def test_txt_lossless_and_encoding(self):
        raw=TXT.read_bytes();corpus=read_txt(raw)
        self.assertEqual(corpus.frame.iloc[0]['text'],raw.decode('utf-8'))
        self.assertEqual(corpus.raw,raw)
        self.assertIn('\r\n\r\n',corpus.frame.iloc[0]['text'])
        self.assertEqual(len(corpus.frame),1)
        same_bytes=b'text\nhello world\n'
        self.assertNotEqual(prepare(read_csv(same_bytes),'text').rows['studio_record_id'][0],
                            prepare(read_txt(same_bytes),'text').rows['studio_record_id'][0])
        for encoding in ('utf-8-sig','gb18030','utf-16'):
            text='首行。\r\n\r\n  第二行！  '
            self.assertEqual(read_txt(text.encode(encoding),encoding).frame.iloc[0]['text'],text)

class OfflineFormatsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.socket=patch('socket.socket.connect',side_effect=AssertionError('network forbidden'))
        cls.socket.start();cls.encoder=load_encoder('cpu');cls.results=[]
    @classmethod
    def tearDownClass(cls):
        cls.socket.stop()
        (ROOT/'outputs/import_formats_validation.json').write_text(json.dumps({'offline_socket_blocked':True,'runs':cls.results},ensure_ascii=False,indent=2)+'\n')

    def verify_roundtrip(self,corpus,column,policy,mode):
        prepared=prepare(corpus,column,policy);result=analyze(prepared,self.encoder,split_mode=mode);f=prepared.fields
        docs,segs=result['rows'],result['segments']
        self.assertEqual(result['metadata']['source_format'],corpus.source_format)
        self.assertEqual(result['metadata']['source_details'],corpus.source_details)
        pd.testing.assert_frame_equal(docs[corpus.frame.columns],corpus.frame)
        for doc_id,parts in segs.groupby(f['document_id'],sort=False):
            original=docs.set_index(f['document_id']).loc[doc_id]
            for _,part in parts.iterrows():
                self.assertEqual(part[f['segment_text']],original[f['original_text']][int(part[f['start_char']]):int(part[f['end_char']])])
                for col in corpus.frame.columns:self.assertEqual(part[col],original[col])
            self.assertEqual(''.join(parts.sort_values(f['segment_order'])[f['segment_text']]),original[f['original_text']])
        with tempfile.TemporaryDirectory() as tmp:
            dest=persist(result,corpus,Path(tmp));self.assertEqual((dest/f'source.{corpus.source_format}').read_bytes(),corpus.raw)
            reopened=open_project(save_project(result));self.assertEqual(reopened['metadata']['source_format'],corpus.source_format)
            self.assertEqual(reopened['metadata']['source_details'],corpus.source_details)
            self.assertEqual(reopened['rows'][f['record_id']].tolist(),docs[f['record_id']].tolist())
            self.assertEqual(reopened['segments'][f['segment_id']].tolist(),segs[f['segment_id']].tolist())
            self.assertEqual(reopened['segments'][f['segment_text']].tolist(),segs[f['segment_text']].tolist())
            out=pd.read_csv(io.BytesIO(csv_bytes(reopened['segments'])),dtype=str,keep_default_na=False)
            self.assertEqual(out[f['segment_text']].tolist(),segs[f['segment_text']].tolist())
            xlsx=pd.read_excel(io.BytesIO(excel_bytes(reopened)),sheet_name='records',dtype=str,keep_default_na=False)
            self.assertEqual(xlsx[f['original_text']].tolist(),docs[f['original_text']].tolist())
        self.results.append({'format':corpus.source_format,'input_documents':len(docs),'segments':len(segs),
                             'duplicate_policy':policy,'source_details':corpus.source_details,'project_format':reopened['project_info']['version']})
        return result

    def test_real_xlsx_analysis_project_export(self):
        result=self.verify_roundtrip(read_xlsx(XLSX.read_bytes(),'摘要'),'正文','deduplicate','auto')
        self.assertEqual(result['metadata']['source_details']['sheet'],'摘要')
        self.assertEqual(result['metadata']['source_details']['formula_cells'][0]['coordinate'],'E3')
        self.assertEqual(len(result['segments']),48)

    def test_real_txt_analysis_project_export(self):
        corpus=read_txt(TXT.read_bytes());prepared=prepare(corpus,'text')
        self.assertTrue(plan_segments(prepared,self.encoder,'stop').blocked)
        result=self.verify_roundtrip(corpus,'text','keep','auto')
        self.assertEqual(len(result['rows']),1)
        self.assertGreaterEqual(len(result['segments']),10)
        self.assertEqual(result['metadata']['source_details']['encoding'],'utf-8-sig')

    def test_xlsx_and_txt_page(self):
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
        w=Workbook();w.active.append(['年份','数量']);w.active.append([2024,10]);buff=io.BytesIO();w.save(buff)
        app.file_uploader(key='csv_upload').set_value(('numbers.xlsx',buff.getvalue(),'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')).run(timeout=90)
        self.assertTrue(any('未发现非空文本单元格' in warning.value for warning in app.warning))
        app.file_uploader(key='csv_upload').set_value(('import_formats.xlsx',XLSX.read_bytes(),'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')).run(timeout=90)
        self.assertFalse(app.exception);self.assertEqual(app.selectbox(key='xlsx_sheet').value,'摘要')
        self.assertEqual(app.selectbox(key='text_column').value,'正文')
        app.selectbox(key='xlsx_sheet').select('公式示例').run(timeout=90)
        self.assertTrue(any('公式' in e.value and '缓存' in e.value for e in app.error))
        app.selectbox(key='xlsx_sheet').select('政策').run(timeout=90)
        self.assertFalse(app.exception);self.assertFalse(app.error)
        app.button(key='run_analysis').click().run(timeout=90)
        self.assertFalse(app.exception);self.assertFalse(app.error)
        self.assertEqual(app.session_state['analysis_result']['result']['metadata']['source_details']['sheet'],'政策')
        app.file_uploader(key='csv_upload').set_value(('import_long.txt',TXT.read_bytes(),'text/plain')).run(timeout=90)
        self.assertFalse(app.exception);self.assertTrue(app.button(key='run_analysis').disabled)
        app.radio(key='split_mode').set_value('自动切分超长文本').run(timeout=90)
        self.assertFalse(app.button(key='run_analysis').disabled)
        app.button(key='run_analysis').click().run(timeout=90)
        self.assertFalse(app.exception);self.assertFalse(app.error)
        result=app.session_state['analysis_result']['result']
        self.assertEqual(result['metadata']['source_format'],'txt');self.assertEqual(len(result['rows']),1)
        self.assertTrue(app.button(key='export_prepare_new'))

if __name__=='__main__':unittest.main(verbosity=2)
