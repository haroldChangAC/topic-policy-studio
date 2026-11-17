from html import escape
import io
import json
import hashlib
import unittest
import zipfile
from unittest.mock import patch
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest
from studio.corpus import read_csv, prepare
from studio.modeling import load_encoder
from studio.project import save_project, open_project
from studio.runtime import ROOT, configure
from studio.workflow import analyze, csv_bytes, excel_bytes

configure()


def change_zip(raw, change, rehash=False):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        files={n:z.read(n) for n in z.namelist()}
    change(files)
    if rehash:
        manifest=json.loads(files['manifest.json'])
        for name in manifest['files']:
            if name in files:
                manifest['files'][name]={'size':len(files[name]),'sha256':hashlib.sha256(files[name]).hexdigest()}
        files['manifest.json']=json.dumps(manifest).encode()
    result=io.BytesIO()
    with zipfile.ZipFile(result,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,data in files.items():z.writestr(name,data)
    return result.getvalue()


def legacy_project(raw, version=3):
    """Build a genuine uncompact v3/v4 fixture from a v5 result."""
    if version not in (3, 4):
        raise ValueError('需要版本 3 或 4。')
    result=open_project(raw)
    def change(files):
        info=json.loads(files['project.json']);info['version']=version
        files['project.json']=json.dumps(info,ensure_ascii=False).encode()
        files['segments.json']=result['segments'].to_json(orient='split',force_ascii=False,double_precision=15).encode()
    return change_zip(raw,change,True)


class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.socket=patch('socket.socket.connect',side_effect=AssertionError('network prohibited'))
        cls.socket.start()
        encoder=load_encoder('cpu')
        corpus=read_csv((ROOT/'data/examples/segmentation_zh.csv').read_bytes())
        cls.result=analyze(prepare(corpus,'正文','deduplicate'),encoder,split_mode='auto')
        cls.zip=save_project(cls.result)

    @classmethod
    def tearDownClass(cls):cls.socket.stop()

    def test_real_offline_roundtrip_and_reexport(self):
        with patch('studio.modeling.load_encoder',side_effect=AssertionError('must not load model')):
            result=open_project(self.zip)
        self.assertEqual(result['metadata'],self.result['metadata'])
        self.assertFalse(result['project_info']['model_included'])
        for key in ['rows','segments','topics']:
            before=self.result[key].to_json(orient='split',force_ascii=False,double_precision=15)
            after=result[key].to_json(orient='split',force_ascii=False,double_precision=15)
            self.assertEqual(json.loads(before)['data'],json.loads(after)['data'])
        f=result['metadata']['generated_fields']
        self.assertEqual(len(result['rows']),60)
        self.assertGreater(len(result['segments']),53)
        self.assertEqual(result['rows'][f['status']].value_counts()['excluded_duplicate'],4)
        self.assertTrue(any('\r\n' in text for text in result['rows'][f['original_text']]))
        data=pd.read_csv(io.BytesIO(csv_bytes(result['segments'])),dtype=str,keep_default_na=False)
        self.assertEqual(data[f['segment_id']].tolist(),result['segments'][f['segment_id']].tolist())
        excel=excel_bytes(result)
        reread=pd.read_excel(io.BytesIO(excel),sheet_name='records',dtype=str,keep_default_na=False)
        self.assertEqual(reread[f['original_text']].tolist(),result['rows'][f['original_text']].tolist())
        (ROOT/'outputs/project_validation.json').write_text(json.dumps({'passed':True,'offline_socket_blocked':True,'documents':60,'segments':len(result['segments']),'duplicates_excluded':4,'format_version':result['project_info']['version'],'real_model_fit':True,'reexport_csv_excel':True},ensure_ascii=False,indent=2)+'\n')

    def test_long_field_survives_project_even_when_excel_cannot(self):
        result={key:(value.copy(deep=True) if isinstance(value,pd.DataFrame) else value) for key,value in self.result.items()}
        f=result['metadata']['generated_fields']
        doc_id=result['rows'].iloc[0][f['document_id']]
        result['rows'].loc[result['rows'][f['document_id']].eq(doc_id),'备注']='长文本'*12000
        result['segments'].loc[result['segments'][f['document_id']].eq(doc_id),'备注']='长文本'*12000
        restored=open_project(save_project(result))
        self.assertEqual(restored['rows'].iloc[0]['备注'],'长文本'*12000)
        self.assertIn(('长文本'*12000).encode(),csv_bytes(restored['rows']))
        with self.assertRaises(ValueError):excel_bytes(restored)

    def test_rename_save_restore_export_and_history(self):
        from studio.topic_names import rename, materialize, original_label
        import copy
        from openpyxl import load_workbook
        original=copy.deepcopy(self.result)
        topic=int(original['topics']['Topic'].iloc[0])
        label=original_label(original,topic)
        self.assertTrue(rename(original,topic,'  东亚研究 / East Asia <&>  '))
        self.assertFalse(rename(original,topic,'东亚研究 / East Asia <&>'))
        self.assertTrue(rename(original,topic,'Second name & 主题'))
        self.assertTrue(rename(original,topic,''))
        self.assertFalse(rename(original,topic,''))
        self.assertTrue(rename(original,topic,'再命名 ✓'))
        for invalid in ('X'*101,'换行\n不允许','x\x00'):
            with self.assertRaises(ValueError):rename(original,topic,invalid)
        self.assertEqual([e['operation'] for e in original['topic_edits']['history']],['add','change','clear','add'])
        self.assertEqual(original['topic_edits']['history'][0]['before'],label)
        reopened=open_project(save_project(original))
        self.assertEqual(reopened['topic_edits'],original['topic_edits'])
        self.assertEqual(reopened['metadata'],self.result['metadata'])
        for key in ('rows','segments','topics'):
            a=json.loads(reopened[key].to_json(orient='split',force_ascii=False,double_precision=15))
            b=json.loads(self.result[key].to_json(orient='split',force_ascii=False,double_precision=15))
            self.assertEqual(a['data'],b['data'])
        exported,names=materialize(reopened)
        self.assertEqual(exported['topics'].set_index('Topic').loc[topic,'Display_Name'],'再命名 ✓')
        f=reopened['metadata']['generated_fields'];matching=exported['segments'][exported['segments'][f['topic_id']].eq(topic)]
        self.assertTrue(matching[names['display_name']].eq('再命名 ✓').all())
        self.assertTrue(matching[names['original_label']].eq(label).all())
        self.assertIn('再命名 ✓'.encode(),csv_bytes(exported['segments']))
        wb=load_workbook(io.BytesIO(excel_bytes(exported)))
        self.assertIn('topic_name_history',wb.sheetnames)
        self.assertTrue(rename(reopened,topic,'从恢复项目再次编辑'))
        reopened2=open_project(save_project(reopened))
        self.assertEqual(reopened2['topic_edits']['labels'][str(topic)],'从恢复项目再次编辑')
        self.assertEqual(len(reopened2['topic_edits']['history']),5)

    def test_outlier_and_same_as_original_noop(self):
        from studio.topic_names import rename, materialize, original_label
        import copy
        result=copy.deepcopy(self.result)
        topic=int(result['topics']['Topic'].iloc[0])
        f=result['metadata']['generated_fields']
        result['topics'].loc[result['topics']['Topic'].eq(topic),'Topic']=-1
        result['segments'].loc[result['segments'][f['topic_id']].eq(topic),f['topic_id']]=-1
        original=original_label(result,-1)
        self.assertFalse(rename(result,-1,original))
        self.assertTrue(rename(result,-1,'异常主题 / Outlier & 特殊'))
        self.assertTrue(rename(result,-1,original))
        self.assertEqual([e['operation'] for e in result['topic_edits']['history']],['add','clear'])
        view,names=materialize(result)
        self.assertTrue(view['topics'].set_index('Topic').loc[-1,'Is_Outlier'])
        self.assertTrue(view['segments'].loc[view['segments'][f['topic_id']].eq(-1),names['is_outlier']].all())

    def test_restored_editor_ui(self):
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
        app.radio(key='work_mode').set_value('打开已有项目').run()
        app.file_uploader(key='project_upload').set_value(('project.zip',self.zip,'application/zip')).run(timeout=60)
        topic=int(app.selectbox(key='edit_topic_restored').value)
        app.text_input(key=f'name_input_restored_{topic}').set_value('人文主题 <Test & 1>')
        next(b for b in app.button if b.label=='保存名称').click().run(timeout=60)
        result=app.session_state['opened_project']['result']
        self.assertEqual(result['topic_edits']['labels'][str(topic)],'人文主题 <Test & 1>')
        self.assertEqual(len(result['topic_edits']['history']),1)
        self.assertTrue(app.button(key='export_prepare_restored'))
        next(b for b in app.button if b.label=='保存名称').click().run(timeout=60)
        self.assertEqual(len(result['topic_edits']['history']),1)
        next(b for b in app.button if b.label=='清除自定义名称').click().run(timeout=60)
        self.assertEqual([e['operation'] for e in result['topic_edits']['history']],['add','clear'])

    def test_export_preserves_conflicting_original_field(self):
        from studio.topic_names import rename, materialize
        import copy
        result=copy.deepcopy(self.result)
        f=result['metadata']['generated_fields']
        collision=f['topic_id']+'_display_name'
        result['metadata']['original_columns'].append(collision)
        result['rows'][collision]='原始用户字段'
        result['segments'][collision]='原始用户字段'
        topic=int(result['topics']['Topic'].iloc[0]);rename(result,topic,'显示名称')
        saved=open_project(save_project(result))
        exported,names=materialize(saved)
        self.assertNotEqual(names['display_name'],collision)
        self.assertTrue(exported['segments'][collision].eq('原始用户字段').all())
        self.assertTrue(exported['segments'].loc[exported['segments'][f['topic_id']].eq(topic),names['display_name']].eq('显示名称').all())

    def test_legacy_v1_open_and_save_v2(self):
        import copy
        from studio.topic_names import rename, materialize
        source=legacy_project(save_project(self.result))
        def legacy(files):
            files.pop('topic_edits.json')
            files.pop('visualization.json')
            info=json.loads(files['project.json']);info['version']=1
            files['project.json']=json.dumps(info,ensure_ascii=False).encode()
            manifest=json.loads(files['manifest.json']);manifest['files'].pop('topic_edits.json');manifest['files'].pop('visualization.json')
            files['manifest.json']=json.dumps(manifest).encode()
        v1=change_zip(source,legacy,True)
        result=open_project(v1)
        self.assertEqual(result['topic_edits'],{'labels':{},'history':[]})
        view,_=materialize(result)
        self.assertEqual(view['topics']['Display_Name'].tolist(),view['topics']['Original_Label'].tolist())
        topic=int(result['topics']['Topic'].iloc[0]); rename(result,topic,'Legacy 名称')
        saved=save_project(result); result2=open_project(saved)
        self.assertEqual(result2['project_info']['version'],5)
        self.assertEqual(result2['topic_edits']['labels'][str(topic)],'Legacy 名称')
        self.assertEqual(len(result2['segments']),len(result['segments']))

    def test_malicious_or_inconsistent_edit_history(self):
        def alter(files):
            files['topic_edits.json']=json.dumps({'labels':{'0':'fake'},'history':[]}).encode()
        with self.assertRaisesRegex(ValueError,'不一致'):
            open_project(change_zip(self.zip,alter,True))

    def test_bad_archives(self):
        with zipfile.ZipFile(io.BytesIO(self.zip)) as archive:
            first=archive.infolist()[0]
            corrupt=bytearray(self.zip)
            position=first.header_offset+30+len(first.filename.encode())+len(first.extra)+max(0,first.compress_size//2)
            corrupt[position] ^= 0xff
        cases=[b'badzip', bytes(corrupt), change_zip(self.zip,lambda f:f.pop('records.json')),
               change_zip(self.zip,lambda f:f.__setitem__('../evil',b'evil')),
               change_zip(self.zip,lambda f:f.__setitem__('records.json',b'bad')),
               change_zip(self.zip,lambda f:f.__setitem__('project.json',json.dumps({'format':'humanities-policy-studio','version':99}).encode()),True),
               change_zip(self.zip,lambda f:f.__setitem__('run.json',b'{bad'),True)]
        for item in cases:
            with self.subTest(size=len(item)),self.assertRaises(ValueError):open_project(item)
        def bad_offset(files):
            t=json.loads(files['segments.json']);meta=json.loads(files['run.json']);f=meta['generated_fields']
            t['data'][0][t['columns'].index(f['start_char'])]=999999
            files['segments.json']=json.dumps(t,ensure_ascii=False).encode()
        with self.assertRaisesRegex(ValueError,'片段位置'):
            open_project(change_zip(self.zip,bad_offset,True))

    def test_size_limits(self):
        for name,value in [('MAX_ARCHIVE',len(self.zip)-1),('MAX_ENTRY',8),('MAX_TOTAL',8)]:
            with self.subTest(limit=name), patch('studio.project.'+name,value), self.assertRaises(ValueError):
                open_project(self.zip)

    def test_streamlit_project_open_and_errors(self):
        with patch('studio.modeling.load_encoder',side_effect=AssertionError('must not load model')):
            app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
            app.radio(key='work_mode').set_value('打开已有项目').run()
            invalid = [b'broken', change_zip(self.zip,lambda f:f.pop('records.json')),
                       change_zip(self.zip,lambda f:f.__setitem__('../evil',b'evil')),
                       change_zip(self.zip,lambda f:f.__setitem__('records.json',b'bad')),
                       change_zip(self.zip,lambda f:f.__setitem__('project.json',json.dumps({'format':'humanities-policy-studio','version':99}).encode()),True)]
            for item in invalid:
                app.file_uploader(key='project_upload').set_value(('bad.zip',item,'application/zip')).run()
                self.assertTrue(app.error)
                self.assertIn('无法打开项目',app.error[0].value)
            app.file_uploader(key='project_upload').set_value(('ok.zip',self.zip,'application/zip')).run(timeout=60)
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual([m.value for m in app.metric][:2],['60',str(len(self.result['segments']))])
            keys=[e.key for e in app.get('download_button')]
            self.assertTrue({'restored_records','restored_topics','restored_run'}.issubset(keys))
            self.assertTrue(app.button(key='export_prepare_restored'))
            f=self.result['metadata']['generated_fields']
            segment=self.result['segments'].iloc[0]
            self.assertTrue(any(f"<span>{escape(segment[f['segment_text']])}</span>" in element.value for element in app.html))


if __name__=='__main__':unittest.main(verbosity=2)
