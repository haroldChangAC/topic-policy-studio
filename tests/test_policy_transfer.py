import copy
import hashlib
import io
import json
import unittest
import zipfile

import pandas as pd
from streamlit.testing.v1 import AppTest

from studio.policy import coding_matrix, record_encoding, score
from studio.policy_project import open_policy_project, save_policy_project, tables
from studio.policy_transfer import apply_transfer, documents, plan_transfer
from studio.project import open_project, save_project
from studio.runtime import ROOT
from studio.topic_names import rename
from test_paper_style import sample_result
from test_policy_append import ready_project
from test_project import change_zip, legacy_project


def theme_project():
    result=sample_result(); f=result['metadata']['generated_fields']
    result['metadata'].update({'text_column':'text','source_sha256':'a'*64})
    result['metadata']['original_columns']=['text','标题','年份','地区','来源']
    for frame in (result['rows'],result['segments']):
        for col in ('标题','年份','地区','来源'):
            frame[col]=frame[f['document_id']].map(lambda d:{'标题':'政策'+d,'年份':'2020','地区':'首尔','来源':'合成样例'}[col])
    old=result['segments'].iloc[0].copy();text=old['text'];mid=len(text)//2
    first=old.copy();first[f['segment_id']]='doc0-part1';first[f['end_char']]=mid;first[f['segment_text']]=text[:mid]
    second=old.copy();second[f['segment_id']]='doc0-part2';second[f['segment_order']]=2
    second[f['start_char']]=mid;second[f['segment_text']]=text[mid:];second[f['topic_id']]=1
    result['segments']=pd.concat([pd.DataFrame([first,second]),result['segments'].iloc[1:]],ignore_index=True)
    result['topics'].loc[result['topics']['Topic'].eq(1),'Count']=2
    result['metadata']['counts']['segments']=5
    rename(result,1,'人工名称 / 主题一')
    raw=save_project(result)
    return open_project(raw),raw


class PolicyTransferTests(unittest.TestCase):
    def setUp(self):
        self.theme,self.raw=theme_project();self.digest=hashlib.sha256(self.raw).hexdigest()
        self.mapping={'title':'标题','date':'年份','region':'地区','source':'来源'}

    def test_filter_one_document_multiple_topics_and_new_project(self):
        before=copy.deepcopy(self.theme);all_docs=documents(self.theme);filtered=documents(self.theme,1)
        self.assertEqual([d['document_id'] for d in filtered],['doc-0','doc-1'])
        self.assertEqual([s['topic_id'] for s in filtered[0]['segments']],[0,1])
        self.assertEqual(filtered[0]['segments'][1]['topic_name'],'人工名称 / 主题一')
        plan=plan_transfer(self.theme,self.digest,['doc-0'],self.mapping)
        self.assertEqual(len(plan['rows']),1);self.assertEqual(plan['rows'][0]['segment_count'],2)
        project=apply_transfer(plan);policy=project['policies'][0]
        self.assertEqual(policy['original_text'],all_docs[0]['original_text'])
        self.assertEqual(policy['original_fields'],all_docs[0]['original_fields'])
        self.assertEqual((policy['title'],policy['date'],policy['region']),('政策doc-0','2020','首尔'))
        f=self.theme['metadata']['generated_fields']
        for part in policy['theme_origin']['segments']:
            source=self.theme['segments'].set_index(f['segment_id']).loc[part['segment_id']]
            self.assertEqual(policy['original_text'][part['start']:part['end']],source[f['segment_text']])
            self.assertEqual((part['start'],part['end'],part['topic_id']),
                             (source[f['start_char']],source[f['end_char']],source[f['topic_id']]))
        self.assertEqual(len(project['policies']),1);self.assertFalse(project['encodings'])
        self.assertIsNone(score(project)[1][0]['raw_pmc'])
        raw=save_policy_project(project)
        with zipfile.ZipFile(io.BytesIO(raw)) as z:self.assertEqual(json.loads(z.read('project.json'))['version'],4)
        self.assertEqual(open_policy_project(raw),project)
        pd.testing.assert_frame_equal(self.theme['rows'],before['rows']);pd.testing.assert_frame_equal(self.theme['segments'],before['segments'])
        self.assertEqual(self.theme['topic_edits'],before['topic_edits']);self.assertEqual(self.theme['visualization'],before['visualization'])

    def test_append_preserves_old_encodings_and_evidence(self):
        project,children=ready_project();before=copy.deepcopy(project);old_scores=score(project)
        plan=plan_transfer(self.theme,self.digest,['doc-0'],self.mapping,project)
        apply_transfer(plan,project)
        self.assertEqual(project['policies'][:2],before['policies']);self.assertEqual(project['encodings'],before['encodings'])
        self.assertEqual(project['versions'],before['versions']);self.assertEqual(score(project)[1][:2],old_scores[1])
        new=project['policies'][-1];pending=[x for x in coding_matrix(project) if x['policy_id']==new['policy_id']]
        self.assertEqual(len(pending),6);self.assertTrue(all(x['status']=='待编码' for x in pending))
        self.assertIsNone(score(project)[1][-1]['raw_pmc'])
        for i,(child,value) in enumerate(zip(children,['yes','yes','yes','no','no','no'])):
            record_encoding(project,new['policy_id'],child,value,'中文',3,5) if i==0 else record_encoding(project,new['policy_id'],child,value)
        self.assertEqual(score(project)[1][-1]['raw_pmc'],1.5)
        loaded=open_policy_project(save_policy_project(project));self.assertEqual(loaded,project)
        self.assertEqual(loaded['encodings'][-6]['evidence'],'中文')
        export=tables(loaded)['policies'].iloc[-1];self.assertEqual(export['theme_document_id'],'doc-0')
        self.assertEqual(len(json.loads(export['theme_segments_json'])),2)

    def test_duplicate_and_id_collision_selection(self):
        plan=plan_transfer(self.theme,self.digest,['doc-0','doc-1'],self.mapping)
        project=apply_transfer(plan);first=copy.deepcopy(project['policies'])
        keep=plan_transfer(self.theme,self.digest,['doc-0'],self.mapping,project,'keep')
        self.assertEqual((keep['duplicate_count'],keep['system_id_collisions'],keep['theme_id_conflicts']),(1,1,1))
        self.assertTrue(keep['rows'][0]['policy_id'].endswith(':append-000001'))
        skip=plan_transfer(self.theme,self.digest,['doc-0'],self.mapping,project,'exclude')
        self.assertFalse(skip['accepted'])
        with self.assertRaisesRegex(ValueError,'没有可转入'):apply_transfer(skip,project)
        self.assertEqual(project['policies'],first)
        apply_transfer(keep,project);self.assertEqual(len(project['policies']),3)
        self.assertEqual(open_policy_project(save_policy_project(project)),project)
        another,_=ready_project();another['policies'][0]['original_fields']['policy_id']='doc-0'
        conflict=plan_transfer(self.theme,self.digest,['doc-0'],self.mapping,another,'keep')
        self.assertEqual(conflict['document_id_conflicts'],1)
        apply_transfer(conflict,another)
        self.assertEqual(another['policies'][-1]['source_id_conflict_with'],another['policies'][0]['policy_id'])
        self.assertEqual(open_policy_project(save_policy_project(another)),another)

    def test_theme_legacy_versions_and_invalid_provenance(self):
        for version in (1,2,3):
            def legacy(files):
                if version < 3: files.pop('visualization.json')
                if version == 1: files.pop('topic_edits.json')
                info=json.loads(files['project.json']);info['version']=version
                files['project.json']=json.dumps(info).encode()
                manifest=json.loads(files['manifest.json'])
                for name in ('visualization.json','topic_edits.json'):
                    if name not in files: manifest['files'].pop(name)
                files['manifest.json']=json.dumps(manifest).encode()
            raw=change_zip(legacy_project(self.raw),legacy,True);theme=open_project(raw)
            project=apply_transfer(plan_transfer(theme,hashlib.sha256(raw).hexdigest(),['doc-0'],self.mapping))
            self.assertEqual(open_policy_project(save_policy_project(project)),project)
        project=apply_transfer(plan_transfer(self.theme,self.digest,['doc-0'],self.mapping))
        project['policies'][0]['theme_origin']['segments'][0]['end']=999999
        with self.assertRaisesRegex(ValueError,'位置'):save_policy_project(project)
        historical=copy.deepcopy(self.theme);historical['metadata'].pop('text_column')
        with self.assertRaisesRegex(ValueError,'明确选择'):plan_transfer(historical,self.digest,['doc-0'],self.mapping)
        with self.assertRaisesRegex(ValueError,'不一致'):plan_transfer(historical,self.digest,['doc-0'],self.mapping,text_column='标题')
        project=apply_transfer(plan_transfer(historical,self.digest,['doc-0'],self.mapping,text_column='text'))
        self.assertEqual(open_policy_project(save_policy_project(project)),project)

    def test_page_append_to_open_policy_project(self):
        project,_=ready_project();before=copy.deepcopy(project)
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
        app.radio(key='work_mode').set_value('基础政策评价').run()
        app.radio(key='pmc_mode').set_value('打开政策评价项目').run()
        app.file_uploader(key='pmc_project_upload').set_value(('old.pmc.zip',save_policy_project(project),'application/zip')).run(timeout=90)
        app.radio(key='work_mode').set_value('打开已有项目').run()
        app.file_uploader(key='project_upload').set_value(('theme.hpts.zip',self.raw,'application/zip')).run(timeout=90)
        app.radio(key='theme_transfer_target').set_value('追加到当前政策项目').run(timeout=90)
        boxes=[x for x in app.checkbox if str(x.key).startswith('theme_transfer_doc_')]
        boxes[0].check().run(timeout=90);app.button(key='theme_transfer_confirm').click().run(timeout=90)
        self.assertFalse(app.exception);actual=app.session_state['pmc_project']
        self.assertEqual(len(actual['policies']),3);self.assertEqual(actual['policies'][:2],before['policies'])
        self.assertEqual(actual['encodings'],before['encodings']);self.assertEqual(score(actual)[1][:2],score(before)[1])

    def test_page_open_filter_select_new_and_enter_policy(self):
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
        app.radio(key='work_mode').set_value('打开已有项目').run()
        app.file_uploader(key='project_upload').set_value(('theme.hpts.zip',self.raw,'application/zip')).run(timeout=90)
        self.assertFalse(app.exception)
        app.selectbox(key='theme_transfer_topic').select(1).run(timeout=90)
        boxes=[x for x in app.checkbox if str(x.key).startswith('theme_transfer_doc_')]
        self.assertEqual(len(boxes),2);boxes[0].check().run(timeout=90)
        self.assertEqual(len(app.session_state['theme_transfer_selection']),1)
        app.selectbox(key='theme_transfer_topic').select(2).run(timeout=90)
        self.assertEqual(app.session_state['theme_transfer_selection'],{'doc-0'})
        app.selectbox(key='theme_transfer_topic').select(1).run(timeout=90)
        boxes=[x for x in app.checkbox if str(x.key).startswith('theme_transfer_doc_')]
        self.assertTrue(boxes[0].value)
        app.button(key='theme_transfer_confirm').click().run(timeout=90)
        self.assertFalse(app.exception);self.assertEqual(app.radio(key='work_mode').value,'基础政策评价')
        project=app.session_state['pmc_project'];self.assertEqual(len(project['policies']),1)
        self.assertEqual(project['policies'][0]['theme_origin']['document_id'],'doc-0')
        self.assertEqual(open_policy_project(save_policy_project(project)),project)


if __name__=='__main__':unittest.main(verbosity=2)
