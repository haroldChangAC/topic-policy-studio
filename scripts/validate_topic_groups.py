"""Non-inferential grouping validation on existing synthetic and research inputs."""
import json
from pathlib import Path
import pandas as pd
from studio.project import open_project,save_project
from studio.runtime import ROOT
from studio.topic_groups import annual,create,mapping,summary,trace
from studio.workflow import csv_bytes

out=ROOT/'outputs/topic_groups_validation';out.mkdir(parents=True,exist_ok=True)
reports={}
for kind,path,topics,name in [('synthetic','outputs/annual_validation/synthetic_annual.hpts.zip',[0,1],'合成样例组'),
                               ('historical','outputs/parameter_comparison_120/A_默认.hpts.zip',[7,12],'法性相关（技术试用）')]:
    p=open_project((ROOT/path).read_bytes());f=p['metadata']['generated_fields'];before=p['segments'].copy(deep=True)
    old=p['topics'].copy(deep=True);gid=create(p,name,topics);assoc=trace(p);stat=summary(p).set_index('view_id').loc[gid]
    chosen=assoc[assoc.view_id.eq(gid)];assert len(chosen)==int(before[f['topic_id']].isin(topics).sum())
    assert stat['document_count']==chosen.document_id.nunique()
    for _,segment in chosen.iterrows():
        original=p['rows'].set_index(f['document_id']).loc[segment.document_id,f['original_text']]
        assert original[int(segment.start_char):int(segment.end_char)]==segment.segment_text
    if kind=='synthetic':
        settings=p['visualization']['annual_panel'];docs,panel,parts=annual(p,settings)
        for _,row in panel[panel.view_id.eq(gid)].iterrows():
            matching=docs[docs.group_key.eq(row.group_key)&(docs.year.isna() if pd.isna(row.year) else docs.year.eq(row.year))&docs.selected_by_metadata]
            part=parts[parts[f['document_id']].isin(matching.document_id)&parts[f['topic_id']].isin(topics)]
            assert row.segment_count==len(part) and row.document_count==part[f['document_id']].nunique()
        (out/'synthetic_annual.csv').write_bytes(csv_bytes(panel.drop(columns=['group_key'])))
    for label,frame in [('mapping',mapping(p)),('summary',summary(p)),('trace',chosen)]:
        (out/f'{kind}_{label}.csv').write_bytes(csv_bytes(frame))
    raw=save_project(p);(out/f'{kind}_grouped.hpts.zip').write_bytes(raw);back=open_project(raw)
    pd.testing.assert_frame_equal(before,p['segments']);pd.testing.assert_frame_equal(old,p['topics']);assert back['visualization']['topic_groups']==p['visualization']['topic_groups']
    report={'source':path,'topics':topics,'display_name':name,'segment_count':len(chosen),'input_record_documents':int(stat['document_count']),
            'original_assignments_unchanged':True,'keywords':'原始主题关键词，未重新计算','all_source_offsets_checked':True,'roundtrip':True}
    if kind=='historical':
        rawids=chosen['raw::witness_id'];report.update(witness_count=int(rawids.nunique()),source_status='pending_researcher_review',
            caveat='每个工作台输入记录是既有片段，不代表独立文献；仅因两个主题原始关键词均含法性而技术试用，不构成研究结论。')
    reports[kind]=report
(out/'summary.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2)+'\n');print(json.dumps(reports,ensure_ascii=False,indent=2))
