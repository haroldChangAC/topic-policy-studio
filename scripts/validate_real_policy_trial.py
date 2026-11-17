"""Reopen one real offline fit, annotate, recalculate, audit, plot and round-trip."""
import copy
import io
import json
from pathlib import Path

import pandas as pd
from PIL import Image
from studio.runtime import ROOT
from studio.project import open_project, save_project
from studio.annual import calculate, group_key, render as annual_line, RULE
from studio.annual_heatmap import ALL, data as heat_data, render as heat_render, samples
from studio.grouped_charts import basis, cells, evidence, for_scope, render_line, signature, source
from studio.topic_groups import create, mapping
from studio.topic_names import rename
from studio.visualization import find_chinese_font
from studio.workflow import csv_bytes

out=ROOT/'outputs/real_policy_2022_2024'
r=open_project((out/'before_grouping.hpts.zip').read_bytes())
f=r['metadata']['generated_fields'];segments=r['segments'];rows=r['rows']
original_segments=segments.copy(deep=True);original_topics=r['topics'].copy(deep=True)
assert len(rows)==6 and len(segments)==685
for doc in rows.to_dict('records'):
    ss=segments[segments[f['document_id']].eq(doc[f['document_id']])].sort_values(f['segment_order'])
    assert ''.join(ss[f['segment_text']])==doc[f['original_text']]
    for part in ss.to_dict('records'):
        assert doc[f['original_text']][int(part[f['start_char']]):int(part[f['end_char']])]==part[f['segment_text']]
        assert int(part[f['token_count']])<=128
rename(r,40,'乡村振兴与城乡融合（试用）')
gid=create(r,'农村与乡村发展（试用）',[8,40,56])
settings={'year_field':'年份','group_field':'分组','exclude_missing_year':True,'exclude_missing_group':True,
          'include_outliers':True,'metric':'segment_share','selected_groups':[group_key('北京市'),group_key('上海市')],
          'selected_topics':[8,40,56], 'rule_version':1}
r['visualization']['annual_panel']=settings
r['visualization']['annual_view']='grouped'
years=[2022,2023,2024]
original_docs,original_panel,_=calculate(r,settings)
docs,panel,parts,members=source(r,settings)
assert docs.year_missing.sum()==0 and docs.group_missing.sum()==0
scopes=[group_key('北京市'),group_key('上海市')]
# Check source records independently of the annual panel implementation.
manual=[]
for city in ('北京市','上海市'):
    for year in years:
        dr=rows[(rows['分组']==city)&(rows['年份']==str(year))]
        subset=segments[segments[f['document_id']].isin(dr[f['document_id']])]
        match=subset[subset[f['topic_id']].isin([8,40,56])]
        table=panel[(panel.group==city)&(panel.year==year)&panel.view_id.eq(gid)].iloc[0]
        assert (table.segment_count,table.segment_denominator,table.document_count,table.document_denominator)==(len(match),len(subset),match[f['document_id']].nunique(),subset[f['document_id']].nunique())
        assert abs(table.segment_share-len(match)/len(subset))<1e-12
        assert abs(table.document_mention_rate-match[f['document_id']].nunique()/subset[f['document_id']].nunique())<1e-12
        matched_docs,matched_parts=evidence(r,docs,parts,members,group_key(city),year,gid)
        assert set(matched_parts[f['segment_id']])==set(match[f['segment_id']])
        manual.append({'group':city,'year':year,'source_document_ids':dr['文档ID'].tolist(),
                       'studio_document_ids':dr[f['document_id']].tolist(),'member_topics':[8,40,56],
                       'group_segment_count':len(match),'segment_denominator':len(subset),
                       'document_count':match[f['document_id']].nunique(),'document_denominator':subset[f['document_id']].nunique(),
                       'member_counts':{str(t):int(match[f['topic_id']].eq(t).sum()) for t in [8,40,56]},
                       'segment_ids':match[f['segment_id']].tolist()})
# Exercise both outlier choices against direct counts.
without=dict(settings,include_outliers=False)
docs_no,panel_no,parts_no,members_no=source(r,without)
for city in ('北京市','上海市'):
    for year in years:
        dr=rows[(rows['分组']==city)&(rows['年份']==str(year))]
        raw=segments[segments[f['document_id']].isin(dr[f['document_id']])]
        included=raw[raw[f['topic_id']].ne(-1)]
        p=panel_no[(panel_no.group==city)&(panel_no.year==year)&panel_no.view_id.eq(gid)].iloc[0]
        assert p.segment_denominator==len(included) and p.document_denominator==included[f['document_id']].nunique()
        expected=len(included[included[f['topic_id']].isin([8,40,56])]);assert p.segment_count==expected
        rec=next(x for x in manual if x['group']==city and x['year']==year)
        rec.update(outlier_segments=int(raw[f['topic_id']].eq(-1).sum()),
                   segment_denominator_without_outliers=len(included),
                   document_denominator_without_outliers=included[f['document_id']].nunique())
# Representative source offsets for later independent review; complete original text remains in project.
tracecols=[f['document_id'],f['segment_id'],f['segment_order'],f['start_char'],f['end_char'],f['segment_text'],f['topic_id'],'文档ID','标题','年份','分组','来源网址']
chosen=segments[segments[f['topic_id']].isin([8,40,56])][tracecols]
(out/'group_evidence.csv').write_bytes(csv_bytes(chosen))
(out/'topic_group_mapping.csv').write_bytes(csv_bytes(mapping(r)))
(out/'manual_audit.json').write_text(json.dumps(manual,ensure_ascii=False,indent=2)+'\n')
(out/'original_annual_panel.csv').write_bytes(csv_bytes(original_panel.drop(columns=['group_key'])))
(out/'grouped_annual_panel.csv').write_bytes(csv_bytes(panel.drop(columns=['group_key'])))
font,family=find_chinese_font();assert font
cfg={'style':'color','show_title':True,'font_size':10}
# Same chart data providers and export functions as the application.
line_data=pd.concat([cells(for_scope(r,docs,panel,parts,members,g),'segment_share',years,[gid],settings,members) for g in scopes],ignore_index=True)
heat_data_group=cells(for_scope(r,docs,panel,parts,members,ALL),'segment_share',years,[gid,'topic:0','topic:16'],settings,members)
heat_beijing=cells(for_scope(r,docs,panel,parts,members,group_key('北京市')),'segment_share',years,[gid,'topic:16'],settings,members)
zero=heat_beijing[(heat_beijing.year==2022)&heat_beijing.topic_group_id.eq('topic:16')].iloc[0]
assert zero.numerator==0 and zero.denominator==80 and zero.value==0 and zero.missing_status=='有纳入样本'
sig=signature(basis(r,settings))
for frame in (line_data,heat_data_group,heat_beijing):frame['scheme_signature']=sig
(out/'grouped_line.csv').write_bytes(csv_bytes(line_data.drop(columns=['group_key'])))
(out/'grouped_heatmap.csv').write_bytes(csv_bytes(heat_data_group.drop(columns=['group_key'])))
(out/'grouped_heatmap_beijing.csv').write_bytes(csv_bytes(heat_beijing.drop(columns=['group_key'])))
orig_heat=heat_data(original_panel,'segment_share',ALL,years,[8,40,56])
for k in ('year_field','group_field','exclude_missing_year','exclude_missing_group','include_outliers'):orig_heat[k]=settings[k]
orig_heat['denominator_rule']=RULE
(out/'original_heatmap.csv').write_bytes(csv_bytes(orig_heat.drop(columns=['group_key'])))
(out/'sample_counts.csv').write_bytes(csv_bytes(samples(r,docs,ALL,years)))
figs={}
for fmt in ('png','svg'):
    output=[('original_line',annual_line(original_panel,'segment_share',scopes,[8,40,56],10,6,150,font,cfg,fmt,True)),
            ('original_heatmap',heat_render(orig_heat,10,6,150,font,cfg,fmt)),
            ('grouped_line',render_line(line_data,10,6,150,font,cfg,fmt)),
            ('grouped_heatmap',heat_render(heat_data_group.assign(topic_id=heat_data_group.topic_group_id,topic_label=heat_data_group.topic_group_name),10,6,150,font,cfg,fmt,grouped=True)),
            ('grouped_heatmap_beijing',heat_render(heat_beijing.assign(topic_id=heat_beijing.topic_group_id,topic_label=heat_beijing.topic_group_name),10,6,150,font,cfg,fmt,grouped=True))]
    for name,(blob,missing) in output:
        assert not missing,missing
        file=out/f'{name}.{fmt}';file.write_bytes(blob)
        if fmt=='png':
            im=Image.open(io.BytesIO(blob));assert im.size==(1500,900);im.verify()
        else:assert b'<svg' in blob
        figs[file.name]=len(blob)
# Data exported is read back from disk, not merely compared with in-memory UI.
for name,frame in [('grouped_line.csv',line_data),('grouped_heatmap.csv',heat_data_group),('grouped_heatmap_beijing.csv',heat_beijing)]:
    read=pd.read_csv(out/name,keep_default_na=False)
    assert len(read)==len(frame)
    for i,item in read.iterrows():
        source_row=frame.iloc[i]
        assert item.topic_group_id==source_row.topic_group_id and item.member_topic_ids==source_row.member_topic_ids
        assert int(item.numerator)==source_row.numerator and int(item.denominator)==source_row.denominator
        assert abs(float(item.value)-float(source_row.value))<1e-12
b=basis(r,settings)
r['visualization']['grouped_annual_charts']={'schema_version':1,'basis':b,'signature':sig,
    'line':{'scopes':scopes,'years':years,'topics':[gid],'metric':'segment_share'},
    'heat':{'scopes':[ALL],'years':years,'topics':[gid,'topic:0','topic:16'],'metric':'segment_share'}}
r['visualization']['annual_heatmap']={'schema_version':1,'group_key':ALL,'years':years,'topics':[8,40,56],'metric':'segment_share'}
raw=save_project(r);(out/'real_policy_grouped.hpts.zip').write_bytes(raw)
back=open_project(raw)
assert back['project_info']['version']==4 and back['visualization']['topic_groups']==r['visualization']['topic_groups']
assert back['visualization']['grouped_annual_charts']==r['visualization']['grouped_annual_charts']
pd.testing.assert_frame_equal(segments,original_segments);pd.testing.assert_frame_equal(r['topics'],original_topics)
assert back['segments'][f['topic_id']].tolist()==segments[f['topic_id']].tolist()
assert back['rows'][f['original_text']].tolist()==rows[f['original_text']].tolist()
d2,p2,parts2,mem2=source(back,settings)
for scope in scopes:
    a=cells(for_scope(r,docs,panel,parts,members,scope),'segment_share',years,[gid],settings,members)
    b2=cells(for_scope(back,d2,p2,parts2,mem2,scope),'segment_share',years,[gid],settings,mem2)
    pd.testing.assert_frame_equal(a,b2)
result={'real_corpus':True,'input_documents':len(rows),'segments':len(segments),'topic_count_excluding_outliers':r['metadata']['topic_count_excluding_outliers'],
        'outliers':r['metadata']['outlier_count'],'group_id':gid,'members':[8,40,56],
        'manual_cells':manual,'font':family,'exports':figs,'format_version':back['project_info']['version'],
        'zip_bytes':len(raw),'settings_reopen_equal':True,'grouped_values_reopen_equal':True,
        'original_text_and_assignments_preserved':True,'token_and_position_checks':True,
        'matching_csv_and_chart_inputs':True}
(out/'trial_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
