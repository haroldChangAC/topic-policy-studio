"""Test-designer review of an already generated synthetic long-policy result."""
import json
from pathlib import Path
from unittest.mock import patch

from openpyxl import load_workbook
from studio.policy import score
from studio.policy_assist import review
from studio.policy_long import candidates, coverage
from studio.policy_project import excel_policy, open_policy_project, save_policy_project, tables
from studio.workflow import csv_bytes

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/policy_long_validation'


def main():
    p=open_policy_project((OUT/'real_generated.pmc.zip').read_bytes());job=p['long_jobs'][0]
    assert coverage(job)['all_returned'] and not p['encodings'] and score(p)[1][0]['raw_pmc'] is None
    found,unresolved=candidates(p,job)
    chosen=next(c for c in found if c['start']<=83 and c['end']>=102 and '现金生活补助' in c['evidence'])
    with patch('socket.socket.connect',side_effect=AssertionError('offline review')):
        event=review(p,chosen['suggestion_ids'][0],'modify','yes',chosen['evidence'],chosen['start'],chosen['end'],
                     '测试设计者人工核对：原文明确每月发放现金生活补助。','测试设计者 Codex')
        assert score(p)[1][0]['raw_pmc']==1
        raw=save_policy_project(p);(OUT/'real_reviewed.pmc.zip').write_bytes(raw)
        restored=open_policy_project(raw);assert restored==p
        for name in ('long_jobs','long_segments','long_evidence','long_unresolved','ai_suggestions','ai_reviews'):
            (OUT/(name+'.csv')).write_bytes(csv_bytes(tables(p)[name]))
        (OUT/'real_reviewed.xlsx').write_bytes(excel_policy(p))
        workbook=load_workbook(OUT/'real_reviewed.xlsx',read_only=True)
        assert {'long_jobs','long_segments','long_evidence','long_unresolved'}.issubset(workbook.sheetnames)
    result={'synthetic':True,'reviewer':'测试设计者 Codex，非独立领域专家','action':event['operation'],
            'selected_evidence':chosen['evidence'],'position':[chosen['start'],chosen['end']],
            'model_raw_status':next(s['model_status'] for s in p['ai_suggestions'] if s['suggestion_id']==chosen['suggestion_ids'][0]),
            'final_formal_status':'yes','score_after_manual_review':score(p)[1][0]['raw_pmc'],
            'candidate_count':len(found),'unresolved':unresolved,'saved_and_reopened':True,'excel_opened':True}
    (OUT/'real_reviewed.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
