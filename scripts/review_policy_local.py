"""Record test-designer reviews of the already observed synthetic responses.

This script does not call a model. The choices below were made after inspecting
the captured output, and are not independent research-policy judgments.
"""
import copy
import json
from pathlib import Path
from unittest.mock import patch

from studio.policy import score
from studio.policy_assist import evidence_matches, review
from studio.policy_project import excel_policy, open_policy_project, save_policy_project, tables
from studio.workflow import csv_bytes

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/policy_local_eval'
CODER='测试设计者（Codex，非领域专家）'


def main():
    project=open_policy_project((OUT/'adaptive.pmc.zip').read_bytes())
    data=json.loads((OUT/'generated.json').read_text())
    cases={x['case_id']:x['suggestion'] for x in data['cases']}
    supplement=json.loads((OUT/'supplement.json').read_text())['suggestion']
    adaptive=json.loads((OUT/'adaptive.json').read_text())['suggestion']
    assert not project['encodings'] and all(x['raw_pmc'] is None for x in score(project)[1])
    assert cases['S01_explicit']['model_status']=='yes' and cases['S01_explicit']['match_status']=='unique'
    assert cases['S03_near_words']['suggested_status']=='pending'
    assert cases['S04_repeated']['suggested_status']=='yes' and cases['S04_repeated']['match_status']=='unique'
    assert supplement['match_status']=='missing' and adaptive['match_status']=='multiple'
    with patch('socket.socket.connect',side_effect=AssertionError('review must be offline')):
        try:
            review(project,supplement['suggestion_id'],'accept')
            raise AssertionError('未匹配的证据不应直接接受')
        except ValueError as exc:
            assert '唯一' in str(exc)
        review(project,cases['S01_explicit']['suggestion_id'],'accept',coder=CODER)
        review(project,cases['S02_absent']['suggestion_id'],'reject',coder=CODER)
        review(project,cases['S03_near_words']['suggestion_id'],'modify','pending',
               rationale='人工核对：补充图书与生活金钱补贴不同；保留待核查。',coder=CODER)
        review(project,cases['S04_repeated']['suggestion_id'],'accept',coder=CODER)
        correct='对低收入家庭每月发放生活补贴'
        policy=next(p for p in project['policies'] if p['policy_id']==supplement['policy_id'])
        state,positions=evidence_matches(policy['original_text'],correct)
        assert state=='multiple' and [(m['start'],m['end']) for m in positions]==[(0,14),(25,39)]
        before=copy.deepcopy(project['encodings'])
        try:
            review(project,supplement['suggestion_id'],'modify','yes',correct,0,14,'人工修正模型返回的坐标字符串。',CODER)
            raise AssertionError('覆盖已有编码需要明确允许')
        except ValueError as exc:
            assert '已有正式编码' in str(exc)
        assert project['encodings']==before
        review(project,supplement['suggestion_id'],'modify','yes',correct,0,14,
               '人工修正模型返回的坐标字符串，并选择第一处原文。',CODER,replace=True)
        try:
            review(project,adaptive['suggestion_id'],'accept',coder=CODER)
            raise AssertionError('多处匹配应选择位置')
        except ValueError as exc:
            assert '唯一' in str(exc)
        second=adaptive['matches'][1]
        assert (second['start'],second['end'])==(888,907)
        review(project,adaptive['suggestion_id'],'accept',start=second['start'],end=second['end'],coder=CODER)
        revised='对低收入家庭每月发放生活津贴'
        fabricated=evidence_matches(policy['original_text'],revised)
        assert fabricated==('missing',[])
        raw=save_policy_project(project); (OUT/'reviewed.pmc.zip').write_bytes(raw)
        loaded=open_policy_project(raw);assert loaded==project
        for name in ('ai_suggestions','ai_reviews','policy_scores'):
            (OUT/(name+'.csv')).write_bytes(csv_bytes(tables(loaded)[name]))
        (OUT/'reviewed.xlsx').write_bytes(excel_policy(loaded))
        totals=score(loaded)[1]
        summary={'synthetic_only':True,'researcher':'测试设计者（Codex，非领域专家）',
                 'unmatched_model_evidence':'[0,15)；直接接受被拒绝，人工改为原文并明确替换',
                 'multiple_model_evidence':adaptive['evidence'],'explicitly_selected_position':[888,907],
                 'controlled_altered_evidence':revised,'controlled_match_status':fabricated[0],
                 'review_operations':[x['operation'] for x in loaded['ai_reviews']],
                 'policy_scores':[{'title':x['title'],'raw_pmc':x['raw_pmc'],'reason':x['reason']} for x in totals],
                 'roundtrip_equal':True,'offline_review':True}
        (OUT/'reviewed-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
