"""Exercise the actual Streamlit policy UI and local model with synthetic data.

Uses Streamlit AppTest, not a browser. The harness preloads only synthetic data.
"""
import json
from pathlib import Path

from streamlit.testing.v1 import AppTest
from studio.policy import score
from studio.policy_project import open_policy_project, save_policy_project

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/policy_local_eval'


def main():
    app=AppTest.from_file(str(ROOT/'tests/fixtures/policy_assist_browser_harness.py'),default_timeout=120).run()
    if app.exception:raise RuntimeError(str(app.exception))
    project=app.session_state['pmc_project'];before=score(project);n=len(project['ai_suggestions'])
    app.text_input(key='ai_model').set_value('qwen2.5:3b-instruct-q4_K_M').run()
    app.button(key='ai_generate').click().run(timeout=120)
    if app.exception:raise RuntimeError(str(app.exception))
    assert len(project['ai_suggestions'])==n+1 and score(project)==before and not project['encodings']
    item=project['ai_suggestions'][-1];sid=item['suggestion_id']
    if item['result']!='ready':raise RuntimeError('实际模型页面请求失败：'+item['error'])
    if item['model_status']=='yes' and item['match_status']=='unique' and item['start']<=item['matches'][0]['start']<item['matches'][0]['end']<=item['end']:
        action='accept'
    else:
        action='reject'
        app.radio(key='ai_review_action_'+sid).set_value('拒绝建议').run(timeout=120)
    app.button(key='FormSubmitter:ai_review_form_'+sid+'-提交审核').click().run(timeout=120)
    if app.exception:raise RuntimeError(str(app.exception))
    assert project['ai_reviews'][-1]['operation']==action
    raw=save_policy_project(project);(OUT/'page-test.pmc.zip').write_bytes(raw)
    assert open_policy_project(raw)==project
    report={'method':'Streamlit AppTest（非浏览器）','actual_local_model':item['model_reported'],
            'generated_suggestion':item,'human_test_action':action,'encoding_count':len(project['encodings']),
            'score_before':before[1][0]['raw_pmc'],'score_after':score(project)[1][0]['raw_pmc'],
            'project_saved_and_reopened':True}
    (OUT/'page-test.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'action':action,'model':item['model_reported'],'match':item['match_status'],
                      'score_before':report['score_before'],'score_after':report['score_after']},ensure_ascii=False))


if __name__=='__main__':main()
