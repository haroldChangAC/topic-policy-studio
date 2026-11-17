"""Exercise both interface locales and the locked offline model in a scratch copy."""

import argparse
import csv
import io
import json
import platform
import time
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree

import pandas as pd
from openpyxl import load_workbook
from PIL import Image
from streamlit.testing.v1 import AppTest

from studio.annual import calculate as annual_panel
from studio.cooccurrence import saved_tokenizer_settings
from studio.corpus import prepare, read_csv
from studio.cross_year import calculate as cross_year
from studio.deferred_exports import create_export
from studio.keywords import make_settings, preview
from studio.modeling import load_encoder
from studio.policy import score
from studio.policy_project import open_policy_project, save_policy_project
from studio.project import open_project
from studio.runtime import ROOT, configure
from studio.topic_groups import annual as grouped_panel
from studio.workflow import analyze

ENGLISH = {
    'transit': 'The bus route connects schools to the rail station. Transfers, bicycle parking and fare passes help passengers reach jobs. The transit agency tracks bus ridership and route frequency.',
    'housing': 'Affordable rental apartments require housing grants and tenant protections. Construction and rehabilitation expand the supply of subsidized homes. The housing office tracks rent, vacancies and applications.',
    'air quality': 'Air quality monitors track ozone and particulate pollution. Factories report emissions and test filters. The environment office records pollution concentrations, dust controls and inspection results.',
}
CHINESE = {
    'transit': '公交线路连接学校和车站，交通部门记录班次和客流。',
    'housing': '住房部门发布保障房租金和申请数据，跟踪租赁补贴。',
    'air quality': '环境部门检测臭氧浓度和颗粒物，跟踪企业排放。',
}


def corpus_bytes(mixed=False):
    """Create distinct fictional records. Years and groups are synthetic metadata."""
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=['title', 'text', 'year', 'group'])
    writer.writeheader()
    for topic, text in ENGLISH.items():
        for index in range(12 if not mixed else 6):
            source = f"  {text}\nReporting area {index + 1}. Internal reference pilot-{index + 1}.  "
            if mixed:
                source += CHINESE[topic]
            elif index == 0:
                source += '\n\n' + '\n\n'.join([text] * 4)
            writer.writerow({'title': f'{topic} {index + 1}', 'text': source,
                             'year': str(2022 + index % 3), 'group': ['North', 'South'][index % 2]})
    return stream.getvalue().encode('utf-8')


def assert_page(app):
    assert not app.exception and not app.error, ([item.value for item in app.exception], [item.value for item in app.error])


def verify_offsets(result):
    f = result['metadata']['generated_fields']
    originals = result['rows'].set_index(f['document_id'])[f['original_text']]
    for item in result['segments'].to_dict('records'):
        source = originals[item[f['document_id']]]
        assert item[f['segment_text']] == source[int(item[f['start_char']]):int(item[f['end_char']])]
    return originals


def summary(result, elapsed):
    metadata = result['metadata']
    return {
        'documents': len(result['rows']), 'segments': len(result['segments']),
        'long_documents': metadata['counts']['long_documents'], 'topics': metadata['topic_count_excluding_outliers'],
        'outlier_segments': metadata['outlier_count'], 'seconds': round(elapsed, 3),
        'keywords': [{'id': int(r.Topic), 'keywords': r.Keywords} for r in result['topics'].itertuples()],
        'keyword_processing': metadata['keyword_processing'], 'max_input_tokens': metadata['max_input_tokens'],
        'model': metadata['model_id'], 'revision': metadata['model_revision'], 'versions': metadata['versions'],
    }


def english_ui(app, raw, out):
    app.file_uploader(key='csv_upload').set_value(('synthetic_english.csv', raw, 'text/csv')).run()
    assert_page(app)
    app.radio(key='split_mode').set_value('自动切分超长文本').run()
    app.radio(key='keyword_mode').set_value('英文分词').run()
    app.checkbox(key='stop_enabled').check().run()
    app.text_area(key='stop_text').set_value('the\nreporting\n').run()
    assert app.radio(key='keyword_mode').label == 'Keyword tokenization'
    assert app.radio(key='keyword_mode').value == '英文分词'
    assert not app.button(key='run_analysis').disabled
    started = time.monotonic()
    app.button(key='run_analysis').click().run()
    elapsed = time.monotonic() - started
    assert_page(app)
    current = app.session_state['analysis_result']['result']
    originals = verify_offsets(current)
    assert originals.tolist() == read_csv(raw).frame.text.tolist()
    assert current['metadata']['max_input_tokens'] <= current['metadata']['model_token_limit']
    settings = current['metadata']['keyword_processing']
    assert saved_tokenizer_settings(current['metadata']) == settings
    assert preview("A city's transit-oriented plan and THE report", settings) == ['a', "city's", 'transit-oriented', 'plan', 'and', 'report']
    weights = current['visualization']['keyword_weights']
    assert weights and all(row['keyword'] not in ('the', 'reporting') for row in weights)

    app.text_input(key=f"name_input_new_{app.selectbox(key='edit_topic_new').value}").set_value('Pilot · 保留原样')
    next(button for button in app.button if button.label == 'Save name').click().run()
    assert_page(app)
    assert 'Pilot · 保留原样' in app.session_state['analysis_result']['result']['topic_edits']['labels'].values()
    app.button(key='export_prepare_new').click().run()
    assert_page(app)
    archive = app.session_state['prepared_export_new']['data']
    run_id = current['metadata']['run_id']
    app.radio(key='studio_language').set_value('中文').run()
    assert_page(app)
    live = app.session_state['analysis_result']['result']
    assert app.file_uploader(key='csv_upload').value.name == 'synthetic_english.csv'
    assert live['metadata']['run_id'] == run_id and 'Pilot · 保留原样' in live['topic_edits']['labels'].values()
    assert app.session_state['prepared_export_new']['data'] == archive
    app.radio(key='studio_language').set_value('English').run()
    assert_page(app)
    assert app.radio(key='keyword_mode').value == '英文分词'
    restored = open_project(archive)
    assert restored['topic_edits'] == live['topic_edits']
    pd.testing.assert_series_equal(verify_offsets(restored), originals)
    for column in ('segment_id', 'topic_id', 'start_char', 'end_char', 'segment_text'):
        f = live['metadata']['generated_fields']
        assert restored['segments'][f[column]].tolist() == live['segments'][f[column]].tolist()
    out.joinpath('synthetic_english.hpts.zip').write_bytes(archive)
    csv_data = create_export(restored, 'segments')['data']
    exported = pd.read_csv(io.BytesIO(csv_data), keep_default_na=False)
    assert len(exported) == len(restored['segments'])
    assert exported[f['segment_text']].tolist() == restored['segments'][f['segment_text']].tolist()
    out.joinpath('synthetic_english_segments.csv').write_bytes(csv_data)
    excel = create_export(restored, 'excel')['data']
    workbook = load_workbook(io.BytesIO(excel), read_only=True)
    assert workbook.sheetnames[:4] == ['records', 'segments', 'topics', 'run_metadata']
    assert [row[1] for row in workbook['records'].iter_rows(min_row=2, values_only=True)] == originals.tolist()
    workbook.close()
    out.joinpath('synthetic_english.xlsx').write_bytes(excel)
    return summary(live, elapsed)


def existing_examples(app, theme_path, policy_path, download_csv):
    raw = theme_path.read_bytes()
    baseline = open_project(raw)
    app.radio(key='work_mode').set_value('打开已有项目').run()
    app.file_uploader(key='project_upload').set_value((theme_path.name, raw, 'application/zip')).run()
    assert_page(app)
    opened = app.session_state['opened_project']['result']
    assert opened['yearly_analysis'] == baseline['yearly_analysis']
    assert opened['topic_edits'] == baseline['topic_edits']
    pd.testing.assert_series_equal(verify_offsets(opened), verify_offsets(baseline))
    s = opened['visualization']['annual_panel']
    pd.testing.assert_frame_equal(annual_panel(opened, s)[1], annual_panel(baseline, s)[1])
    pd.testing.assert_frame_equal(grouped_panel(opened, s)[1], grouped_panel(baseline, s)[1])
    nodes, links, _, _ = cross_year(opened, opened['visualization']['cross_year_links'])
    if download_csv:
        browser = pd.read_csv(download_csv)
        pd.testing.assert_series_equal(browser.similarity, links.similarity, check_names=False)
        for name in ('source_year', 'target_year', 'source_topic_id', 'target_topic_id', 'in_graph'):
            assert browser[name].tolist() == links[name].tolist()
    png = create_export(opened, 'cross_png')['data']
    with Image.open(io.BytesIO(png)) as image:
        image.verify()
        size = image.size
    svg = create_export(opened, 'cross_svg')['data']
    assert ElementTree.fromstring(svg).tag.endswith('svg')
    assert app.session_state.get('prepared_export_restored') is None
    app.button(key='export_prepare_restored').click().run()
    assert_page(app)
    reopened = open_project(app.session_state['prepared_export_restored']['data'])
    assert reopened['yearly_analysis'] == baseline['yearly_analysis']
    assert reopened['visualization']['topic_groups'] == baseline['visualization']['topic_groups']

    raw = policy_path.read_bytes()
    baseline = open_policy_project(raw)
    app.radio(key='work_mode').set_value('基础政策评价').run()
    app.radio(key='pmc_mode').set_value('打开政策评价项目').run()
    app.file_uploader(key='pmc_project_upload').set_value((policy_path.name, raw, 'application/zip')).run()
    assert_page(app)
    app.radio(key='studio_language').set_value('中文').run()
    assert_page(app)
    app.radio(key='studio_language').set_value('English').run()
    assert_page(app)
    opened_policy = app.session_state['pmc_project']
    assert score(opened_policy) == score(baseline)
    assert opened_policy['encodings'] == baseline['encodings']
    assert open_policy_project(save_policy_project(opened_policy)) == opened_policy
    return {'real_theme': {'documents': len(opened['rows']), 'segments': len(opened['segments']),
                           'nodes': len(nodes), 'candidate_pairs': len(links), 'drawn_links': int(links.in_graph.sum()),
                           'sankey_png_pixels': list(size), 'svg_parsed': True, 'browser_csv_matches': bool(download_csv)},
            'synthetic_policy': {'documents': len(baseline['policies']), 'indicator_versions': len(baseline['versions']),
                                 'codes': len(baseline['encodings']), 'scores': score(baseline)[1]}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    parser.add_argument('theme', type=Path)
    parser.add_argument('policy', type=Path)
    parser.add_argument('--browser-csv', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True)
    configure()
    raw = corpus_bytes()
    args.out.joinpath('synthetic_english.csv').write_bytes(raw)
    with patch('socket.socket.connect', side_effect=AssertionError('Offline validation blocked a socket connection')):
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=120).run()
        assert_page(app)
        app.radio(key='studio_language').set_value('English').run()
        assert_page(app)
        english = english_ui(app, raw, args.out)
        examples = existing_examples(app, args.theme, args.policy, args.browser_csv)
        mixed_bytes = corpus_bytes(mixed=True)
        args.out.joinpath('synthetic_mixed.csv').write_bytes(mixed_bytes)
        settings = make_settings('chinese', True, ['the'], ['保障房'])
        corpus = read_csv(mixed_bytes)
        started = time.monotonic()
        mixed = analyze(prepare(corpus, 'text', 'keep'), load_encoder('cpu'), split_mode='auto', keyword_settings=settings)
        assert verify_offsets(mixed).tolist() == corpus.frame.text.tolist()
        assert '保障房' in preview('Affordable 保障房 and transit', settings)
        report = {'hardware': platform.machine(), 'os': platform.platform(), 'python': platform.python_version(),
                  'socket_connections_blocked': True, 'english': english,
                  'mixed': summary(mixed, time.monotonic() - started), **examples}
        args.out.joinpath('results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
