import copy
import io
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
from streamlit.testing.v1 import AppTest
from matplotlib.axes import Axes

from test_annual import fixture
from studio.annual import calculate, group_key
from studio.annual_heatmap import ALL, data, evidence, render, samples, validate_heatmap
from studio.project import save_project,open_project
from studio.runtime import ROOT
from studio.topic_names import rename
from studio.visualization import find_chinese_font
from studio.workflow import csv_bytes


def prepared(outliers=True):
    result,settings=fixture();settings['include_outliers']=outliers
    docs,panel,parts=calculate(result,settings)
    settings['selected_groups']=docs.loc[docs.selected_by_metadata,'group_key'].unique().tolist()
    settings['selected_topics']=sorted(panel.topic_id.unique().tolist())
    result['visualization']['annual_panel']=settings
    return result,settings,docs,panel,parts


class HeatmapTests(unittest.TestCase):
    def test_hand_and_zero_vs_missing(self):
        result,settings,docs,panel,parts=prepared();beijing=group_key('北京');shanghai=group_key('上海')
        t=data(panel,'segment_share',beijing,[2020,2021],[0,1,-1]).set_index(['year','topic_id'])
        self.assertEqual(t.loc[(2020,0),'value'],2/5);self.assertEqual(t.loc[(2020,1),'value'],2/5)
        self.assertEqual((t.loc[(2020,0),'numerator'],t.loc[(2020,0),'denominator']),(2,5))
        self.assertTrue(pd.isna(t.loc[(2021,0),'value']));self.assertTrue(pd.isna(t.loc[(2021,0),'plot_value']))
        t=data(panel,'segment_count',shanghai,[2020,2021],[0,1,-1]).set_index(['year','topic_id'])
        self.assertEqual(t.loc[(2021,0),'value'],0);self.assertEqual(t.loc[(2021,0),'plot_value'],0)
        self.assertEqual(t.loc[(2021,0),'missing_status'],'有纳入样本')
        self.assertEqual(t.loc[(2020,0),'numerator'],0);self.assertTrue(pd.isna(t.loc[(2020,0),'value']));self.assertTrue(pd.isna(t.loc[(2020,0),'plot_value']))
        self.assertEqual(t.loc[(2020,0),'missing_status'],'无纳入样本或分母为零')
        for metric in ('segment_count','segment_share','document_count','document_mention_rate'):
            cells=data(panel,metric,shanghai,[2020,2021],[0]).set_index('year')
            self.assertEqual((cells.loc[2020,'numerator'],cells.loc[2020,'denominator']),(0,0))
            self.assertTrue(pd.isna(cells.loc[2020,'value']))
            self.assertEqual(cells.loc[2021,'value'],0)
            exported=pd.read_csv(io.BytesIO(csv_bytes(cells.reset_index())),keep_default_na=False).set_index('year')
            self.assertEqual(exported.loc[2020,'value'],'')
            self.assertEqual(float(exported.loc[2021,'value']),0)
        t=data(panel,'document_mention_rate',beijing,[2020],[0,1]).set_index('topic_id')
        self.assertEqual(t.loc[0,'value'],.5);self.assertEqual(t.loc[1,'value'],1)
        matched,segments=evidence(result,docs,parts,beijing,2020,1)
        self.assertEqual(len(matched),2);self.assertEqual(len(segments),2)
        f=result['metadata']['generated_fields'];rows=result['rows'].set_index(f['document_id'])
        for _,seg in segments.iterrows():
            text=rows.loc[seg[f['document_id']],f['original_text']]
            self.assertEqual(text[seg[f['start_char']]:seg[f['end_char']]],seg[f['segment_text']])

    def test_outliers_sample_counts_and_aggregate(self):
        result,settings,docs,panel,parts=prepared(False);shanghai=group_key('上海')
        stat=samples(result,docs,shanghai,[2020,2021]).set_index('year')
        self.assertEqual((stat.loc[2021,'raw_valid_documents'],stat.loc[2021,'document_denominator'],stat.loc[2021,'included_segments']),(1,0,0))
        self.assertEqual((stat.loc[2021,'outlier_segments'],stat.loc[2021,'modeled_segments_before_outlier_filter']),(1,1))
        t=data(panel,'segment_share',group_key('北京'),[2020],[0,1]).set_index('topic_id')
        self.assertEqual(t.loc[0,'value'],.5);self.assertEqual(t.loc[0,'denominator'],4)
        # Make the third policy a separate group in the same year, retaining one group per document.
        p,c=fixture();f=p['metadata']['generated_fields'];doc_id=p['rows'].iloc[2][f['document_id']]
        p['rows'].loc[p['rows'][f['document_id']].eq(doc_id),'年份']='2020'
        p['segments'].loc[p['segments'][f['document_id']].eq(doc_id),'年份']='2020'
        docs,panel,parts=calculate(p,c);combined=data(panel,'segment_share',ALL,[2020],[0,1,-1]).set_index('topic_id')
        self.assertEqual((combined.loc[0,'numerator'],combined.loc[0,'denominator']),(2,6));self.assertAlmostEqual(combined.loc[0,'value'],2/6)
        s=samples(p,docs,ALL,[2020]).iloc[0];self.assertEqual((s.raw_valid_documents,s.document_denominator,s.included_segments,s.outlier_segments),(3,3,6,2))

    def test_missing_metadata_settings_and_persistence(self):
        result,settings,docs,panel,parts=prepared();settings['exclude_missing_year']=False;settings['exclude_missing_group']=False
        docs,panel,parts=calculate(result,settings)
        self.assertTrue(any(docs.group_missing & docs.year_missing));all_year=data(panel,'segment_count',ALL,[2020,2021],[0,1])
        self.assertEqual(all_year.year.unique().tolist(),[2020,2021])
        obj={'schema_version':1,'group_key':ALL,'years':[2020,2021],'topics':[0,1], 'metric':'document_mention_rate'}
        result['visualization']['annual_heatmap']=obj;validate_heatmap(result,result['visualization'],obj)
        before=data(panel,obj['metric'],ALL,obj['years'],obj['topics']);result=open_project(save_project(result));self.assertEqual(result['visualization']['annual_heatmap'],obj)
        _,panel2,_=calculate(result,settings);pd.testing.assert_frame_equal(before,data(panel2,obj['metric'],ALL,obj['years'],obj['topics']))
        rename(result,0,'很长的中文主题名称用于测试换行与论文图表显示')
        _,panel3,_=calculate(result,settings);self.assertTrue((panel3[panel3.topic_id.eq(0)].topic_label=='很长的中文主题名称用于测试换行与论文图表显示').all())
        original=result['visualization']['annual_heatmap'];result['visualization']['annual_heatmap']={**original,'years':[9999]}
        with self.assertRaises(ValueError):save_project(result)
        result['visualization'].pop('annual_heatmap');self.assertNotIn('annual_heatmap',open_project(save_project(result))['visualization'])

    def test_all_groups_respects_missing_group_filter(self):
        p,c=fixture();f=p['metadata']['generated_fields'];doc_id=p['rows'].iloc[3][f['document_id']]
        p['rows'].loc[p['rows'][f['document_id']].eq(doc_id),'年份']='2020'
        p['segments'].loc[p['segments'][f['document_id']].eq(doc_id),'年份']='2020'
        docs,panel,_=calculate(p,c);excluded=data(panel,'segment_count',ALL,[2020],[1]).iloc[0]
        self.assertEqual((excluded.numerator,excluded.denominator),(2,5))
        c['exclude_missing_group']=False;docs,panel,_=calculate(p,c);included=data(panel,'segment_count',ALL,[2020],[1]).iloc[0]
        self.assertEqual((included.numerator,included.denominator),(3,6))
        missing=group_key('（缺失分组）',True);stats=samples(p,docs,missing,[2020,2021])
        self.assertTrue(stats.group_missing.all());self.assertEqual(stats.iloc[1]['group'],'（缺失分组）')

    def test_export_values_plot_and_styles(self):
        result,settings,docs,panel,parts=prepared();rename(result,0,'很长的中文主题名称用于测试换行与论文图表显示')
        _,panel,_=calculate(result,settings);font,_=find_chinese_font();self.assertTrue(font)
        t=data(panel,'segment_share',group_key('上海'),[2020,2021],[0,1]);expected=t.pivot(index='topic_id',columns='year',values='plot_value').sort_index().to_numpy(dtype=float)
        plotted=[];original=Axes.imshow
        def spy(ax,arr,*args,**kwargs):
            plotted.append(np.ma.filled(arr,np.nan));return original(ax,arr,*args,**kwargs)
        for style in ['color','mono']:
            with patch.object(Axes,'imshow',spy):raw,_=render(t,10,6,150,font,{'style':style},'png')
            self.assertTrue(np.allclose(plotted[-1],expected,equal_nan=True));self.assertEqual(Image.open(io.BytesIO(raw)).size,(1500,900))
            svg,_=render(t,10,6,150,font,{'style':style},'svg');self.assertIn(b'<svg',svg)
        out=pd.read_csv(io.BytesIO(csv_bytes(t)),keep_default_na=False)
        for i,row in out.iterrows():
            self.assertEqual(int(row.numerator),int(t.iloc[i].numerator));self.assertEqual(int(row.denominator),int(t.iloc[i].denominator))
            self.assertEqual(row.missing_status,t.iloc[i].missing_status)
            if pd.isna(t.iloc[i].value):self.assertEqual(row.value,'')
            else:self.assertAlmostEqual(float(row.value),float(t.iloc[i].value))
        for metric in ('segment_count','segment_share','document_count','document_mention_rate'):
            cells=data(panel,metric,group_key('上海'),[2020,2021],[0,1])
            expected=panel[panel.group_key.eq(group_key('上海'))].set_index(['year','topic_id'])
            for row in cells.itertuples():
                want=expected.loc[(row.year,row.topic_id),metric]
                self.assertTrue(pd.isna(row.value) if row.denominator==0 else row.value==want)
        empty=data(panel,'segment_share',group_key('上海'),[2020],[0,1]);png,_=render(empty,10,6,150,font,fmt='png');self.assertGreater(len(png),1000)

    def test_page_flow_and_project_reopen(self):
        result,settings,docs,panel,parts=prepared();raw=save_project(result)
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=100).run();app.radio(key='work_mode').set_value('打开已有项目').run()
        app.file_uploader(key='project_upload').set_value(('test.hpts.zip',raw,'application/zip')).run(timeout=100)
        self.assertFalse(app.exception)
        self.assertTrue(any(x.label=='下载年度热力图数据 CSV' for x in app.get('download_button')))
        tables=[element.value for element in app.dataframe if 'sample_scope' in element.value.columns]
        self.assertEqual(len(tables),1);self.assertEqual(set(tables[0]['sample_scope']),{'全部纳入分组汇总','单个分组'})
        group=next(x for x in app.selectbox if x.label=='热力图分组');group.select(group_key('北京')).run(timeout=100)
        self.assertFalse(app.exception)
        metric=next(x for x in app.selectbox if x.label=='热力图指标');metric.select('document_mention_rate').run(timeout=100)
        selected_years=next(x for x in app.multiselect if x.label=='热力图年份');selected_years.set_value([2020]).run(timeout=100)
        selected_topics=next(x for x in app.multiselect if x.label=='热力图主题');selected_topics.set_value([0,1]).run(timeout=100)
        self.assertFalse(app.exception)
        saved=app.session_state['opened_project']['result'];self.assertEqual(saved['visualization']['annual_heatmap']['group_key'],group_key('北京'))
        self.assertEqual((saved['visualization']['annual_heatmap']['metric'],saved['visualization']['annual_heatmap']['years'],saved['visualization']['annual_heatmap']['topics']),('document_mention_rate',[2020],[0,1]))
        restored=open_project(save_project(saved));self.assertEqual(restored['visualization']['annual_heatmap'],saved['visualization']['annual_heatmap'])


if __name__=='__main__':unittest.main()
