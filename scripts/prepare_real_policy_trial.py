"""Reproduce the local 2022–2024 government-report trial CSV from saved HTML.

Source pages must be downloaded separately; no network access is performed here.
The CSV text is an HTML-to-text extraction, not byte-identical to the web page.
Original HTML is retained alongside the data for checking the extraction.
"""
import csv
import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs/real_policy_2022_2024'
SOURCES = [
    ('beijing', 2022, '北京市', 'https://www.beijing.gov.cn/gongkai/jihua/zfgzbg/202201/t20220113_2589854.html'),
    ('beijing', 2023, '北京市', 'https://www.beijing.gov.cn/gongkai/jihua/zfgzbg/202301/t20230128_2907344.html'),
    ('beijing', 2024, '北京市', 'https://www.beijing.gov.cn/gongkai/jihua/zfgzbg/202401/t20240129_3547363.html'),
    ('shanghai', 2022, '上海市', 'https://www.shanghai.gov.cn/nw12336/20220125/c1905a92364e418f96bb785c5ce97f5e.html'),
    ('shanghai', 2023, '上海市', 'https://www.shanghai.gov.cn/nw12336/20230117/b511b08dd4e54a13bc592fed41ce2510.html'),
    ('shanghai', 2024, '上海市', 'https://www.shanghai.gov.cn/nw12336/20240129/2dfb672cc8b840f1b97788f74fb403cd.html'),
]

class Page(HTMLParser):
    def __init__(self, city):
        super().__init__(convert_charrefs=True)
        self.city, self.depth, self.fragments, self.meta = city, 0, [], {}
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'meta' and a.get('name') in ('ArticleTitle', 'PubDate', 'ContentSource'):
            self.meta[a['name']] = a.get('content', '')
        target = tag == 'div' and ((self.city == 'beijing' and 'TRS_UEDITOR' in a.get('class', '').split()) or (self.city == 'shanghai' and a.get('id') == 'ivs_content'))
        if self.depth:
            if tag == 'div': self.depth += 1
            if tag in ('p', 'br', 'h1', 'h2', 'h3', 'h4', 'tr'): self.fragments.append('\n')
        elif target:
            self.depth = 1
    def handle_endtag(self, tag):
        if self.depth:
            if tag in ('p', 'h1', 'h2', 'h3', 'h4', 'tr'): self.fragments.append('\n')
            if tag == 'div': self.depth -= 1
    def handle_data(self, data):
        if self.depth: self.fragments.append(data)
    def body(self):
        text = ''.join(self.fragments)
        # Keep only the address and report body: the Beijing div includes a photo caption.
        index = text.find('各位代表')
        if index < 0: raise ValueError('正文起始标记缺失')
        text = text[index:]
        # HTML formatting whitespace between paragraphs is not report content.
        text = '\n'.join(line.strip(' \t\r') for line in text.splitlines())
        return re.sub(r'\n{2,}', '\n', text).strip('\n')

def main():
    rows, provenance = [], []
    for city, year, group, url in SOURCES:
        path = OUT / 'raw' / f'{city}_{year}.html'
        raw = path.read_bytes()
        parser = Page(city)
        parser.feed(raw.decode('utf-8'))
        body = parser.body()
        meta = parser.meta
        if not all(meta.get(k) for k in ('ArticleTitle', 'PubDate', 'ContentSource')) or not meta['PubDate'].startswith(str(year)) or len(body) < 3000:
            raise ValueError(f'内容或来源元数据异常：{path.name}')
        item = {'文档ID':f'{city.upper()}-{year}', '标题':meta['ArticleTitle'], '正文':body,
                '年份':str(year), '分组':group, '官网刊载日期':meta['PubDate'],
                '网站注明来源':meta['ContentSource'], '来源网址':url}
        rows.append(item)
        provenance.append({'document_id':item['文档ID'], 'title':item['标题'], 'year':year,
                           'group':group, 'publication_time':meta['PubDate'], 'credited_source':meta['ContentSource'],
                           'url':url, 'raw_file':str(path.relative_to(OUT)),
                           'raw_sha256':hashlib.sha256(raw).hexdigest(), 'text_characters':len(body),
                           'text_sha256':hashlib.sha256(body.encode()).hexdigest()})
    with (OUT / 'real_reports.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (OUT/'sources.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for r in provenance: print(r['document_id'],r['publication_time'],r['credited_source'],r['text_characters'],r['text_sha256'][:12])

if __name__ == '__main__': main()
