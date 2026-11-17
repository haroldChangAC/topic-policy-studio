"""Per-analysis keyword tokenization; never changes semantic-model inputs."""
import importlib.metadata
import platform
import re
import unicodedata

from studio.runtime import ROOT

MAX_FILE = 1024 * 1024
MAX_WORDS = 10000
ENGLISH_TOKEN = re.compile(r"[a-z][a-z0-9]*(?:['’\-][a-z0-9]+)*|[0-9]+")


def parse_word_list(text):
    if not isinstance(text, str) or len(text.encode('utf-8')) > MAX_FILE:
        raise ValueError('词表必须是 UTF-8 文本且不超过 1 MiB。')
    words = []
    seen = set()
    for line in text.splitlines():
        word = line.strip()
        if not word:
            continue
        if len(word) > 100 or any(c.isspace() or unicodedata.category(c).startswith('C') for c in word):
            raise ValueError('每行只能填写一个词，词内不能含空白或控制字符，最多 100 个字符。')
        if word not in seen:
            words.append(word)
            seen.add(word)
    if len(words) > MAX_WORDS:
        raise ValueError('词表最多包含 10000 个词。')
    return words


def decode_word_file(raw):
    if len(raw) > MAX_FILE:
        raise ValueError('词表文件不能超过 1 MiB。')
    try:
        return raw.decode('utf-8-sig', errors='strict')
    except UnicodeError as exc:
        raise ValueError('词表文件必须使用 UTF-8 编码。') from exc


def make_settings(mode='legacy', stopwords_enabled=False, stopwords=None, dictionary=None):
    if mode not in ('legacy', 'chinese', 'english') or not isinstance(stopwords_enabled, bool):
        raise ValueError('关键词分词设置无效。')
    stopwords = parse_word_list('\n'.join(stopwords or []))
    dictionary = parse_word_list('\n'.join(dictionary or []))
    if mode == 'english':
        if dictionary:
            raise ValueError('英文分词不使用自定义词典。')
        return {
            'mode': mode, 'component': 'python-re', 'component_version': platform.python_version(),
            'stopwords_enabled': stopwords_enabled, 'stopwords': stopwords,
            'stopwords_applied': stopwords if stopwords_enabled else [], 'user_dictionary': [],
            'dictionary_frequency': None, 'hmm': False, 'vectorizer_lowercase': True, 'min_df': 1,
            'semantic_input': 'Original segments before keyword tokenization and stopword filtering',
            'keyword_preprocessing': 'BERTopic replaces line feeds and tabs with spaces. CountVectorizer lowercases text.',
            'token_rule': 'ASCII English words and numbers, with internal apostrophes and hyphens',
        }
    return {'mode': mode, 'component': 'jieba', 'component_version': importlib.metadata.version('jieba'),
            'stopwords_enabled': stopwords_enabled, 'stopwords': stopwords,
            'stopwords_applied': stopwords if stopwords_enabled else [], 'user_dictionary': dictionary,
            'dictionary_frequency': 1000000000, 'hmm': True,
            'vectorizer_lowercase': True, 'min_df': 1,
            'semantic_input': '未经关键词分词或停用词处理的原始文本片段',
            'keyword_preprocessing': 'BERTopic 将换行和制表符替换为空格；CountVectorizer 小写化；不转换简繁体',
            'token_rule': '旧方式：去除首尾空白，保留长度大于 1 的词' if mode == 'legacy' else
                          '中文分词：去除首尾空白，保留含字母或数字的词（包括单字），过滤标点'}


class KeywordTokenizer:
    def __init__(self, settings):
        import jieba
        self.engine = jieba.Tokenizer()
        self.engine.tmp_dir = str(ROOT / '.cache')
        self.engine.initialize()
        for word in settings['user_dictionary']:
            self.engine.add_word(word.lower(), freq=settings['dictionary_frequency'])
        self.mode = settings['mode']
        self.stopwords = {word.lower() for word in settings['stopwords_applied']}

    def __call__(self, text):
        words = [token.strip() for token in self.engine.lcut(text, HMM=True)]
        if self.mode == 'legacy':
            words = [word for word in words if len(word) > 1]
        else:
            words = [word for word in words if word and any(c.isalnum() for c in word)]
        return [word for word in words if word not in self.stopwords]


class EnglishKeywordTokenizer:
    def __init__(self, stopwords):
        self.stopwords = {word.lower() for word in stopwords}

    def __call__(self, text):
        return [word for word in ENGLISH_TOKEN.findall(text) if word not in self.stopwords]


def vectorizer_for(settings):
    from sklearn.feature_extraction.text import CountVectorizer
    # The untouched default remains the exact existing global jieba path.
    if settings['mode'] == 'english':
        tokenizer = EnglishKeywordTokenizer(settings['stopwords_applied'])
    elif settings['mode'] == 'legacy' and not settings['user_dictionary'] and not settings['stopwords_applied']:
        from studio.modeling import tokenize_zh
        tokenizer = tokenize_zh
    else:
        tokenizer = KeywordTokenizer(settings)
    return CountVectorizer(tokenizer=tokenizer, token_pattern=None, min_df=1)


def keyword_input(text):
    """Matches BERTopic 0.17.4 preprocessing when an embedding model is supplied."""
    return text.replace('\n', ' ').replace('\t', ' ')


def preview(text, settings):
    return vectorizer_for(settings).build_analyzer()(keyword_input(text))
