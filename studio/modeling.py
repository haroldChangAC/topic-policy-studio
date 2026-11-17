"""Offline BERTopic backend shared by the workbench and smoke verification."""
import json
import time
from pathlib import Path

from studio.runtime import ROOT, configure
from studio.segmentation import TokenBudget


def model_config():
    return json.loads((ROOT / "models/model-lock.json").read_text(encoding="utf-8"))


def load_encoder(device="cpu"):
    configure()
    import torch
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(4)
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("当前进程无法使用 Apple GPU，请选择 CPU。")
    config = model_config()
    try:
        snapshot = snapshot_download(repo_id=config["model_id"], revision=config["revision"],
                                     allow_patterns=config["files"], local_files_only=True)
        for name in config["files"]:
            if not (Path(snapshot) / name).is_file():
                raise FileNotFoundError(name)
    except (OSError, ValueError) as exc:
        raise ValueError("本地模型不完整，请先运行 bash scripts/download_model.sh。") from exc
    return SentenceTransformer(snapshot, device=device, local_files_only=True, trust_remote_code=False)


def tokenize_zh(text):
    import jieba
    jieba.dt.tmp_dir = str(ROOT / ".cache")
    return [token.strip() for token in jieba.lcut(text) if len(token.strip()) > 1]


def default_clustering_settings():
    """Explicit user-facing defaults; supported inputs always have at least 10 segments."""
    return {"n_neighbors": 8, "min_dist": 0.0, "min_cluster_size": 4, "min_samples": 2}


def validate_clustering_settings(settings, sample_count):
    requested = default_clustering_settings() if settings is None else dict(settings)
    if set(requested) != set(default_clustering_settings()):
        raise ValueError("聚类参数不完整或包含不支持的选项。")
    for key, label in (("n_neighbors", "UMAP 邻居数"), ("min_cluster_size", "最小聚类规模"),
                       ("min_samples", "HDBSCAN 最小样本数")):
        if isinstance(requested[key], bool) or not isinstance(requested[key], int):
            raise ValueError(f"{label}必须是整数。")
    if isinstance(requested['min_dist'], bool) or not isinstance(requested['min_dist'], (int, float)):
        raise ValueError("UMAP 最小距离必须是数值。")
    if not 2 <= requested['n_neighbors'] < sample_count:
        raise ValueError(f"UMAP 邻居数须至少为 2，且小于参与建模的片段数 {sample_count}。")
    if not 0 <= requested['min_dist'] <= 1:
        raise ValueError("UMAP 最小距离须在 0 到 1 之间（当前 spread 为 1）。")
    if not 2 <= requested['min_cluster_size'] <= sample_count:
        raise ValueError(f"最小聚类规模须在 2 至片段数 {sample_count} 之间。")
    if not 1 <= requested['min_samples'] < sample_count:
        raise ValueError(f"HDBSCAN 最小样本数须至少为 1，且小于片段数 {sample_count}。")
    requested['min_dist'] = float(requested['min_dist'])
    return requested


def fit_topics(docs, encoder, progress=lambda message: None, keyword_settings=None, clustering_settings=None):
    configure()
    import numpy as np
    import torch
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP
    from studio.keywords import keyword_input, make_settings, vectorizer_for

    requested = keyword_settings or {}
    settings = make_settings(requested.get('mode', 'legacy'), requested.get('stopwords_enabled', False),
                             requested.get('stopwords', []), requested.get('user_dictionary', []))

    if len(docs) < 10 or len(set(docs)) < 5:
        raise ValueError("本轮建模至少需要 10 条有效文本，且至少有 5 种不同文本。")
    requested_clustering = validate_clustering_settings(clustering_settings, len(docs))
    if any(not text.strip() for text in docs):
        raise ValueError("空文本不能送入模型。")
    budget = TokenBudget(encoder)
    lengths = [budget.count(text) for text in docs]
    too_long = [i + 1 for i, length in enumerate(lengths) if length > budget.limit]
    if too_long:
        raise ValueError(f"{len(too_long)} 条有效文本超过模型 {budget.limit} token 上限；"
                         f"有效文本序号示例：{too_long[:8]}。请先切分为较短记录，本次未截断或建模。")
    vectorizer = vectorizer_for(settings)
    try:
        vectorizer.fit([keyword_input(doc) for doc in docs])
    except ValueError as exc:
        raise ValueError("没有可提取的关键词：整个语料经过当前分词和停用词过滤后没有有效词，请调整设置或检查正文。") from exc
    parameters = {
        "random_state": 42,
        "umap": {"n_neighbors": requested_clustering['n_neighbors'], "n_components": 5,
                 "min_dist": requested_clustering['min_dist'], "metric": "cosine", "n_jobs": 1},
        "hdbscan": {"min_cluster_size": requested_clustering['min_cluster_size'], "min_samples": requested_clustering['min_samples'], "metric": "euclidean", "prediction_data": True},
        "vectorizer": {"tokenizer": {
            'legacy': 'jieba.lcut; retain tokens longer than one character',
            'chinese': 'isolated jieba; include single alphanumeric tokens',
            'english': settings['token_rule'],
        }[settings['mode']], "min_df": 1},
        "calculate_probabilities": False,
    }
    np.random.seed(42)
    torch.manual_seed(42)
    progress("正在本地计算文本嵌入……")
    budget.verify_model_inputs(docs)
    start = time.perf_counter()
    embeddings = encoder.encode(docs, batch_size=16, show_progress_bar=False, convert_to_numpy=True)
    if embeddings.shape != (len(docs), model_config()["expected_embedding_dimension"]) or not np.isfinite(embeddings).all():
        raise ValueError("模型输出的嵌入形状或数值异常。")
    embedding_seconds = time.perf_counter() - start
    progress("正在降维、聚类和提取主题关键词……")
    model = BERTopic(embedding_model=encoder,
                     umap_model=UMAP(**parameters["umap"], random_state=42),
                     hdbscan_model=HDBSCAN(**parameters["hdbscan"]),
                     vectorizer_model=vectorizer, calculate_probabilities=False)
    start = time.perf_counter()
    topics, strengths = model.fit_transform(docs, embeddings=embeddings)
    fit_seconds = time.perf_counter() - start
    if len(topics) != len(docs):
        raise ValueError("主题分配条数与输入不一致，结果未导出。")
    # BERTopic 0.17.4 creates topic_embeddings_ in sorted topic-ID order from
    # the mean of assigned input embeddings (including the -1 group, if any).
    topic_ids = sorted(set(int(topic) for topic in topics))
    raw_topic_vectors = getattr(model, 'topic_embeddings_', None)
    expected = (len(topic_ids), embeddings.shape[1])
    if raw_topic_vectors is not None and np.asarray(raw_topic_vectors).shape == expected:
        raw_topic_vectors = np.asarray(raw_topic_vectors)
        topic_vectors = [{'topic_id': topic, 'values': row.astype(float).tolist() if np.isfinite(row).all() else None}
                         for topic, row in zip(topic_ids, raw_topic_vectors)]
    else:
        # Without a complete aligned array the row-to-topic mapping is unknown.
        topic_vectors = [{'topic_id': topic, 'values': None} for topic in topic_ids]
    info = model.get_topic_info()
    representations = {int(topic): model.get_topic(int(topic)) for topic in set(topics)}
    keywords = {topic: "、".join(word for word, _ in pairs if word.strip()) for topic, pairs in representations.items()}
    empty_topics = sorted(topic for topic, words in keywords.items() if not words)
    for topic in empty_topics:
        info.loc[info['Topic'].eq(topic), 'Name'] = f'{topic}_无有效关键词'
    keyword_weights = [{'topic_id': topic, 'rank': rank, 'keyword': word, 'weight': float(weight)}
                       for topic, pairs in representations.items() for rank, (word, weight) in enumerate(pairs, 1)]
    info["Keywords"] = info["Topic"].map(keywords)
    strengths = np.asarray(strengths) if strengths is not None else None
    if strengths is not None and (strengths.shape != (len(docs),) or not np.isfinite(strengths).all()):
        strengths = None
    return {"topics": topics, "strengths": strengths, "topic_info": info, "keywords": keywords,
            "embeddings": embeddings, "keyword_weights": keyword_weights, "topic_representations": representations, "parameters": parameters, "clustering_settings_requested": requested_clustering, "max_input_tokens": max(lengths),
            "model_token_limit": budget.limit, "keyword_settings": settings, "empty_keyword_topics": empty_topics,
            "topic_vectors": topic_vectors, "topic_vector_dimension": embeddings.shape[1],
            "timings_seconds": {"embedding": round(embedding_seconds, 3), "bertopic_fit": round(fit_seconds, 3)}}
