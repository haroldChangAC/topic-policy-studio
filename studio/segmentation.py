"""Token-bounded original-text spans; zero-based, end-exclusive Unicode offsets."""
import hashlib
import re
from dataclasses import dataclass

import pandas as pd

RULE_VERSION = "paragraph-sentence-character-v1"
BOUNDARIES = (re.compile(r"(?:\r\n|\r|\n)+[ \t]*"),
              re.compile(r'''(?:[。！？!?；;]+|\.(?=\s|$))[”’"')）】]*\s*'''))


class TokenBudget:
    def __init__(self, encoder):
        self.encoder = encoder
        self.tokenizer = encoder.tokenizer
        self.limits = {
            "sentence_transformer": int(encoder.max_seq_length),
            "tokenizer": int(self.tokenizer.model_max_length),
            "position_embeddings": int(encoder[0].auto_model.config.max_position_embeddings),
        }
        self.limit = min(value for value in self.limits.values() if value > 0)
        self.special_tokens = self.tokenizer.num_special_tokens_to_add(pair=False)
        if self.limit <= self.special_tokens:
            raise ValueError("模型没有可用于正文的 token 容量。")
        prompt = getattr(encoder, "default_prompt_name", None)
        if prompt and encoder.prompts.get(prompt):
            raise ValueError("当前模型启用了额外提示词，需要重新核对长度预算。")

    def ids(self, text):
        return self.tokenizer.encode(text, add_special_tokens=True, truncation=False, verbose=False)

    def count(self, text):
        return len(self.ids(text))

    def metadata(self):
        return {"effective_max_tokens": self.limit, "special_tokens": self.special_tokens,
                "limits": self.limits, "length_includes_special_tokens": True,
                "tokenizer_class": type(self.tokenizer).__name__}

    def verify_model_inputs(self, texts):
        # Verify the actual encoder preprocessing does not truncate, add a prompt,
        # or otherwise produce a different token sequence from the measured input.
        for start in range(0, len(texts), 16):
            batch = texts[start:start + 16]
            features = self.encoder.tokenize(batch)
            for text, ids, mask in zip(batch, features["input_ids"], features["attention_mask"]):
                actual = ids[mask.bool()].tolist()
                expected = self.ids(text)
                if len(expected) > self.limit or actual != expected:
                    raise ValueError("实际模型输入与未截断 token 序列不一致，已停止建模。")


def split_text(text, budget):
    """Cover every code point exactly once, keeping whitespace and delimiters."""
    if not text.strip():
        return []

    def fits(start, end):
        return budget.count(text[start:end]) <= budget.limit

    def character_spans(start, end):
        output = []
        while start < end:
            # Exponential probing + bounded search. Only verified fitting endpoints
            # are accepted; tokenizer prefix length need not be perfectly monotonic.
            low = start
            high = min(end, start + budget.limit)
            while fits(start, high):
                low = high
                if high == end:
                    break
                high = min(end, start + 2 * (high - start))
            if low < end:
                while high - low > 1:
                    mid = (low + high) // 2
                    if fits(start, mid):
                        low = mid
                    else:
                        high = mid
            if low == start:
                raise ValueError("单个字符超过模型容量，无法进行无损切分。")
            output.append((start, low))
            start = low
        return output

    def split(start, end, level):
        if fits(start, end):
            return [(start, end)]
        if level == len(BOUNDARIES):
            return character_spans(start, end)
        points = [start]
        for match in BOUNDARIES[level].finditer(text, start, end):
            point = match.end()
            if text[points[-1]:point].strip():
                points.append(point)
        if points[-1] < end:
            if not text[points[-1]:end].strip() and len(points) > 1:
                points[-1] = end
            else:
                points.append(end)
        parts = []
        for a, b in zip(points, points[1:]):
            parts.extend(split(a, b, level + 1))
        packed = []
        for a, b in parts:
            if packed and fits(packed[-1][0], b):
                packed[-1] = (packed[-1][0], b)
            else:
                packed.append((a, b))
        return packed

    spans = split(0, len(text), 0)
    # Attach whitespace-only trailing spans to their predecessor without dropping
    # characters. If that cannot fit, fail explicitly rather than truncate.
    merged = []
    for a, b in spans:
        if not text[a:b].strip() and merged and fits(merged[-1][0], b):
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    cursor = 0
    for a, b in merged:
        if a != cursor or b <= a or not text[a:b].strip() or not fits(a, b):
            raise ValueError("片段覆盖或长度校验失败，未丢弃任何原文。")
        cursor = b
    if cursor != len(text):
        raise ValueError("片段未完整覆盖原文。")
    return merged


@dataclass
class SegmentPlan:
    documents: pd.DataFrame
    segments: pd.DataFrame
    counts: dict
    settings: dict
    blocked: bool


def plan_segments(prepared, encoder, mode="stop"):
    if mode not in ("stop", "auto"):
        raise ValueError("未知的长文本处理方式。")
    budget = TokenBudget(encoder)
    f = prepared.fields
    documents = prepared.rows.copy(deep=True)
    documents[f["document_id"]] = documents[f["record_id"]]
    documents[f["token_count"]] = pd.Series(pd.NA, index=documents.index, dtype="Int64")
    documents[f["segment_count"]] = 0
    documents[f["is_long"]] = False
    selected = prepared.selected
    for index in documents.index[selected]:
        count = budget.count(documents.at[index, f["original_text"]])
        documents.at[index, f["token_count"]] = count
        documents.at[index, f["is_long"]] = count > budget.limit
    long_count = int(documents[f["is_long"]].sum())
    blocked = mode == "stop" and long_count > 0
    records = []
    if not blocked:
        for index in documents.index[selected]:
            original = documents.at[index, f["original_text"]]
            spans = split_text(original, budget) if mode == "auto" else [(0, len(original))]
            documents.at[index, f["segment_count"]] = len(spans)
            for order, (start, end) in enumerate(spans, 1):
                row = documents.loc[index].to_dict()
                identity = f"{row[f['document_id']]}:{prepared.text_column}:{RULE_VERSION}:{start}:{end}"
                row.update({f["segment_id"]: hashlib.sha256(identity.encode()).hexdigest(),
                            f["segment_order"]: order, f["start_char"]: start, f["end_char"]: end,
                            f["segment_text"]: original[start:end], f["token_count"]: budget.count(original[start:end])})
                records.append(row)
    columns = list(documents.columns) + [f[key] for key in ("segment_id", "segment_order", "start_char", "end_char", "segment_text")]
    segments = pd.DataFrame(records, columns=columns)
    counts = {"valid_documents": int(selected.sum()), "long_documents": long_count,
              "segments": len(segments),
              "unique_segments": segments[f["segment_text"]].nunique() if len(segments) else 0}
    settings = {"mode": mode, "rule_version": RULE_VERSION, "overlap_characters": 0,
                "boundary_priority": ["paragraph", "sentence", "character"],
                "offset_convention": "0-based Unicode code points, end exclusive; original_text[start:end]",
                **budget.metadata()}
    return SegmentPlan(documents, segments, counts, settings, blocked)
