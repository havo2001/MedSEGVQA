"""
Lightweight, dependency-free text metrics for VQA evaluation:
  - token-level F1   (SQuAD-style)
  - BLEU-1           (unigram precision with brevity penalty)
  - ROUGE-L (F1)     (LCS-based)

All three operate on a single (prediction, reference) string pair and
return floats in [0, 1]. Tokenization is whitespace + simple punctuation
stripping + lowercasing — appropriate for the short descriptive answers
in this dataset (e.g. "One small circle-shaped benign tumor").
"""

import math
import re
from collections import Counter

_PUNCT = re.compile(r"[^\w\s\-]")


def tokenize(text: str) -> list:
    """Lowercase, strip punctuation (keep hyphens), split on whitespace."""
    if text is None:
        return []
    text = text.lower().strip()
    text = _PUNCT.sub(" ", text)
    return text.split()


def f1_score(pred: str, ref: str) -> float:
    """Token-level F1 between prediction and reference."""
    p_toks = tokenize(pred)
    r_toks = tokenize(ref)
    if not p_toks and not r_toks:
        return 1.0
    if not p_toks or not r_toks:
        return 0.0
    common = Counter(p_toks) & Counter(r_toks)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p_toks)
    recall = overlap / len(r_toks)
    return 2 * precision * recall / (precision + recall)


def bleu1_score(pred: str, ref: str) -> float:
    """BLEU-1: unigram precision with brevity penalty."""
    p_toks = tokenize(pred)
    r_toks = tokenize(ref)
    if not p_toks and not r_toks:
        return 1.0
    if not p_toks or not r_toks:
        return 0.0
    ref_counts = Counter(r_toks)
    matched = 0
    pred_counts = Counter(p_toks)
    for tok, cnt in pred_counts.items():
        matched += min(cnt, ref_counts.get(tok, 0))
    precision = matched / len(p_toks)
    # brevity penalty
    bp = 1.0 if len(p_toks) >= len(r_toks) else math.exp(1.0 - len(r_toks) / len(p_toks))
    return bp * precision


def _lcs_len(a: list, b: list) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l_score(pred: str, ref: str) -> float:
    """ROUGE-L F1 based on longest common subsequence."""
    p_toks = tokenize(pred)
    r_toks = tokenize(ref)
    if not p_toks and not r_toks:
        return 1.0
    if not p_toks or not r_toks:
        return 0.0
    lcs = _lcs_len(p_toks, r_toks)
    if lcs == 0:
        return 0.0
    precision = lcs / len(p_toks)
    recall = lcs / len(r_toks)
    return 2 * precision * recall / (precision + recall)


def compute_all(pred: str, ref: str) -> dict:
    """Return {bleu1, rouge_l, f1} for one (pred, ref) pair."""
    return {
        "bleu1":   bleu1_score(pred, ref),
        "rouge_l": rouge_l_score(pred, ref),
        "f1":      f1_score(pred, ref),
    }
