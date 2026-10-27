"""Every table and figure of the paper, from the finished runs of the paper protocol (CPU only).

    python paper_analysis.py --root <root> [--prefix paper] [--output <root>/paper]

main.py runs this at the end of every paper-protocol job; it can also be run by hand at any time. It only
reads the runs' prediction files (results/<run>/proposed_<set>.csv), their reference labels and validation
thresholds, so it never changes a run, and every number can be recomputed from the files it names. Runs that
have not finished yet are listed as missing; the analysis is simply redone when they have.

Protocol (the same for every method):
  * test = the 32 standard Cholec80 test videos; every sampled frame (1 fps) x 14 queries; frames without an
    annotation (-100) are left out, never counted as negatives;
  * the operating threshold of each method is the one selected on the validation videos (evaluate.py);
  * 95 % confidence intervals: percentile bootstrap over test videos (resampled with replacement, the same
    resamples for every method, so differences are paired); p-values: paired bootstrap (two-sided) and the
    Wilcoxon signed-rank test on per-video AP; Holm correction within each table;
  * segments: runs of frames at or above the threshold (the same post-processing for every method);
    ground truth: runs of annotated positive frames; unannotated frames are excluded from both;
  * phase recognition: per frame, the phase query with the highest score; the standard (non-relaxed) Cholec80
    metrics - accuracy per video, then precision / recall / Jaccard per phase averaged over videos;
  * tool presence: AP per tool over all test frames, and their mean (mAP).

Writes <output>/PAPER_RESULTS.md, tables/*.csv and *.tex, figures/*.png and *.pdf, analysis.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import paper_protocol as protocol  # noqa: E402
from metrics import binary_metrics, temporal_iou  # noqa: E402

VERSION = 'paper-analysis-1'
KEYS = ['video_id', 'text_query', 'frame_idx']
IOUS = (0.3, 0.5, 0.7)
BINS = 15
# Main-table order: the comparison methods (weakest assumptions first), then ours.
MAIN_ORDER = ('clip_zeroshot', 'peskavlp_zeroshot', 'xclip_finetune', 'peskavlp_finetune', 'univtg',
              'baseline', 'advanced')


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', str(s))]


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()[:16]


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# --------------------------------------------------------------------------- weighted metrics (bootstrap)
def tie_ends(sorted_scores):
    """Index of the last element of every group of equal scores (scores sorted in descending order)."""
    return np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), len(sorted_scores) - 1]


def weighted_ap(y, w, ends):
    """Average precision with sample weights, on data already sorted by descending score (ends = tie_ends).
    Identical to sklearn.metrics.average_precision_score(y, score, sample_weight=w)."""
    tp = np.cumsum(w * y)[ends]
    fp = np.cumsum(w * (1 - y))[ends]
    if tp[-1] <= 0:
        return math.nan
    total = tp + fp
    precision = np.divide(tp, total, out=np.zeros_like(tp, dtype=np.float64), where=total > 0)
    recall = tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def weighted_auroc(y, w, ends):
    """ROC AUC with sample weights (trapezoids at distinct thresholds), as sklearn.metrics.roc_auc_score."""
    tp = np.r_[0.0, np.cumsum(w * y)[ends]]
    fp = np.r_[0.0, np.cumsum(w * (1 - y))[ends]]
    if tp[-1] <= 0 or fp[-1] <= 0:
        return math.nan
    return float(np.trapezoid(tp / tp[-1], fp / fp[-1]) if hasattr(np, 'trapezoid') else np.trapz(tp / tp[-1], fp / fp[-1]))


class Ranked:
    """One method's scores on a set of rows, sorted once by descending score, for fast weighted metrics."""
    def __init__(self, y, scores, video):
        order = np.argsort(-scores, kind='stable')
        self.y = y[order].astype(np.float64)
        self.video = video[order]
        self.ends = tie_ends(scores[order])

    def ap(self, video_weights=None):
        w = np.ones_like(self.y) if video_weights is None else video_weights[self.video].astype(np.float64)
        return weighted_ap(self.y, w, self.ends)

    def auroc(self, video_weights=None):
        w = np.ones_like(self.y) if video_weights is None else video_weights[self.video].astype(np.float64)
        return weighted_auroc(self.y, w, self.ends)


def percentile_ci(values, level=0.95):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return (math.nan, math.nan)
    lo, hi = np.percentile(values, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return (float(lo), float(hi))


def bootstrap_p(differences):
    """Two-sided p-value of a paired bootstrap: how often the resampled difference falls on the other side of 0."""
    d = np.asarray(differences, dtype=np.float64)
    d = d[np.isfinite(d)]
    if not len(d):
        return math.nan
    below, above = (np.sum(d <= 0) + 1) / (len(d) + 1), (np.sum(d >= 0) + 1) / (len(d) + 1)
    return float(min(1.0, 2 * min(below, above)))


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values (NaN entries are left out of the family)."""
    p = np.asarray(pvalues, dtype=np.float64)
    out = np.full_like(p, math.nan)
    idx = [i for i in range(len(p)) if np.isfinite(p[i])]
    order = sorted(idx, key=lambda i: p[i])
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p[i]))
        out[i] = running
    return out


# --------------------------------------------------------------------------- segments
def runs(times, ends, positive, values, merge_gap=0.0, min_duration=0.0):
    """[start, end, max value] of every run of positive cells (cells [times[i], ends[i])), merging runs that
    are at most merge_gap apart and dropping runs shorter than min_duration (as metrics.segments_from_scores)."""
    groups = []
    for i in np.flatnonzero(positive):
        a, b = float(times[i]), float(ends[i])
        if groups and a - groups[-1][1] <= merge_gap + 1e-9:
            groups[-1][1] = b
            groups[-1][2] = max(groups[-1][2], float(values[i]))
        else:
            groups.append([a, b, float(values[i])])
    return [g for g in groups if g[1] - g[0] >= min_duration]


def segment_pairs(frames, scores, threshold, meta, min_duration, merge_gap, native=None):
    """Per video-query pair: ground-truth segments (runs of annotated positive frames), predicted segments
    (runs of annotated frames scoring at least the threshold, scored by their highest frame score - or the
    method's own windows) and the frame IoU. Unannotated frames are excluded from both."""
    pairs = []
    for (start, stop) in frames.groups:
        video = frames.video_ids[frames.video[start]]
        query = frames.queries[frames.query[start]]
        m = meta[video]
        fps = float(m['source_fps'])
        duration = m['frame_count'] / fps
        times = frames.frame_idx[start:stop] / fps
        ends = np.r_[times[1:], duration]
        labels = frames.label[start:stop]
        s = scores[start:stop]
        observed = labels != -100
        positive = observed & (s >= threshold)
        gt = [g[:2] for g in runs(times, ends, labels == 1, np.ones(len(times)))]
        if native is None:
            pred = runs(times, ends, positive, s, merge_gap, min_duration)
        else:
            pred = [[float(w[0]), float(w[1]), float(w[2])] for w in native[(video, query)]]
        truth = labels == 1
        frame_iou = float(np.sum(positive & truth) / np.sum(positive | truth)) if gt else None
        if native is not None:
            frame_iou = None                         # a frame-score measure: not defined for a method's own windows
        pairs.append(dict(video=int(frames.video[start]), key=str((video, query)), gt=gt, pred=pred, frame_iou=frame_iou))
    return pairs


def pair_hits(pred, gt, t):
    """Greedy matching of metrics.temporal_ap within one pair: predictions by descending score (ties: list
    order), each to the unused ground-truth segment of highest tIoU (ties: the later one)."""
    hit = np.zeros(len(pred), dtype=bool)
    if not pred or not gt:
        return hit
    P, G = np.asarray(pred, dtype=np.float64)[:, :2], np.asarray(gt, dtype=np.float64)
    inter = np.clip(np.minimum(P[:, None, 1], G[None, :, 1]) - np.maximum(P[:, None, 0], G[None, :, 0]), 0, None)
    union = (P[:, 1] - P[:, 0])[:, None] + (G[:, 1] - G[:, 0])[None, :] - inter
    iou = inter / np.maximum(union, 1e-12)
    used = np.zeros(len(G), dtype=bool)
    for j in sorted(range(len(pred)), key=lambda j: (-pred[j][2], j)):
        row = np.where(used, -1.0, iou[j])
        i = len(row) - 1 - int(np.argmax(row[::-1]))
        if row[i] >= t:
            used[i] = True
            hit[j] = True
    return hit


class SegmentMatch:
    """The matching of one method's segments, done once; metrics() re-weights it per bootstrap resample."""
    def __init__(self, pairs, threshold, n_videos):
        self.n_videos = n_videos
        self.pair_video = np.asarray([p['video'] for p in pairs], dtype=np.int64)
        self.n_gt = np.asarray([len(p['gt']) for p in pairs], dtype=np.float64)
        self.has_gt = self.n_gt > 0
        rows = sorted(((-pred[2], p['key'], j, k) for k, p in enumerate(pairs) for j, pred in enumerate(p['pred'])))
        self.order_video = np.asarray([pairs[k]['video'] for *_, k in rows], dtype=np.int64)
        self.hits = {}
        for t in IOUS:
            per_pair = [pair_hits(p['pred'], p['gt'], t) for p in pairs]
            self.hits[t] = np.asarray([per_pair[k][j] for _, _, j, k in rows], dtype=np.float64)
        top = []
        for p in pairs:
            best = max(p['pred'], key=lambda x: x[2], default=None)
            top.append(max((temporal_iou(best, g) for g in p['gt']), default=0.0) if (best and p['gt']) else 0.0)
        self.top_overlap = np.asarray(top)
        self.absent_alarm = np.asarray([any(x[2] >= threshold for x in p['pred']) for p in pairs], dtype=np.float64)
        self.frame_iou = np.asarray([p['frame_iou'] if p['frame_iou'] is not None else math.nan for p in pairs])

    def metrics(self, weights=None):
        w = np.ones(self.n_videos) if weights is None else np.asarray(weights, dtype=np.float64)
        out = {}
        total = float(np.sum(w[self.pair_video] * self.n_gt))
        pw = w[self.order_video]
        for t in IOUS:
            out[f'mAP@{t}'] = pooled_ap(self.hits[t], pw, total)
        wg = w[self.pair_video]

        def wmean(values, mask):
            ws = wg[mask]
            return float(np.sum(values[mask] * ws) / ws.sum()) if mask.any() and ws.sum() > 0 else math.nan
        for t in IOUS:
            out[f'R1@{t}'] = wmean((self.top_overlap >= t).astype(np.float64), self.has_gt)
        out['mAP'] = float(np.nanmean([out[f'mAP@{t}'] for t in IOUS]))
        out['mIoU'] = wmean(self.top_overlap, self.has_gt)
        out['frame_IoU'] = wmean(self.frame_iou, self.has_gt & np.isfinite(self.frame_iou))
        out['absent_false_alarm'] = wmean(self.absent_alarm, ~self.has_gt)
        out['absent_pairs'] = int((~self.has_gt).sum())
        return out


def pooled_ap(hits, weights, total):
    """metrics.temporal_ap with weights, on predictions already in its ranking order."""
    if total <= 0:
        return math.nan
    if not len(hits):
        return 0.0
    tp = np.cumsum(weights * hits)
    n = np.cumsum(weights)
    precision = np.divide(tp, n, out=np.zeros_like(tp), where=n > 0)
    recall = np.r_[0.0, tp / total, 1.0]
    precision = np.r_[0.0, precision, 0.0]
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    changes = np.flatnonzero(np.diff(recall) > 0)
    return float(np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1]))


# --------------------------------------------------------------------------- phase recognition
def phase_recognition(frames, scores, phase_queries):
    """Per frame: predicted phase = arg max of the 7 phase-query scores. {video: (true, predicted)}."""
    q_index = {q: i for i, q in enumerate(frames.queries)}
    wanted = [q_index[q] for q in phase_queries]
    per_video = {}
    for v in range(len(frames.video_ids)):
        blocks = [frames.block.get((v, qi)) for qi in wanted]
        if any(b is None for b in blocks):
            continue
        idx = [frames.frame_idx[a:b] for a, b in blocks]
        if any(not np.array_equal(idx[0], other) for other in idx[1:]):
            raise ValueError(f'{frames.video_ids[v]}: phase queries are not on the same frames')
        s = np.stack([scores[a:b] for a, b in blocks], axis=1)
        y = np.stack([frames.label[a:b] for a, b in blocks], axis=1)
        observed = (y != -100).all(axis=1)
        if np.any((y != -100).any(axis=1) & ~observed):
            raise ValueError(f'{frames.video_ids[v]}: partially annotated phase frames')
        y, s = y[observed], s[observed]
        if len(y) == 0:
            continue
        if not np.all((y == 1).sum(axis=1) == 1):
            raise ValueError(f'{frames.video_ids[v]}: a frame does not have exactly one phase')
        per_video[v] = (np.argmax(y, axis=1), np.argmax(s, axis=1))
    return per_video


def phase_tables(per_video, n_phases=7):
    """Per video: accuracy, macro F1, and per phase precision / recall / Jaccard (NaN when the phase does not
    occur in the video, and precision NaN when it is never predicted), as in the Cholec80 evaluation script."""
    from sklearn.metrics import f1_score
    videos = sorted(per_video)
    acc, f1, prec, rec, jac = [], [], [], [], []
    for v in videos:
        gt, pred = per_video[v]
        acc.append(float(np.mean(gt == pred)))
        f1.append(float(f1_score(gt, pred, average='macro')))
        p_row, r_row, j_row = [], [], []
        for k in range(n_phases):
            in_gt, in_pred = gt == k, pred == k
            if not in_gt.any():
                p_row.append(math.nan); r_row.append(math.nan); j_row.append(math.nan)
                continue
            tp = float(np.sum(in_gt & in_pred))
            p_row.append(tp / in_pred.sum() if in_pred.any() else math.nan)
            r_row.append(tp / in_gt.sum())
            j_row.append(tp / np.sum(in_gt | in_pred))
        prec.append(p_row); rec.append(r_row); jac.append(j_row)
    return dict(videos=np.asarray(videos, dtype=np.int64), accuracy=np.asarray(acc), f1=np.asarray(f1),
                precision=np.asarray(prec, dtype=np.float64).reshape(-1, n_phases),
                recall=np.asarray(rec, dtype=np.float64).reshape(-1, n_phases),
                jaccard=np.asarray(jac, dtype=np.float64).reshape(-1, n_phases))


def phase_summary(t):
    if not len(t['videos']):
        return None
    out = dict(accuracy=float(t['accuracy'].mean()), accuracy_std=float(t['accuracy'].std()),
               f1_videowise=float(t['f1'].mean()))
    import warnings
    for key in ('precision', 'recall', 'jaccard'):
        with warnings.catch_warnings():                                # a phase absent from every video: NaN
            warnings.simplefilter('ignore', RuntimeWarning)
            per_phase = np.nanmean(t[key], axis=0)                     # over videos, per phase
        out[key], out[key + '_std'] = float(np.nanmean(per_phase)), float(np.nanstd(per_phase))
        out.setdefault('per_phase', {})[key] = per_phase.tolist()
    return out


# --------------------------------------------------------------------------- loading
class Frames:
    """The rows of one evaluation set (sorted by video, canonical query, frame), shared by every method."""
    def __init__(self, df):
        self.video_ids = sorted(df['video_id'].unique(), key=natural_key)
        self.queries = list(protocol.canonical_queries())
        unknown = set(df['text_query'].unique()) - set(self.queries)
        if unknown:
            raise ValueError(f'unexpected queries: {sorted(unknown)[:3]}')
        vpos = {v: i for i, v in enumerate(self.video_ids)}
        qpos = {q: i for i, q in enumerate(self.queries)}
        self.video = df['video_id'].map(vpos).to_numpy(np.int64)
        self.query = df['text_query'].map(qpos).to_numpy(np.int64)
        self.frame_idx = df['frame_idx'].to_numpy(np.int64)
        self.label = df['label'].to_numpy(np.int64)
        if not np.isin(self.label, (-100, 0, 1)).all():
            raise ValueError('labels must be -100, 0 or 1')
        key = self.video * len(self.queries) + self.query
        starts = np.r_[0, np.flatnonzero(np.diff(key) != 0) + 1]
        stops = np.r_[starts[1:], len(key)]
        self.groups = list(zip(starts.tolist(), stops.tolist()))
        self.block = {(int(self.video[a]), int(self.query[a])): (a, b) for a, b in self.groups}
        self.observed = self.label != -100

    def same_rows(self, other):
        return (self.video_ids == other.video_ids and np.array_equal(self.video, other.video)
                and np.array_equal(self.query, other.query) and np.array_equal(self.frame_idx, other.frame_idx)
                and np.array_equal(self.label, other.label))


def load_set(results, set_name, rename=None):
    """Predictions of one run on one set, merged one-to-one with its reference labels and sorted."""
    pred, ref = results / f'proposed_{set_name}.csv', results / f'{set_name}_reference.csv'
    if not pred.exists() or not ref.exists():
        return None
    r = pd.read_csv(ref, usecols=KEYS + ['label'])
    p = pd.read_csv(pred)
    columns = KEYS + ['score'] + (['evidence_uncertainty'] if 'evidence_uncertainty' in p else [])
    m = r.merge(p[columns], on=KEYS, how='outer', indicator=True, validate='one_to_one')
    if not (m['_merge'] == 'both').all():
        raise ValueError(f'{pred}: prediction and reference frames differ')
    m = m.drop(columns='_merge')
    if rename is not None:
        mapped = m['text_query'].map(rename)
        if mapped.isna().any():
            raise ValueError(f'{pred}: queries without a canonical concept: {m.text_query[mapped.isna()].unique()[:3]}')
        m['text_query'] = mapped
    m['_q'] = m['text_query'].map({q: i for i, q in enumerate(protocol.canonical_queries())})
    if m['_q'].isna().any():
        raise ValueError(f'{pred}: queries that are not among the 14 concepts: {m.text_query[m._q.isna()].unique()[:3]}')
    order = {v: i for i, v in enumerate(sorted(m['video_id'].unique(), key=natural_key))}
    m['_v'] = m['video_id'].map(order)
    m = m.sort_values(['_v', '_q', 'frame_idx'], kind='mergesort').reset_index(drop=True)
    scores = m['score'].to_numpy(np.float64)
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError(f'{pred}: scores must be probabilities in [0, 1]')
    unc = m['evidence_uncertainty'].to_numpy(np.float64) if 'evidence_uncertainty' in m else None
    return Frames(m), scores, unc, dict(predictions=file_sha(pred), reference=file_sha(ref))


def train_prevalence(root, sample_fps):
    """Fraction of positive (annotated) training frames of every concept: defines rare vs common concepts."""
    from data_contract import AnnotationIndex, load_video_metadata, query_spec, sample_indices
    data = root / 'data'
    split = json.loads((data / 'splits.json').read_text())
    meta = load_video_metadata(data / 'video_metadata.json')
    index = AnnotationIndex(data / 'parsed_annotations.csv')
    out = {}
    for q in protocol.canonical_queries():
        spec = query_spec(q)
        pos = obs = 0
        for video in split['train']:
            m = meta[video]
            for i in sample_indices(m['frame_count'], m['source_fps'], sample_fps):
                y = index.label(video, int(i), spec)
                if y != -100:
                    obs += 1
                    pos += y
        out[q] = pos / obs if obs else math.nan
    return out


# --------------------------------------------------------------------------- per-method analysis
class Method:
    def __init__(self, plan, root):
        self.plan = plan
        self.name, self.kind, self.label, self.group = plan['name'], plan['kind'], plan['label'], plan['group']
        self.results = root / 'results' / self.name
        self.run = root / 'runs' / self.name
        self.root = root
        calib = self.results / 'proposed_calibration.json'
        self.threshold = json.loads(calib.read_text())['threshold'] if calib.exists() else None
        self.short = self.kind if self.kind != 'proposed' else (
            'advanced' if plan['flags'] == protocol.ADVANCED else 'baseline' if not plan['flags'] else 'ablation')

    @property
    def finished(self):
        """Evaluated completely (main.py writes the evaluate marker only after every set was evaluated)."""
        return (self.threshold is not None and (self.results / 'proposed_test_metrics.json').exists()
                and (self.root / 'state' / f'evaluate.{self.name}.done.json').exists())


def frame_summary(frames, scores, threshold, unc=None):
    obs = frames.observed
    m = binary_metrics(frames.label[obs], scores[obs], threshold, bins=BINS)
    per_query = {}
    for qi, q in enumerate(frames.queries):
        sel = obs & (frames.query == qi)
        y = frames.label[sel]
        per_query[q] = float(Ranked(y, scores[sel], frames.video[sel]).ap()) if y.sum() else math.nan
    out = {k: m[k] for k in ('average_precision', 'auroc', 'f1', 'precision', 'recall', 'brier', 'ece', 'aurc', 'n',
                             'positives')}
    out['threshold'] = threshold
    out['per_query_ap'] = per_query
    values = np.asarray(list(per_query.values()))
    out['macro_map'] = float(np.nanmean(values))
    out['phase_map'] = float(np.nanmean(values[:7]))
    out['tool_map'] = float(np.nanmean(values[7:]))
    if unc is not None:
        from metrics import uncertainty_aurc
        out['evidence_uncertainty_aurc'] = uncertainty_aurc(frames.label[obs], scores[obs], unc[obs], threshold)
    # per video (pooled over queries): for the Wilcoxon signed-rank test
    out['per_video_ap'] = []
    for v in range(len(frames.video_ids)):
        sel = obs & (frames.video == v)
        y = frames.label[sel]
        out['per_video_ap'].append(float(Ranked(y, scores[sel], frames.video[sel]).ap()) if y.sum() else math.nan)
    return out


def frame_replicates(frames, scores, threshold, counts):
    """Bootstrap replicates (rows of `counts` = video multiplicities) of pooled AP, F1 and every query's AP."""
    obs = frames.observed
    y, v, s = frames.label[obs], frames.video[obs], scores[obs]
    pooled = Ranked(y, s, v)
    queries = [(qi, Ranked(y[frames.query[obs] == qi], s[frames.query[obs] == qi], v[frames.query[obs] == qi]))
               for qi in range(len(frames.queries))]
    n_v = len(frames.video_ids)
    pred = s >= threshold
    tp = np.bincount(v, weights=(pred & (y == 1)).astype(float), minlength=n_v)
    fp = np.bincount(v, weights=(pred & (y == 0)).astype(float), minlength=n_v)
    fn = np.bincount(v, weights=(~pred & (y == 1)).astype(float), minlength=n_v)
    B = len(counts)
    ap = np.empty(B)
    qap = np.empty((B, len(queries)))
    for b in range(B):
        w = counts[b]
        ap[b] = pooled.ap(w)
        for j, (_, r) in enumerate(queries):
            qap[b, j] = r.ap(w)
    TP, FP, FN = counts @ tp, counts @ fp, counts @ fn
    f1 = np.divide(2 * TP, 2 * TP + FP + FN, out=np.zeros(B), where=(2 * TP + FP + FN) > 0)
    return dict(average_precision=ap, f1=f1, per_query_ap=qap)


# --------------------------------------------------------------------------- formatting
def pct(x, digits=1):
    if x is None or not np.isfinite(float(x)):
        return 'n/a'
    value = round(100 * float(x), digits)
    return f'{0.0 if value == 0 else value:.{digits}f}'                 # never "-0.0"


def with_ci(x, ci):
    if x is None or not np.isfinite(x):
        return 'n/a'
    if ci is None or not all(np.isfinite(ci)):
        return pct(x)
    return f'{pct(x)} [{pct(ci[0])}, {pct(ci[1])}]'


def fmt_p(p):
    if p is None or not np.isfinite(p):
        return 'n/a'
    return '<0.001' if p < 0.001 else f'{p:.3f}'


def write_table(out, name, df, caption, notes=''):
    """CSV (raw values as given), Markdown (for PAPER_RESULTS.md) and a booktabs LaTeX table."""
    (out / 'tables').mkdir(parents=True, exist_ok=True)
    df.to_csv(out / 'tables' / f'{name}.csv', index=False)
    cols = list(df.columns)

    def tex(x):
        return str(x).replace('\\', r'\textbackslash{}').replace('&', r'\&').replace('%', r'\%').replace('_', r'\_') \
            .replace('#', r'\#').replace('<', r'$<$').replace('>', r'$>$').replace('±', r'$\pm$')
    lines = [r'\begin{table}[t]', r'\centering', r'\small', rf'\caption{{{tex(caption)}}}', rf'\label{{tab:{name}}}',
             r'\resizebox{\linewidth}{!}{%', r'\begin{tabular}{l' + 'c' * (len(cols) - 1) + '}', r'\toprule',
             ' & '.join(tex(c) for c in cols) + r' \\', r'\midrule']
    lines += [' & '.join(tex(v) for v in row) + r' \\' for row in df.astype(str).itertuples(index=False)]
    lines += [r'\bottomrule', r'\end{tabular}}']                       # (\resizebox needs \usepackage{graphicx})
    if notes:
        lines += [r'\par\smallskip\footnotesize ' + tex(notes)]
    lines += [r'\end{table}']
    (out / 'tables' / f'{name}.tex').write_text('\n'.join(lines) + '\n')
    md = [f'**{caption}**', '', '| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    md += ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in df.astype(str).itertuples(index=False)]
    if notes:
        md += ['', f'_{notes}_']
    return '\n'.join(md) + '\n'


def save_figure(fig, out, name):
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(out / 'figures' / f'{name}.{ext}', dpi=160, bbox_inches='tight')


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True)
    ap.add_argument('--prefix', default='paper')
    ap.add_argument('--output')
    ap.add_argument('--sample-fps', type=float, default=1.0)
    ap.add_argument('--bootstrap', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    out = Path(a.output).resolve() if a.output else root / 'paper'
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import wilcoxon
    from data_contract import load_video_metadata
    from project_config import config

    started = time.time()
    meta = load_video_metadata(root / 'data' / 'video_metadata.json')
    plans = protocol.plans(a.prefix)
    methods = {p['name']: Method(p, root) for p in plans}
    finished = [m for m in methods.values() if m.finished]
    missing = [m.name for m in methods.values() if not m.finished]
    log(f'{len(finished)} of {len(methods)} runs finished' + (f'; missing: {missing}' if missing else ''))
    report = dict(version=VERSION, generated=time.ctime(), root=str(root), prefix=a.prefix, bootstrap=a.bootstrap,
                  seed=a.seed, finished=[m.name for m in finished], missing=missing, methods={})
    md = ['# Paper results', '', f'Generated {time.ctime()} by `paper_analysis.py` ({VERSION}) from `{root}`.', '',
          f'Runs finished: **{len(finished)} of {len(methods)}**.' + (
              f' Not finished yet (left out below): {", ".join(missing)}.' if missing else ''), '']
    if not finished:
        (out / 'PAPER_RESULTS.md').write_text('\n'.join(md + ['No run has finished yet.']) + '\n')
        (out / 'analysis.json').write_text(json.dumps(report, indent=2))
        return 0

    # ---- the shared test rows, every method's scores on them, and the bootstrap resamples
    frames, scores, unc, shas = None, {}, {}, {}
    for m in list(finished):
        loaded = load_set(m.results, 'test')
        if loaded is None:
            continue
        f, s, u, sha = loaded
        if frames is None:
            frames = f
        elif not frames.same_rows(f):
            raise ValueError(f'{m.name}: its test rows/labels differ from the other runs (different data?)')
        scores[m.name], unc[m.name], shas[m.name] = s.astype(np.float64), u, sha
    finished = [m for m in finished if m.name in scores]
    if frames is None:
        (out / 'PAPER_RESULTS.md').write_text('\n'.join(md + ['No test predictions found yet.']) + '\n')
        (out / 'analysis.json').write_text(json.dumps(report, indent=2))
        return 0
    n_v = len(frames.video_ids)
    split = json.loads((root / 'data' / 'splits.json').read_text()) if (root / 'data' / 'splits.json').exists() else {}

    def span(vs):
        nums = sorted(int(m.group(1)) for m in (re.search(r'(\d+)$', v) for v in vs) if m)
        if nums and len(nums) > 1 and nums == list(range(nums[0], nums[-1] + 1)):
            return f'{nums[0]}-{nums[-1]}'
        return ', '.join(map(str, nums))
    md += [f'Test set: the {n_v} test videos ({span(frames.video_ids)}) of the split in data/splits.json, every frame '
           f'sampled at {a.sample_fps:g} fps x 14 queries; unannotated frames are excluded. Every method uses its own '
           f'threshold selected on the {len(split.get("val", []))} validation videos ({span(split.get("val", []))}). '
           f'Square brackets: 95 % bootstrap confidence interval over test videos ({a.bootstrap} resamples, the same '
           f'resamples for every method). Values in %.', '']
    report['split'] = {k: len(v) for k, v in split.items()}
    rng = np.random.default_rng(a.seed)
    counts = np.stack([np.bincount(rng.integers(0, n_v, n_v), minlength=n_v) for _ in range(a.bootstrap)]).astype(np.float64)
    report['test_videos'] = frames.video_ids

    stats, reps = {}, {}
    for m in finished:
        t0 = time.time()
        stats[m.name] = frame_summary(frames, scores[m.name], m.threshold, unc[m.name])
        reps[m.name] = frame_replicates(frames, scores[m.name], m.threshold, counts)
        qap = reps[m.name]['per_query_ap']
        reps[m.name]['macro_map'] = np.nanmean(qap, axis=1)
        reps[m.name]['phase_map'] = np.nanmean(qap[:, :7], axis=1)
        reps[m.name]['tool_map'] = np.nanmean(qap[:, 7:], axis=1)
        for key in ('average_precision', 'f1', 'macro_map', 'phase_map', 'tool_map'):
            stats[m.name][f'{key}_ci'] = percentile_ci(reps[m.name][key])
        log(f'{m.name}: frame metrics and bootstrap ({time.time() - t0:.0f} s)')

    # ---- segments
    seg = {}
    min_dur, gap = config.MIN_SEGMENT_DURATION, config.MERGE_GAP
    for m in finished:
        t0 = time.time()
        variants = [(m.name, None)]
        native_file = m.results / 'native_segments_test.jsonl'
        if m.kind == 'univtg' and native_file.exists():
            native = {}
            for line in native_file.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    native[(r['video_id'], r['text_query'])] = r['predictions']
            variants.append((m.name + ':native', native))
        for key, native in variants:
            match = SegmentMatch(segment_pairs(frames, scores[m.name], m.threshold, meta, min_dur, gap, native),
                                 m.threshold, n_v)
            seg[key] = match.metrics()
            boot = np.asarray([match.metrics(counts[b])['mAP@0.5'] for b in range(len(counts))])
            seg[key]['mAP@0.5_ci'] = percentile_ci(boot)
            if native is None:
                reps[m.name]['seg_map50'] = boot
        log(f'{m.name}: segment metrics ({time.time() - t0:.0f} s)')

    def by_short(short, group=None):
        for m in finished:
            if m.short == short and (group is None or m.group == group) and m.plan['seed'] == protocol.MAIN_SEED:
                return m
        return None

    # ---- main table
    main_rows, tables = [], {}
    main_methods = [x for x in (by_short(k, 'comparison' if k not in ('baseline', 'advanced') else 'main')
                                for k in MAIN_ORDER) if x is not None]
    for m in main_methods:
        st, sg = stats[m.name], seg[m.name]
        main_rows.append({'Method': m.label, 'AP': with_ci(st['average_precision'], st['average_precision_ci']),
                          'mAP (14 queries)': with_ci(st['macro_map'], st['macro_map_ci']),
                          'AUROC': pct(st['auroc']), 'F1': with_ci(st['f1'], st['f1_ci']),
                          'Phase mAP': pct(st['phase_map']), 'Tool mAP': pct(st['tool_map']),
                          'Segment mAP@0.5': with_ci(sg['mAP@0.5'], sg['mAP@0.5_ci']),
                          'ECE': pct(st['ece']), 'Brier': f"{st['brier']:.3f}"})
    if main_rows:
        md += ['## 1. Main results (test videos)', '',
               write_table(out, 'main_results', pd.DataFrame(main_rows),
                           'Language-guided temporal localization on the Cholec80 test videos',
                           'AP: frame-level average precision over all frames and queries; mAP: mean of the 14 per-query APs; '
                           'F1 at the threshold selected on validation; seed 42 for trained methods.'), '']

    # ---- significance
    sig_rows = []
    ours = [x for x in (by_short('advanced', 'main'), by_short('baseline', 'main')) if x is not None]
    pairs_to_test = []
    if len(ours) == 2:
        pairs_to_test.append((ours[0], ours[1]))
    for o in ours:
        pairs_to_test += [(o, c) for c in main_methods if c.group == 'comparison']
    base = by_short('baseline', 'main')
    pairs_to_test += [(m, base) for m in finished if m.group == 'ablation' and base is not None]
    for metric, label in (('average_precision', 'AP'), ('macro_map', 'mAP'), ('f1', 'F1'), ('seg_map50', 'Segment mAP@0.5')):
        rows = []
        for x, y in pairs_to_test:
            point_x = stats[x.name][metric] if metric != 'seg_map50' else seg[x.name]['mAP@0.5']
            point_y = stats[y.name][metric] if metric != 'seg_map50' else seg[y.name]['mAP@0.5']
            d = reps[x.name][metric] - reps[y.name][metric]
            ci = percentile_ci(d)
            p_boot = bootstrap_p(d)
            p_w = math.nan
            if metric == 'average_precision':
                xa, ya = np.asarray(stats[x.name]['per_video_ap']), np.asarray(stats[y.name]['per_video_ap'])
                ok = np.isfinite(xa) & np.isfinite(ya)
                if ok.sum() >= 5 and np.any(xa[ok] != ya[ok]):
                    p_w = float(wilcoxon(xa[ok], ya[ok]).pvalue)
            rows.append(dict(metric=label, a=x.label, b=y.label,
                             delta=point_x - point_y, ci=ci, p_boot=p_boot, p_wilcoxon=p_w))
        adj = holm([r['p_boot'] for r in rows])
        adj_w = holm([r['p_wilcoxon'] for r in rows])
        for r, h, hw in zip(rows, adj, adj_w):
            sig_rows.append({'Metric': r['metric'], 'A': r['a'], 'B': r['b'], 'A - B': pct(r['delta']),
                             '95% CI': f'[{pct(r["ci"][0])}, {pct(r["ci"][1])}]', 'p (bootstrap)': fmt_p(r['p_boot']),
                             'p (Holm)': fmt_p(h), 'p (Wilcoxon, per-video AP)': fmt_p(r['p_wilcoxon']),
                             'p (Wilcoxon, Holm)': fmt_p(hw)})
            report.setdefault('significance', []).append(dict(r, p_holm=h, p_wilcoxon_holm=hw))
    if sig_rows:
        md += ['## 2. Paired significance tests (test videos)', '',
               write_table(out, 'significance', pd.DataFrame(sig_rows), 'Paired comparisons on the test videos',
                           'Differences in percentage points; paired bootstrap over test videos (two-sided); Holm '
                           'correction within each metric; Wilcoxon signed-rank test on the 32 per-video APs.'), '']

    # ---- segments table
    seg_rows = []
    for m in main_methods:
        for key, label in ((m.name, m.label), (m.name + ':native', m.label + ' - own windows')):
            if key not in seg:
                continue
            s = seg[key]
            seg_rows.append({'Method': label, **{f'R@1 IoU{t}': pct(s[f'R1@{t}']) for t in IOUS},
                             **{f'mAP@{t}': pct(s[f'mAP@{t}']) for t in IOUS}, 'mAP avg': pct(s['mAP']),
                             'mIoU (top-1)': pct(s['mIoU']), 'Frame IoU': pct(s['frame_IoU']),
                             'Absent-query false alarms': pct(s['absent_false_alarm'])})
    if seg_rows:
        md += ['## 3. Temporal segments (test videos)', '',
               write_table(out, 'segment_results', pd.DataFrame(seg_rows), 'Temporal segment localization',
                           'Segments = runs of frames at or above the validation threshold (same for every method); '
                           'R@1: the highest-scoring segment overlaps a ground-truth segment by at least the tIoU; '
                           'mAP: pooled over all video-query pairs; mIoU: tIoU of the top-1 segment; frame IoU: '
                           'Jaccard index of predicted and true frames; absent-query false alarms: share of '
                           'video-query pairs without the concept that still get a segment. UniVTG is also '
                           'evaluated on its own windows (after its NMS).'), '']
    report['segments'] = seg

    # ---- phase recognition and tool presence
    phase_rows, phase_stats = [], {}
    phase_q = protocol.phase_queries()
    candidates = [(m.name, m.label, scores[m.name]) for m in main_methods]
    pz = by_short('peskavlp_zeroshot', 'comparison')
    if pz is not None and (pz.results / 'authorprompts_test.csv').exists():
        df = pd.read_csv(pz.results / 'authorprompts_test.csv', usecols=KEYS + ['score'])
        rows = pd.DataFrame(dict(video_id=[frames.video_ids[v] for v in frames.video],
                                 text_query=[frames.queries[q] for q in frames.query],
                                 frame_idx=frames.frame_idx, _row=np.arange(len(frames.video))))
        df = rows.merge(df, on=KEYS, how='left', validate='one_to_one')
        s = df.sort_values('_row')['score'].to_numpy(np.float64)
        if np.isfinite(s).all() and np.all((s >= 0) & (s <= 1)):
            candidates.append((pz.name + ':author', pz.label + ' (authors\' phase prompts)', s))
        else:
            log('(authorprompts_test.csv does not cover the test frames; left out)')
    for key, label, s in candidates:
        tables = phase_tables(phase_recognition(frames, s, phase_q))
        res = phase_summary(tables)
        if res is None:
            continue
        w = counts[:, tables['videos']]
        res['accuracy_ci'] = percentile_ci((w @ tables['accuracy']) / w.sum(axis=1))
        phase_stats[key] = res
        phase_rows.append({'Method': label, 'Accuracy': f"{with_ci(res['accuracy'], res['accuracy_ci'])} (± {pct(res['accuracy_std'])})",
                           'Precision': f"{pct(res['precision'])} ± {pct(res['precision_std'])}",
                           'Recall': f"{pct(res['recall'])} ± {pct(res['recall_std'])}",
                           'Jaccard': f"{pct(res['jaccard'])} ± {pct(res['jaccard_std'])}",
                           'F1 (video-wise)': pct(res['f1_videowise'])})
    if phase_rows:
        md += ['## 4. Surgical phase recognition (test videos)', '',
               write_table(out, 'phase_recognition', pd.DataFrame(phase_rows),
                           'Phase recognition from the seven phase queries (arg max per frame)',
                           'Standard non-relaxed Cholec80 metrics: accuracy per video (mean, bootstrap CI, ± std over '
                           'videos); precision/recall/Jaccard per phase averaged over videos (± std over phases); '
                           'F1: macro F1 per video, averaged (the protocol of the PeskaVLP paper). Not trained as a '
                           'classifier: a by-product of query scoring.'), '']
    report['phase_recognition'] = phase_stats
    tool_rows = []
    tool_q = protocol.tool_queries()
    for m in main_methods:
        st = stats[m.name]
        tool_rows.append({'Method': m.label, **{q.replace(' is present', '').replace(' are present', '').replace('a ', '').replace('an ', ''): pct(st['per_query_ap'][q]) for q in tool_q},
                          'mAP': with_ci(st['tool_map'], st['tool_map_ci'])})
    if tool_rows:
        md += ['## 5. Tool presence (test videos)', '',
               write_table(out, 'tool_presence', pd.DataFrame(tool_rows), 'Tool presence detection: AP per tool',
                           'AP over all test frames per tool (the standard Cholec80 tool-presence metric) and their mean.'), '']

    # ---- per query and breakdowns
    try:
        prevalence = train_prevalence(root, a.sample_fps)
    except Exception as e:                                                  # noqa: BLE001
        log(f'(training prevalence unavailable: {e!r})')
        prevalence = {}
    q_rows = []
    for qi, q in enumerate(frames.queries):
        row = {'Query': q, 'Train prevalence': pct(prevalence.get(q, math.nan))}
        for m in main_methods:
            row[m.label] = pct(stats[m.name]['per_query_ap'][q])
        q_rows.append(row)
    md += ['## 6. Per-query AP and breakdowns', '',
           write_table(out, 'per_query_ap', pd.DataFrame(q_rows), 'AP per query (test videos)',
                       'Train prevalence: share of annotated training frames where the concept is present.'), '']
    if prevalence:
        median = float(np.nanmedian(list(prevalence.values())))
        rare = [qi for qi, q in enumerate(frames.queries) if prevalence.get(q, math.nan) < median]
        common = [qi for qi, q in enumerate(frames.queries) if prevalence.get(q, math.nan) >= median]
    else:
        median, rare, common = math.nan, [], []
    b_rows = []
    groups = [('Phases (7)', list(range(7))), ('Tools (7)', list(range(7, 14)))]
    if rare and common:
        groups += [(f'Rare concepts ({len(rare)}, train prevalence < {pct(median)} %)', rare),
                   (f'Common concepts ({len(common)})', common)]
    for m in main_methods:
        qap = reps[m.name]['per_query_ap']
        row = {'Method': m.label}
        for name, idx in groups:
            point = float(np.nanmean([stats[m.name]['per_query_ap'][frames.queries[i]] for i in idx]))
            row[name] = with_ci(point, percentile_ci(np.nanmean(qap[:, idx], axis=1)))
        b_rows.append(row)
    md += [write_table(out, 'breakdown', pd.DataFrame(b_rows), 'mAP by query type and by concept frequency',
                       'Rare/common: split at the median training prevalence of the 14 concepts.'), '']
    report['prevalence'] = prevalence

    # ---- ablations
    abl_rows = []
    base = by_short('baseline', 'main')
    adv = by_short('advanced', 'main')
    abl = ([base] if base else []) + [m for m in finished if m.group == 'ablation'] + ([adv] if adv else [])
    for m in abl:
        st, sg = stats[m.name], seg[m.name]
        row = {'Model': m.label, 'AP': with_ci(st['average_precision'], st['average_precision_ci']),
               'mAP': pct(st['macro_map']), 'F1': pct(st['f1']), 'Segment mAP@0.5': pct(sg['mAP@0.5']),
               'ECE': pct(st['ece'])}
        if base is not None and m is not base:
            d = reps[m.name]['average_precision'] - reps[base.name]['average_precision']
            ci = percentile_ci(d)
            row['AP vs baseline'] = f"{pct(st['average_precision'] - stats[base.name]['average_precision'])} " \
                                    f"[{pct(ci[0])}, {pct(ci[1])}], p={fmt_p(bootstrap_p(d))}"
        else:
            row['AP vs baseline'] = '-'
        abl_rows.append(row)
    if len(abl_rows) > 1:
        md += ['## 7. Ablations (seed 42; one add-on at a time)', '',
               write_table(out, 'ablations', pd.DataFrame(abl_rows), 'Contribution of each add-on',
                           'Each ablation adds exactly one add-on to the baseline; the advanced model has all four.'), '']

    # ---- seeds
    seed_rows = []
    for short, label in (('baseline', 'Baseline (ours)'), ('advanced', 'Advanced (ours)')):
        runs = [m for m in finished if m.short == short and m.group in ('main', 'seed')]
        if not runs:
            continue
        row = {'Model': label, 'Seeds': ', '.join(str(m.plan['seed']) for m in runs)}
        for key, name in (('average_precision', 'AP'), ('macro_map', 'mAP'), ('auroc', 'AUROC'), ('f1', 'F1'),
                          ('ece', 'ECE')):
            values = np.asarray([stats[m.name][key] for m in runs], dtype=np.float64)
            row[name] = f'{pct(values.mean())} ± {pct(values.std(ddof=1)) if len(values) > 1 else "n/a"}'
        values = np.asarray([seg[m.name]['mAP@0.5'] for m in runs])
        row['Segment mAP@0.5'] = f'{pct(values.mean())} ± {pct(values.std(ddof=1)) if len(values) > 1 else "n/a"}'
        seed_rows.append(row)
        report.setdefault('seeds', {})[short] = {m.plan['seed']: dict(ap=stats[m.name]['average_precision'],
                                                                      macro_map=stats[m.name]['macro_map'],
                                                                      f1=stats[m.name]['f1']) for m in runs}
    if seed_rows:
        md += ['## 8. Variation over training seeds', '',
               write_table(out, 'seeds', pd.DataFrame(seed_rows), 'Mean ± standard deviation over training seeds',
                           'Same data and split; only the training seed differs (42, 43, 44).'), '']

    # ---- unseen wordings
    gen_rows = []
    rename = {}
    try:
        for name, mapping in protocol.load_paraphrases().items():
            from data_contract import canonical_query
            rename[name] = {text: canonical_query(spec) for spec, text in mapping.items()}
    except Exception as e:                                                  # noqa: BLE001
        log(f'(paraphrase sets unavailable: {e!r})')
    for m in finished:
        if not m.plan['paraphrases'] or not rename:
            continue
        row = {'Method': m.label + (f' (seed {m.plan["seed"]})' if m.group == 'seed' else ''),
               'AP (training wording)': pct(stats[m.name]['average_precision'])}
        maps = {'mAP (training wording)': pct(stats[m.name]['macro_map'])}
        found = False
        for name in sorted(rename):
            row[f'AP set {name}'] = maps[f'mAP set {name}'] = 'n/a'
            loaded = load_set(m.results, f'test_para{name}', rename=rename[name])
            if loaded is None:
                continue
            f, s, _, _ = loaded
            if not frames.same_rows(f):
                raise ValueError(f'{m.name}: paraphrase set {name} is not on the test frames/labels')
            found = True
            para = frame_summary(frames, s, m.threshold)
            point = para['average_precision']
            ranked = Ranked(frames.label[frames.observed], s[frames.observed], frames.video[frames.observed])
            d = np.asarray([ranked.ap(w) for w in counts]) - reps[m.name]['average_precision']
            ci = percentile_ci(d)
            row[f'AP set {name}'] = f'{pct(point)} ({pct(point - stats[m.name]["average_precision"])} ' \
                                    f'[{pct(ci[0])}, {pct(ci[1])}])'
            maps[f'mAP set {name}'] = pct(para['macro_map'])
            report.setdefault('paraphrases', {}).setdefault(m.name, {})[name] = dict(
                ap=point, macro_map=para['macro_map'], f1=para['f1'], ap_change_ci=ci)
        if found:
            gen_rows.append({**row, **maps})
    if gen_rows:
        md += ['## 9. Generalisation to unseen query wordings', '',
               write_table(out, 'generalisation', pd.DataFrame(gen_rows),
                           'Frame AP on the test videos with query wordings never seen in training',
                           'Set A: synonyms and clinical terms; set B: descriptions of the surgical action '
                           '(protocol/paraphrases.json). In brackets: change vs the training wording and its paired '
                           'bootstrap CI. Same frames, labels and threshold as with the training wording.'), '']

    # ---- efficiency
    eff_rows = []
    for m in main_methods + [x for x in finished if x.group == 'ablation']:
        eff_rows.append(efficiency(m, root))
    if eff_rows:
        md += ['## 10. Efficiency', '',
               write_table(out, 'efficiency', pd.DataFrame(eff_rows), 'Model size and compute',
                           'Training time: sum of the logged epoch times (train + validation) where available, '
                           'otherwise the wall time of the training stage. Throughput: video frames per second '
                           'for all 14 queries on the test videos, including frame loading. UniVTG works on '
                           'precomputed M2CRL features (feature extraction not included); its parameters exclude '
                           'the frozen backbone.'), '']

    # ---- literature (template: values must be copied from the cited papers)
    lit = pd.DataFrame([
        {'Method': 'PeskaVLP zero-shot (Yuan et al., NeurIPS 2024)', 'Task': 'phase recognition', 'Split': 'test 49-80',
         'Reported': 'fill in from the paper', 'Ours (same protocol)': pct(phase_stats.get(pz.name + ':author', {}).get('accuracy')) if pz else 'n/a'},
        {'Method': 'SurgVLP / HecVL zero-shot (Yuan et al.)', 'Task': 'phase recognition', 'Split': 'test 49-80',
         'Reported': 'fill in from the paper', 'Ours (same protocol)': '-'},
        {'Method': 'Supervised phase recognition (e.g. TeCNO, Trans-SVNet)', 'Task': 'phase recognition',
         'Split': '40 train / 40 test (differs)', 'Reported': 'fill in from the paper', 'Ours (same protocol)': '-'},
        {'Method': 'Supervised tool presence (e.g. EndoNet)', 'Task': 'tool presence mAP', 'Split': 'differs',
         'Reported': 'fill in from the paper', 'Ours (same protocol)': '-'},
    ])
    md += ['## 11. Published results (template)', '',
           write_table(out, 'literature_template', lit, 'Published Cholec80 results for context (to be completed)',
                       'Copy the numbers from the cited papers and state their protocol (split, relaxed or not); '
                       'only the PeskaVLP row is directly comparable (same test videos, the authors\' prompts).'), '']

    # ---- figures
    figs = []
    try:
        figs += reliability_figure(plt, out, frames, scores, main_methods)
        figs += risk_coverage_figure(plt, out, frames, scores, unc, main_methods)
        figs += timeline_figures(plt, out, frames, scores, meta, main_methods, stats)
        figs += training_figure(plt, out, [x for x in (by_short('baseline', 'main'), by_short('advanced', 'main')) if x])
        figs += per_query_figure(plt, out, frames, stats, main_methods)
        if gen_rows:
            figs += generalisation_figure(plt, out, report, stats, finished)
    except Exception as e:                                                  # noqa: BLE001
        import traceback
        log(f'!! a figure failed: {e!r}\n{traceback.format_exc()}')
        report['figure_error'] = repr(e)
    explanations = sorted((out / 'explanations').glob('*/explanation_*.png')) if (out / 'explanations').exists() else []
    md += ['## Figures', '', *[f'- `figures/{n}.png` (and .pdf)' for n in figs],
           *[f'- `{p.relative_to(out)}` - language-guided patch relevance (before the temporal head)' for p in explanations], '']
    md += ['## Files', '', 'Every table is also in `tables/` as CSV and as a LaTeX (booktabs) table; all numbers are in '
           '`analysis.json`. Inputs (prediction file hashes) are listed there too.', '']

    for m in finished:
        st = {k: v for k, v in stats[m.name].items() if k != 'per_video_ap'}
        report['methods'][m.name] = dict(label=m.label, kind=m.kind, group=m.group, seed=m.plan['seed'],
                                         threshold=m.threshold, inputs=shas[m.name], frame=st, segments=seg.get(m.name))
    report['seconds'] = round(time.time() - started, 1)
    (out / 'PAPER_RESULTS.md').write_text('\n'.join(md) + '\n')
    (out / 'analysis.json').write_text(json.dumps(report, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)))
    log(f'paper analysis written to {out} ({report["seconds"]:.0f} s)')
    return 0


# --------------------------------------------------------------------------- efficiency
def efficiency(m, root):
    row = {'Method': m.label, 'Parameters (M)': 'n/a', 'Trainable (M)': 'n/a', 'Training (h)': 'n/a',
           'Throughput (frames/s)': 'n/a', 'Peak training memory (GB)': 'n/a'}
    run = m.run
    counts = run / 'parameter_counts.json'
    if counts.exists():
        c = json.loads(counts.read_text())
        row['Parameters (M)'] = f"{sum(v['total'] for v in c.values()) / 1e6:.1f}"
        row['Trainable (M)'] = f"{sum(v['trainable'] for v in c.values()) / 1e6:.1f}"
    journals = [run / 'training_metrics.jsonl'] + sorted(run.glob('*/training_metrics.jsonl'))
    seconds, memory = 0.0, []
    for j in journals:
        if j.exists():
            for line in j.read_text().splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                seconds += float(r.get('train_seconds', 0) or 0) + float(r.get('val_seconds', 0) or 0)
                if r.get('peak_gpu_memory_gb') is not None:
                    memory.append(float(r['peak_gpu_memory_gb']))
    if seconds == 0:
        marker = root / 'state' / f'train.{m.name}.done.json'
        if marker.exists():
            seconds = float(json.loads(marker.read_text()).get('seconds', 0))
    if m.kind in ('clip_zeroshot', 'peskavlp_zeroshot'):
        row['Training (h)'] = '0 (zero-shot)'
        row['Trainable (M)'] = '0'
    elif seconds:
        row['Training (h)'] = f'{seconds / 3600:.1f}'
    if memory:
        row['Peak training memory (GB)'] = f'{max(memory):.1f}'
    manifest = m.results / 'proposed_test.csv.manifest.json'
    if manifest.exists():
        man = json.loads(manifest.read_text())
        if man.get('parameters') and row['Parameters (M)'] == 'n/a':
            row['Parameters (M)'] = f"{man['parameters'] / 1e6:.1f}"
        if man.get('seconds'):
            if man.get('frame_scores'):                     # one score per frame and query
                fps = man['frame_scores'] / 14 / man['seconds']
            elif man.get('frames'):                         # frames encoded once, all queries scored
                fps = man['frames'] / man['seconds']
            else:
                fps = None
            if fps:
                row['Throughput (frames/s)'] = f'{fps:.0f}'
    return row


# --------------------------------------------------------------------------- figures
def palette(n):
    base = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860', '#DA8BC3', '#8C8C8C']
    return [base[i % len(base)] for i in range(n)]


def reliability_figure(plt, out, frames, scores, methods):
    obs = frames.observed
    y = frames.label[obs]
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.plot([0, 1], [0, 1], ls='--', color='0.6', lw=1)
    for m, c in zip(methods, palette(len(methods))):
        p = scores[m.name][obs]
        edges = np.linspace(0, 1, BINS + 1)
        which = np.clip(np.digitize(p, edges[1:-1]), 0, BINS - 1)
        conf = np.bincount(which, weights=p, minlength=BINS)
        freq = np.bincount(which, weights=y, minlength=BINS)
        n = np.bincount(which, minlength=BINS)
        ok = n > 0
        ece = np.sum(n[ok] / n.sum() * np.abs(conf[ok] / n[ok] - freq[ok] / n[ok]))
        ax.plot(conf[ok] / n[ok], freq[ok] / n[ok], marker='o', ms=3, lw=1.2, color=c, label=f'{m.label} (ECE {100 * ece:.1f})')
    ax.set_xlabel('predicted score'); ax.set_ylabel('observed frequency of the concept')
    ax.set_title('Reliability (test frames)', fontsize=10); ax.legend(fontsize=6.5, loc='upper left')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save_figure(fig, out, 'reliability'); plt.close(fig)
    return ['reliability']


def risk_coverage(y, p, threshold, ranking=None):
    pred = p >= threshold
    confidence = np.where(pred, p, 1 - p) if ranking is None else ranking
    order = np.argsort(-confidence, kind='stable')
    errors = np.cumsum(pred[order] != y[order])
    k = np.arange(1, len(y) + 1)
    pick = np.unique(np.linspace(0, len(y) - 1, 400).astype(int))
    return k[pick] / len(y), errors[pick] / k[pick]


def risk_coverage_figure(plt, out, frames, scores, unc, methods):
    obs = frames.observed
    y = frames.label[obs]
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for m, c in zip(methods, palette(len(methods))):
        cov, risk = risk_coverage(y, scores[m.name][obs], m.threshold)
        ax.plot(cov, risk, color=c, lw=1.3, label=m.label)
        if unc.get(m.name) is not None:
            cov, risk = risk_coverage(y, scores[m.name][obs], m.threshold, ranking=-unc[m.name][obs])
            ax.plot(cov, risk, color=c, lw=1.3, ls='--', label=m.label + ' (evidential uncertainty)')
    ax.set_xlabel('coverage (share of frames kept, most confident first)'); ax.set_ylabel('error rate')
    ax.set_title('Risk-coverage (test frames)', fontsize=10); ax.legend(fontsize=6.5)
    save_figure(fig, out, 'risk_coverage'); plt.close(fig)
    return ['risk_coverage']


TIMELINE_QUERIES = ('Calot triangle dissection phase', 'Clipping and cutting phase', 'a clip applier is present',
                    'a hook cautery is present')


def timeline_figures(plt, out, frames, scores, meta, methods, stats):
    ours = [m for m in methods if m.short in ('baseline', 'advanced')]
    others = sorted([m for m in methods if m.short not in ('baseline', 'advanced')],
                    key=lambda m: -stats[m.name]['average_precision'])
    shown = ours + others[:1]
    names = []
    q_index = {q: i for i, q in enumerate(frames.queries)}
    for v in range(min(2, len(frames.video_ids))):
        video = frames.video_ids[v]
        fps = float(meta[video]['source_fps'])
        fig, axes = plt.subplots(len(TIMELINE_QUERIES), 1, figsize=(9, 1.5 * len(TIMELINE_QUERIES)), sharex=True)
        for ax, q in zip(np.atleast_1d(axes), TIMELINE_QUERIES):
            block = frames.block.get((v, q_index[q]))
            if block is None:
                continue
            a, b = block
            t = frames.frame_idx[a:b] / fps / 60
            lab = frames.label[a:b]
            ax.fill_between(t, 0, 1, where=lab == 1, color='#55A868', alpha=0.25, step='post', lw=0, label='ground truth')
            ax.fill_between(t, 0, 1, where=lab == -100, color='0.85', alpha=0.6, step='post', lw=0, label='not annotated')
            for m, c in zip(shown, palette(len(shown))):
                ax.plot(t, scores[m.name][a:b], color=c, lw=0.9, label=m.label)
                ax.axhline(m.threshold, color=c, lw=0.6, ls=':')
            ax.set_ylim(0, 1); ax.set_ylabel('score', fontsize=7); ax.set_title(q, fontsize=8, loc='left')
        np.atleast_1d(axes)[-1].set_xlabel('time (min)')
        handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', fontsize=7, ncol=len(labels), frameon=False,
                   bbox_to_anchor=(0.5, 0.0))
        fig.suptitle(f'{video} (test): scores over time; dotted = each method\'s validation threshold', fontsize=9)
        fig.tight_layout(rect=(0, 0.05, 1, 0.97))
        name = f'timeline_{video}'
        save_figure(fig, out, name); plt.close(fig)
        names.append(name)
    return names


def training_figure(plt, out, methods):
    rows = {}
    for m in methods:
        j = m.run / 'training_metrics.jsonl'
        if j.exists():
            rows[m.label] = [json.loads(line) for line in j.read_text().splitlines() if line.strip()]
    if not rows:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.0))
    for (label, r), c in zip(rows.items(), palette(len(rows))):
        ep = [x['epoch'] for x in r]
        axes[0].plot(ep, [x['train_loss'] for x in r], color=c, ls='--', marker='.', label=f'{label}: train loss')
        axes[0].plot(ep, [x['val_frame_nll'] for x in r], color=c, marker='o', ms=3, label=f'{label}: val NLL')
        axes[1].plot(ep, [x['average_precision'] for x in r], color=c, marker='o', ms=3, label=f'{label}: val AP')
        best = min(r, key=lambda x: x['val_frame_nll'])['epoch']
        for ax in axes:
            ax.axvline(best, color=c, lw=0.6, ls=':')
    axes[0].set_xlabel('epoch'); axes[1].set_xlabel('epoch'); axes[0].legend(fontsize=6); axes[1].legend(fontsize=6)
    fig.suptitle('Training dynamics (dotted: epoch selected by validation NLL)', fontsize=9)
    fig.tight_layout()
    save_figure(fig, out, 'training_dynamics'); plt.close(fig)
    return ['training_dynamics']


def per_query_figure(plt, out, frames, stats, methods):
    fig, ax = plt.subplots(figsize=(10, 3.4))
    width = 0.8 / max(1, len(methods))
    x = np.arange(len(frames.queries))
    for i, (m, c) in enumerate(zip(methods, palette(len(methods)))):
        ax.bar(x + i * width - 0.4 + width / 2, [stats[m.name]['per_query_ap'][q] for q in frames.queries], width,
               color=c, label=m.label)
    ax.set_xticks(x)
    ax.set_xticklabels(frames.queries, rotation=40, ha='right', fontsize=7)
    ax.set_ylabel('AP'); ax.set_ylim(0, 1); ax.legend(fontsize=6.5, ncol=4)
    save_figure(fig, out, 'per_query_ap'); plt.close(fig)
    return ['per_query_ap']


def generalisation_figure(plt, out, report, stats, finished):
    para = report.get('paraphrases', {})
    ms = [m for m in finished if m.name in para and m.group != 'seed']
    if not ms:
        return []
    sets = sorted({k for v in para.values() for k in v})
    fig, ax = plt.subplots(figsize=(7, 3.2))
    width = 0.8 / (1 + len(sets))
    x = np.arange(len(ms))
    ax.bar(x - 0.4 + width / 2, [stats[m.name]['average_precision'] for m in ms], width, label='training wording',
           color='#4C72B0')
    for i, (s, c) in enumerate(zip(sets, ['#DD8452', '#55A868', '#C44E52'])):
        ax.bar(x - 0.4 + width * (i + 1.5), [para[m.name].get(s, {}).get('ap', math.nan) for m in ms], width,
               label=f'unseen wording set {s}', color=c)
    ax.set_xticks(x)
    ax.set_xticklabels([m.label for m in ms], rotation=25, ha='right', fontsize=7)
    ax.set_ylabel('frame AP (test)'); ax.set_ylim(0, 1); ax.legend(fontsize=7)
    save_figure(fig, out, 'generalisation'); plt.close(fig)
    return ['generalisation']


if __name__ == '__main__':
    sys.exit(main())
