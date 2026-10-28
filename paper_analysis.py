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
    resamples for every method, so differences are paired);
  * tests: primary endpoint = the 14-query mAP, paired permutation test (the two methods' scores swapped
    within random test videos) with Holm correction over all its comparisons; secondary endpoints (pooled AP,
    F1, segment mAP@0.5): paired bootstrap test centred at the null, Holm within each; plus the Wilcoxon
    signed-rank test on per-video AP;
  * segments: runs of frames at or above the threshold (the same post-processing for every method);
    ground truth: runs of annotated positive frames; unannotated frames are excluded from both;
  * phase recognition: per frame, the phase query with the highest score; the standard (non-relaxed) Cholec80
    metrics - accuracy per video, then precision / recall / Jaccard per phase averaged over videos - and, next to
    them, the relaxed-boundary metrics of the legacy Cholec80/M2CAI evaluation script (an exact port of its
    Evaluate.m / Main.m, see relaxed_evaluate) and the same relaxation applied as intended (both ends);
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

VERSION = 'paper-analysis-2'
KEYS = ['video_id', 'text_query', 'frame_idx']
IOUS = (0.3, 0.5, 0.7)
BINS = 15
# Main-table order: the comparison methods (weakest assumptions first), then ours.
MAIN_ORDER = ('clip_zeroshot', 'surgvlp_zeroshot', 'hecvl_zeroshot', 'peskavlp_zeroshot', 'xclip_finetune',
              'peskavlp_finetune', 'univtg', 'tecno', 'closedset', 'closedset_matched', 'baseline', 'advanced')


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


def bootstrap_p(differences, observed):
    """Two-sided paired-bootstrap test of "no difference": the bootstrap distribution of the difference is
    shifted to be centred at 0 (the null hypothesis) and compared with the observed difference,
    p = (1 + #{|d_b - d_obs| >= |d_obs|}) / (B + 1) (Efron & Tibshirani 1993, sec. 16.4)."""
    d = np.asarray(differences, dtype=np.float64)
    d = d[np.isfinite(d)]
    if not len(d) or not np.isfinite(observed):
        return math.nan
    return float(min(1.0, (np.sum(np.abs(d - observed) >= abs(observed) - 1e-12) + 1) / (len(d) + 1)))


def fast_ap(y, scores):
    """Unweighted AP (= sklearn average_precision_score, ties as one threshold)."""
    order = np.argsort(-scores)                         # order within ties is irrelevant (tie_ends)
    return weighted_ap(y[order], np.ones(len(y)), tie_ends(scores[order]))


_PERMUTATION = {}


def _permutation_chunk(task):
    """Permutation replicates of the mAP difference for one pair (runs in a worker process)."""
    pair, seed, n = task
    per_q = _PERMUTATION['queries']
    sx, sy = _PERMUTATION['scores'][pair[0]], _PERMUTATION['scores'][pair[1]]
    n_v = _PERMUTATION['n_videos']
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for b in range(n):
        swap = rng.random(n_v) < 0.5
        ax, ay = [], []
        for y, v, rows in per_q:
            s = swap[v]
            a, c = sx[rows], sy[rows]
            ax.append(fast_ap(y, np.where(s, c, a)))
            ay.append(fast_ap(y, np.where(s, a, c)))
        out[b] = np.nanmean(ax) - np.nanmean(ay)
    return out


def permutation_map_tests(frames, scores, pairs, n_perm, seed=0, workers=1):
    """Paired permutation tests (approximate randomisation) of the 14-query mAP difference for every pair:
    under the null hypothesis the two methods are exchangeable within a test video, so each permutation swaps
    the two methods' scores on a random subset of the videos (each video with probability 1/2, the same for all
    queries). Two-sided p = (1 + #{|d_perm| >= |d_obs|}) / (n_perm + 1). Returns {pair: (d_obs, p)}."""
    import warnings
    obs = frames.observed
    per_q = []
    for qi in range(len(frames.queries)):
        rows = np.flatnonzero(obs & (frames.query == qi))
        per_q.append((frames.label[rows].astype(np.float64), frames.video[rows], rows))
    _PERMUTATION.update(queries=per_q, scores=scores, n_videos=len(frames.video_ids))
    tasks = [(pair, seed + 7919 * i, n_perm) for i, pair in enumerate(pairs)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        if workers > 1 and len(tasks) > 1:
            import multiprocessing as mp
            with mp.get_context('fork').Pool(min(workers, len(tasks))) as pool:
                reps = pool.map(_permutation_chunk, tasks)
        else:
            reps = [_permutation_chunk(t) for t in tasks]
        out = {}
        for pair, d in zip(pairs, reps):
            sx, sy = scores[pair[0]], scores[pair[1]]
            d_obs = float(np.nanmean([fast_ap(y, sx[r]) for y, _, r in per_q]) -
                          np.nanmean([fast_ap(y, sy[r]) for y, _, r in per_q]))
            p = float((np.sum(np.abs(d) >= abs(d_obs) - 1e-12) + 1) / (len(d) + 1)) if np.isfinite(d_obs) else math.nan
            out[pair] = (d_obs, p)
    _PERMUTATION.clear()
    return out


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
    """Per frame: predicted phase = arg max of the 7 phase-query scores.
    {video: (true, predicted, position of each annotated frame on the video's sampling grid)}."""
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
        y, s, pos = y[observed], s[observed], np.flatnonzero(observed)
        if len(y) == 0:
            continue
        if not np.all((y == 1).sum(axis=1) == 1):
            raise ValueError(f'{frames.video_ids[v]}: a frame does not have exactly one phase')
        per_video[v] = (np.argmax(y, axis=1), np.argmax(s, axis=1), pos)   # pos: place on the sampling grid
    return per_video


def phase_tables(per_video, n_phases=7):
    """Per video: accuracy, macro F1, and per phase precision / recall / Jaccard (NaN when the phase does not
    occur in the video, and precision NaN when it is never predicted), as in the Cholec80 evaluation script."""
    from sklearn.metrics import f1_score
    videos = sorted(per_video)
    acc, f1, prec, rec, jac = [], [], [], [], []
    for v in videos:
        gt, pred = per_video[v][:2]
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
    out = dict(accuracy=float(t['accuracy'].mean()), accuracy_std=std1(t['accuracy']),
               f1_videowise=float(t['f1'].mean()))
    import warnings
    for key in ('precision', 'recall', 'jaccard'):
        with warnings.catch_warnings():                                # a phase absent from every video: NaN
            warnings.simplefilter('ignore', RuntimeWarning)
            per_phase = np.nanmean(t[key], axis=0)                     # over videos, per phase
        out[key], out[key + '_std'] = float(np.nanmean(per_phase)), std1(per_phase)
        out.setdefault('per_phase', {})[key] = per_phase.tolist()
    return out


def std1(values):
    """Sample standard deviation (N - 1, as MATLAB's std/nanstd), ignoring NaN; NaN with fewer than two values."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(v.std(ddof=1)) if len(v) > 1 else math.nan


# Relaxed-boundary phase metrics: a line-by-line port of the evaluation script distributed with Cholec80 /
# the M2CAI 2016 workflow challenge (Evaluate.m and Main.m; the copy in the TMRNet repository, MIT licence,
# is the one used by TMRNet, Trans-SVNet and many later papers). Phases are numbered 1-7 as in the script.
#
# What the script INTENDS: within the first and last 10 s of every ground-truth phase, a prediction of the
# neighbouring phase counts as correct (for phases 4-7 also two phases away).
# What it COMPUTES (legacy=True, the default here, reproduced exactly so that numbers can be compared with
# published ones): the statement for the end of a phase, curDiff(curDiff(end-t+1:end)==1) = 0, indexes with a
# logical mask of length t, which MATLAB applies to the FIRST t elements. So the end-of-phase condition, tested
# on the last 10 s, zeroes errors in the first 10 s; a too-early transition at the end of a phase is not
# relaxed. legacy=False applies the end condition to the last 10 s, as intended ("symmetric").
# Values above 100 % (relaxed true positives can exceed the predicted or true frames of a phase) are clipped to
# 100, as in Main.m.
#
# The script assumes one label per frame on a regular time grid. Here, frames without a phase annotation are
# kept on the grid as unknown (-1): they break a phase into separate runs and are left out of every count, so
# an annotation gap is never compressed into adjacent frames (frames on the 1 fps grid are 1 s apart).
def relaxed_evaluate(gt, pred, fps, legacy=True):
    """Evaluate.m for one video: gt / pred are 0-based phase ids per frame on the sampling grid, in time order
    (frames at `fps`); gt -1 = not annotated (left out). Returns (jaccard, precision, recall) per phase in %
    (NaN for phases absent from the ground truth; precision may be inf or NaN when a phase is never predicted)
    and the accuracy in %."""
    gt = np.asarray(gt, dtype=np.int64) + 1
    pred = np.asarray(pred, dtype=np.int64) + 1
    known = gt >= 1
    if not known.any():
        raise ValueError('no annotated frame')
    ori_t = int(round(10 * fps))                                     # 10 seconds relaxed boundary
    diff = pred - gt
    updated = diff.copy()
    for phase in range(1, 8):
        inside = np.flatnonzero(gt == phase)
        if not len(inside):
            continue
        breaks = np.flatnonzero(np.diff(inside) > 1)                 # bwconncomp: connected runs
        for start, end in zip(np.r_[inside[0], inside[breaks + 1]], np.r_[inside[breaks], inside[-1]]):
            cur = diff[start:end + 1].copy()
            t = min(ori_t, len(cur))
            if t > 0:
                head = cur[:t]                                       # views: assignments change cur
                late = (head == -1) | (head == -2) if phase in (6, 7) else head == -1
                head[late] = 0                                       # late transition
                tail = cur[len(cur) - t:]
                early = (tail == 1) | (tail == 2) if phase in (4, 5, 6, 7) else tail == 1
                (head if legacy else tail)[early] = 0                # early transition (legacy: applied to the head)
            updated[start:end + 1] = cur
    jac, prec, rec = [], [], []
    for phase in range(1, 8):
        in_gt, in_pred = gt == phase, (pred == phase) & known
        if not in_gt.any():
            jac.append(math.nan); prec.append(math.nan); rec.append(math.nan)
            continue
        union = in_gt | in_pred
        tp = float(np.sum(updated[union] == 0))
        jac.append(100 * tp / union.sum())
        sum_pred = int(in_pred.sum())
        prec.append(100 * tp / sum_pred if sum_pred else (math.inf if tp else math.nan))
        rec.append(100 * tp / in_gt.sum())
    acc = 100 * float(np.sum(updated[known] == 0)) / int(known.sum())
    return np.asarray(jac), np.asarray(prec), np.asarray(rec), acc


def on_grid(entry):
    """(gt, pred) of a phase_recognition entry on the full sampling grid, unknown frames as -1."""
    gt, pred = entry[0], entry[1]
    if len(entry) < 3 or entry[2] is None:
        return gt, pred
    pos = np.asarray(entry[2], dtype=np.int64)
    full_gt = np.full(int(pos.max()) + 1, -1, dtype=np.int64)
    full_pred = np.full(int(pos.max()) + 1, -1, dtype=np.int64)
    full_gt[pos], full_pred[pos] = gt, pred
    return full_gt, full_pred


def relaxed_phase_metrics(per_video, fps, legacy=True):
    """Main.m over the videos of `per_video` ({video: (true, predicted[, grid positions])}), as fractions."""
    import warnings
    videos = sorted(per_video)
    if not videos:
        return None
    rows = [relaxed_evaluate(*on_grid(per_video[v]), fps, legacy=legacy) for v in videos]
    jac = np.minimum(np.stack([r[0] for r in rows]), 100)          # (videos, 7); NaN stays NaN, inf -> 100
    prec = np.minimum(np.stack([r[1] for r in rows]), 100)
    rec = np.minimum(np.stack([r[2] for r in rows]), 100)
    acc = np.asarray([r[3] for r in rows])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        jac_phase, prec_phase, rec_phase = (np.nanmean(x, axis=0) for x in (jac, prec, rec))
    return dict(videos=np.asarray(videos, dtype=np.int64), accuracy_per_video=acc / 100,
                accuracy=float(acc.mean()) / 100, accuracy_std=std_all(acc) / 100,
                jaccard=float(jac_phase.mean()) / 100, jaccard_std=std_all(jac_phase) / 100,     # mean / std
                precision=float(np.nanmean(prec_phase)) / 100, precision_std=std1(prec_phase) / 100,  # nanmean
                recall=float(rec_phase.mean()) / 100, recall_std=std_all(rec_phase) / 100,
                per_phase=dict(jaccard=(jac_phase / 100).tolist(), precision=(prec_phase / 100).tolist(),
                               recall=(rec_phase / 100).tolist()))


def phase_gaps(per_video):
    """Frames without a phase annotation between the first and the last annotated frame, per video."""
    out = {}
    for v, entry in per_video.items():
        if len(entry) >= 3 and entry[2] is not None and len(entry[2]):
            pos = np.asarray(entry[2])
            out[v] = int(pos.max() - pos.min() + 1 - len(pos))
    return out


def std_all(values):
    """MATLAB std (N - 1): NaN if any value is NaN."""
    v = np.asarray(values, dtype=np.float64)
    return float(v.std(ddof=1)) if len(v) > 1 else math.nan


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
    ap.add_argument('--permutations', type=int, default=1000, help='random swaps per paired permutation test')
    ap.add_argument('--workers', type=int, default=None, help='processes for the permutation tests')
    a = ap.parse_args(argv)
    if a.workers is None:                                # the CPUs this job may use (Slurm allocation, affinity)
        import os
        try:
            n = len(os.sched_getaffinity(0))
        except AttributeError:
            n = os.cpu_count() or 2
        slurm = os.environ.get('SLURM_CPUS_PER_TASK', '')
        if slurm.isdigit() and int(slurm) > 0:
            n = min(n, int(slurm))
        a.workers = max(1, min(16, n - 1))
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
                  permutations=a.permutations,
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
                           'AP: frame-level average precision over all frames and queries; mAP: mean of the 14 per-query APs '
                           '(the primary endpoint); F1 at the threshold selected on validation; seed 42 for trained methods. '
                           'TeCNO: phase queries from its temporal network, tool queries from the auxiliary tool head '
                           'of its stage-1 CNN (TeCNO itself predicts phases only). Closed-set models: one output per '
                           'concept, no text input.'), '']

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
    # Primary endpoint: the 14-query mAP (every concept counts equally), tested with paired permutation tests
    # and Holm-corrected over all of its comparisons. Secondary endpoints (pooled AP, F1, segment mAP@0.5):
    # null-centred paired-bootstrap tests, Holm-corrected within each endpoint.
    t0 = time.time()
    permuted = permutation_map_tests(frames, scores, [(x.name, y.name) for x, y in pairs_to_test], a.permutations,
                                     a.seed, a.workers) if pairs_to_test else {}
    if pairs_to_test:
        log(f'permutation tests: {len(pairs_to_test)} pairs x {a.permutations} ({time.time() - t0:.0f} s)')
    for metric, label in (('macro_map', 'mAP'), ('average_precision', 'AP'), ('f1', 'F1'), ('seg_map50', 'Segment mAP@0.5')):
        rows = []
        for x, y in pairs_to_test:
            point_x = stats[x.name][metric] if metric != 'seg_map50' else seg[x.name]['mAP@0.5']
            point_y = stats[y.name][metric] if metric != 'seg_map50' else seg[y.name]['mAP@0.5']
            d = reps[x.name][metric] - reps[y.name][metric]
            ci = percentile_ci(d)
            if metric == 'macro_map':
                p_boot, test = permuted[(x.name, y.name)][1], 'permutation (primary)'
            else:
                p_boot, test = bootstrap_p(d, point_x - point_y), 'bootstrap (secondary)'
            p_w = math.nan
            if metric == 'average_precision':
                xa, ya = np.asarray(stats[x.name]['per_video_ap']), np.asarray(stats[y.name]['per_video_ap'])
                ok = np.isfinite(xa) & np.isfinite(ya)
                if ok.sum() >= 5 and np.any(xa[ok] != ya[ok]):
                    p_w = float(wilcoxon(xa[ok], ya[ok]).pvalue)
            rows.append(dict(metric=label, a=x.label, b=y.label, test=test,
                             delta=point_x - point_y, ci=ci, p=p_boot, p_wilcoxon=p_w))
        adj = holm([r['p'] for r in rows])
        adj_w = holm([r['p_wilcoxon'] for r in rows])
        for r, h, hw in zip(rows, adj, adj_w):
            sig_rows.append({'Metric': r['metric'], 'A': r['a'], 'B': r['b'], 'A - B': pct(r['delta']),
                             '95% CI': f'[{pct(r["ci"][0])}, {pct(r["ci"][1])}]', 'Test': r['test'],
                             'p': fmt_p(r['p']), 'p (Holm)': fmt_p(h), 'p (Wilcoxon, per-video AP)': fmt_p(r['p_wilcoxon']),
                             'p (Wilcoxon, Holm)': fmt_p(hw)})
            report.setdefault('significance', []).append(dict(r, p_holm=h, p_wilcoxon_holm=hw))
    if sig_rows:
        md += ['## 2. Paired significance tests (test videos)', '',
               write_table(out, 'significance', pd.DataFrame(sig_rows), 'Paired comparisons on the test videos',
                           f'Differences in percentage points, with 95 % paired-bootstrap CIs over test videos. Primary '
                           f'endpoint: 14-query mAP, two-sided paired permutation test ({a.permutations} random swaps of '
                           f'the two methods\' scores within test videos), Holm correction over its {len(pairs_to_test)} '
                           f'comparisons (null hypothesis: within every test video the two methods\' score vectors are '
                           f'exchangeable, which is stronger than equal mAP). Secondary endpoints (pooled AP, F1, '
                           f'segment mAP@0.5): two-sided paired '
                           f'bootstrap test with the bootstrap distribution centred at the null, Holm correction within '
                           f'each endpoint. Wilcoxon signed-rank test on the per-video APs. One training seed (42); '
                           f'the variation over seeds is in section 8.'), '']

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
    author = {}                                         # SurgVLP-family zero-shot models with the authors' prompts
    for kind in protocol.SURGVLP_ZERO_SHOT:
        z = by_short(kind, 'comparison')
        if z is None or not (z.results / 'authorprompts_test.csv').exists():
            continue
        df = pd.read_csv(z.results / 'authorprompts_test.csv', usecols=KEYS + ['score'])
        rows = pd.DataFrame(dict(video_id=[frames.video_ids[v] for v in frames.video],
                                 text_query=[frames.queries[q] for q in frames.query],
                                 frame_idx=frames.frame_idx, _row=np.arange(len(frames.video))))
        df = rows.merge(df, on=KEYS, how='left', validate='one_to_one')
        s = df.sort_values('_row')['score'].to_numpy(np.float64)
        if np.isfinite(s).all() and np.all((s >= 0) & (s <= 1)):
            candidates.append((z.name + ':author', z.label + ' (authors\' phase prompts)', s))
            author[kind] = z.name + ':author'
        else:
            log(f'({z.name}: authorprompts_test.csv does not cover the test frames; left out)')
    relaxed_rows = {'relaxed': [], 'relaxed_symmetric': []}
    gaps = None
    for key, label, s in candidates:
        per_video = phase_recognition(frames, s, phase_q)
        if gaps is None:                                 # the same for every method: a property of the labels
            gaps = {frames.video_ids[v]: n for v, n in phase_gaps(per_video).items()}
        tables = phase_tables(per_video)
        res = phase_summary(tables)
        if res is None:
            continue
        w = counts[:, tables['videos']]
        res['accuracy_ci'] = percentile_ci((w @ tables['accuracy']) / w.sum(axis=1))
        for name, legacy in (('relaxed', True), ('relaxed_symmetric', False)):
            relaxed = relaxed_phase_metrics(per_video, a.sample_fps, legacy=legacy)
            relaxed['accuracy_ci'] = percentile_ci((w @ relaxed.pop('accuracy_per_video')) / w.sum(axis=1))
            relaxed.pop('videos')
            res[name] = relaxed
            relaxed_rows[name].append({
                'Method': label,
                'Accuracy': f"{with_ci(relaxed['accuracy'], relaxed['accuracy_ci'])} (± {pct(relaxed['accuracy_std'])})",
                'Precision': f"{pct(relaxed['precision'])} ± {pct(relaxed['precision_std'])}",
                'Recall': f"{pct(relaxed['recall'])} ± {pct(relaxed['recall_std'])}",
                'Jaccard': f"{pct(relaxed['jaccard'])} ± {pct(relaxed['jaccard_std'])}"})
        phase_stats[key] = res
        phase_rows.append({'Method': label, 'Accuracy': f"{with_ci(res['accuracy'], res['accuracy_ci'])} (± {pct(res['accuracy_std'])})",
                           'Precision': f"{pct(res['precision'])} ± {pct(res['precision_std'])}",
                           'Recall': f"{pct(res['recall'])} ± {pct(res['recall_std'])}",
                           'Jaccard': f"{pct(res['jaccard'])} ± {pct(res['jaccard_std'])}",
                           'F1 (video-wise)': pct(res['f1_videowise'])})
    gaps = gaps or {}
    report['phase_annotation_gaps'] = dict(frames_without_phase_inside_annotated_span=gaps,
                                           videos_with_gaps=sorted(v for v, n in gaps.items() if n))
    gap_note = (' No test video has a frame without a phase annotation between its first and last annotated frame.'
                if not any(gaps.values()) else
                f' {sum(1 for n in gaps.values() if n)} test video(s) have frames without a phase annotation inside the '
                f'annotated span ({sum(gaps.values())} frames, listed in analysis.json); they are kept on the time grid '
                f'as unknown and excluded from every count, never merged into neighbouring frames.')
    if phase_rows:
        md += ['## 4. Surgical phase recognition (test videos)', '',
               write_table(out, 'phase_recognition', pd.DataFrame(phase_rows),
                           'Phase recognition from the seven phase queries (arg max per frame)',
                           'Standard non-relaxed Cholec80 metrics: accuracy per video (mean, bootstrap CI, ± std over '
                           'videos); precision/recall/Jaccard per phase averaged over videos (± std over phases); '
                           'F1: macro F1 per video, averaged (the video-wise F1 of the PeskaVLP paper). TeCNO and the '
                           'closed-set models are trained phase classifiers; for the language models phase '
                           'recognition is a by-product of query scoring.'), '',
               write_table(out, 'phase_recognition_relaxed', pd.DataFrame(relaxed_rows['relaxed']),
                           'Phase recognition, relaxed boundaries (legacy Cholec80 evaluation script)',
                           'The Cholec80/M2CAI evaluation script (Evaluate.m and Main.m) ported line by line, '
                           'including its behaviour: errors in the first 10 s of a ground-truth phase are forgiven '
                           'when the previous phase (for cleaning/retraction also the one before) is predicted; its '
                           'end-of-phase rule tests the last 10 s but, through MATLAB logical indexing, also acts on '
                           'the first 10 s, so a too-early transition is not forgiven. Values above 100 clipped as in '
                           'the script. Accuracy per video (mean, bootstrap CI, ± std over videos); Jaccard/precision/'
                           'recall per phase averaged over videos (± std over phases). Comparable with papers that '
                           'used this script.' + gap_note), '',
               write_table(out, 'phase_recognition_relaxed_symmetric', pd.DataFrame(relaxed_rows['relaxed_symmetric']),
                           'Phase recognition, relaxed boundaries applied as intended (both ends)',
                           'As the legacy table, but the end-of-phase rule is applied to the last 10 s of each phase, '
                           'so that a too-early transition into the next phase is forgiven as the script describes.'), '']
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
                           'AP over all test frames per tool (the standard Cholec80 tool-presence metric) and their mean. '
                           'TeCNO: the auxiliary tool head of its stage-1 CNN (not its temporal network).'), '']

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
                                    f"[{pct(ci[0])}, {pct(ci[1])}], p={fmt_p(bootstrap_p(d, st['average_precision'] - stats[base.name]['average_precision']))}"
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
    eff_rows, missing_rows = [], {}
    for m in main_methods + [x for x in finished if x.group == 'ablation']:
        eff_rows.append(efficiency(m, root, missing_rows))
    report['journal_epochs_without_a_row'] = missing_rows
    if eff_rows:
        md += ['## 10. Efficiency', '',
               write_table(out, 'efficiency', pd.DataFrame(eff_rows), 'Model size and compute',
                           'Training time: sum of the logged epoch times (train + validation) where available, '
                           'otherwise the wall time of the training stage' + (
                               ' (some epochs have no logged row, so these times are lower bounds; see analysis.json)'
                               if missing_rows else '') + '. Training clips: training windows (clips of the clip '
                           'length) passed through the video backbone (ours: one per window and query; closed set: one '
                           'per window and repeat); optimiser steps: the parameter updates they made. Throughput: video frames per second '
                           'for all 14 queries on the test videos, including frame loading. UniVTG works on '
                           'precomputed M2CRL features (feature extraction not included); its parameters exclude '
                           'the frozen backbone. TeCNO: both stages (ResNet-50 and TCN; its throughput includes the '
                           'ResNet-50 pass).'), '']

    # ---- literature (template: values must be copied from the cited papers)
    def phase_acc(key):
        r = phase_stats.get(key) if key else None
        if r is None:
            return 'n/a'
        return f"{pct(r['accuracy'])} (legacy relaxed {pct(r['relaxed']['accuracy'])}; video-wise F1 {pct(r['f1_videowise'])})"
    tecno = by_short('tecno', 'comparison')
    ours_col = 'Ours: accuracy on our frames'
    lit_rows = [{'Method': f'{protocol.COMPARISONS[k]["model"]} zero-shot (Yuan et al.)', 'Task': 'phase recognition',
                 'What matches': 'weights, authors\' prompts and image transform, test videos 49-80',
                 'What differs': 'our 1 fps frame extraction and labels; their own evaluation frames and script',
                 'Reported': 'fill in from the paper', ours_col: phase_acc(author.get(k))} for k in protocol.SURGVLP_ZERO_SHOT]
    lit_rows += [
        {'Method': 'TeCNO (Czempiel et al., MICCAI 2020)', 'Task': 'phase recognition',
         'What matches': 'split 40 / 8 / 32, architecture, losses, optimisers, epochs, 1 fps',
         'What differs': 'our re-implementation (see tecno.py: logits CE, epoch selection, augmentation library)',
         'Reported': 'fill in from the paper (state whether relaxed)', ours_col: phase_acc(tecno.name if tecno else None)},
        {'Method': 'Trans-SVNet (Gao et al., MICCAI 2021)', 'Task': 'phase recognition', 'What matches': '-',
         'What differs': 'split 40 train / 40 test (its test set includes our validation videos)',
         'Reported': 'fill in from the paper (relaxed)', ours_col: '-'},
        {'Method': 'Supervised tool presence (e.g. EndoNet)', 'Task': 'tool presence mAP', 'What matches': '-',
         'What differs': 'split and frames', 'Reported': 'fill in from the paper', ours_col: '-'},
    ]
    md += ['## 11. Published results (template)', '',
           write_table(out, 'literature_template', pd.DataFrame(lit_rows),
                       'Published Cholec80 results for context (to be completed)',
                       'Copy the numbers from the cited papers and state their protocol (split, frames, relaxed or '
                       'not). Our numbers are reproductions on our own extracted 1 fps frames and labels, with our '
                       'metric code; they are close to, but not the same as, the authors\' evaluation, so differences '
                       'of a few points need not indicate a different model.'), '']

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
def efficiency(m, root, missing=None):
    """One row of the efficiency table. `missing` (a dict) receives {journal: epochs without a row}."""
    row = {'Method': m.label, 'Parameters (M)': 'n/a', 'Trainable (M)': 'n/a', 'Training (h)': 'n/a',
           'Training clips': 'n/a', 'Optimiser steps': 'n/a', 'Throughput (frames/s)': 'n/a',
           'Peak training memory (GB)': 'n/a'}
    run = m.run
    counts = run / 'parameter_counts.json'
    if counts.exists():
        c = json.loads(counts.read_text())
        row['Parameters (M)'] = f"{sum(v['total'] for v in c.values()) / 1e6:.1f}"
        row['Trainable (M)'] = f"{sum(v['trainable'] for v in c.values()) / 1e6:.1f}"
    journals = [run / 'training_metrics.jsonl'] + sorted(run.glob('*/training_metrics.jsonl'))
    seconds, memory, clips, steps = 0.0, [], 0, 0
    for j in journals:
        if j.exists():
            epochs = set()
            for line in j.read_text().splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if isinstance(r.get('epoch'), int):
                    epochs.add(r['epoch'])
                seconds += float(r.get('train_seconds', 0) or 0) + float(r.get('val_seconds', 0) or 0)
                if r.get('peak_gpu_memory_gb') is not None:
                    memory.append(float(r['peak_gpu_memory_gb']))
                if r.get('train_clips') is not None:           # clips through the backbone (ours, closed set)
                    clips += int(r['train_clips'])
                    steps += int(r.get('optimizer_steps') or 0)
            gaps = sorted(set(range(1, max(epochs) + 1)) - epochs) if epochs else []
            if gaps and missing is not None:
                missing[str(j.relative_to(root))] = gaps
    if seconds == 0:
        marker = root / 'state' / f'train.{m.name}.done.json'
        if marker.exists():
            seconds = float(json.loads(marker.read_text()).get('seconds', 0))
    if m.kind in protocol.ZERO_SHOT:
        row['Training (h)'] = '0 (zero-shot)'
        row['Trainable (M)'] = '0'
    elif seconds:
        row['Training (h)'] = f'{seconds / 3600:.1f}'
    if memory:
        row['Peak training memory (GB)'] = f'{max(memory):.1f}'
    if clips:
        row['Training clips'] = f'{clips:,}'
        row['Optimiser steps'] = f'{steps:,}'
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
    base = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860', '#DA8BC3', '#8C8C8C', '#CCB974',
            '#64B5CD', '#1B9E77', '#E7298A']
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
    fig, ax = plt.subplots(figsize=(12, 3.6))
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
