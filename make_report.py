#!/usr/bin/env python3
import os, csv, argparse, math, sys, glob
import numpy as np
import matplotlib.pyplot as plt

try:
    from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
    HAVE_SK = True
except Exception:
    HAVE_SK = False

def parse_spans(span_strs):
    spans = []
    for s in (span_strs or []):
        s = s.strip().replace(" ", "")
        if not s:
            continue
        parts = s.split(";") if ";" in s else [s]
        for p in parts:
            a, b = p.split(",")
            spans.append((float(a), float(b)))
    return spans

def load_scores(run_dir):
    path = os.path.join(run_dir, "scores.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"scores.csv not found in {run_dir}")
    t, p = [], []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:

            t.append(float(row["time_sec"]))
            p.append(float(row["prob"]))
    t = np.array(t, dtype=np.float32)
    p = np.array(p, dtype=np.float32)
    return t, p

def load_pred_segments(run_dir):
    # Prefer frozen segments if present
    frozen = os.path.join(run_dir, "pred_segments.frozen.csv")
    path = frozen if os.path.exists(frozen) else os.path.join(run_dir, "pred_segments.csv")
    segs = []
    if os.path.exists(path):
        with open(path) as f:
            r = csv.DictReader(f)
            for row in r:
                segs.append((float(row["start_sec"]), float(row["end_sec"]), float(row.get("score", 0.0))))
    return segs


def labels_from_spans(times, spans):
    y = np.zeros_like(times, dtype=np.int32)
    for (a,b) in spans:
        y |= ((times >= a) & (times < b)).astype(np.int32)
    return y

def basic_metrics(y_true, y_pred):
    tp = int(((y_true==1)&(y_pred==1)).sum())
    fp = int(((y_true==0)&(y_pred==1)).sum())
    fn = int(((y_true==1)&(y_pred==0)).sum())
    tn = int(((y_true==0)&(y_pred==0)).sum())
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    acc  = (tp+tn)/max(len(y_true),1)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=f1, accuracy=acc)

def threshold_sweep(y_true, y_prob, grid=None):
    grid = grid if grid is not None else np.linspace(0.05, 0.95, 19, dtype=np.float32)
    rows = []
    best = dict(f1=-1.0, thr=None, metrics=None)
    for thr in grid:
        pred = (y_prob >= thr).astype(np.int32)
        m = basic_metrics(y_true, pred)
        rows.append((float(thr), m["accuracy"], m["precision"], m["recall"], m["f1"]))
        if (m["f1"] > best["f1"]) or (m["f1"] == best["f1"] and m["recall"] > (best["metrics"]["recall"] if best["metrics"] else -1)):
            best = dict(f1=m["f1"], thr=float(thr), metrics=m)
    return best, rows

def tiou(seg, gt):
    # temporal IoU between [s,e] and [a,b]
    s,e = seg
    a,b = gt
    inter = max(0.0, min(e,b) - max(s,a))
    union = max(e,s) - min(a,b)
    if union <= 0: return 0.0
    return inter / union

def pick_best_segment(pred_segments, gt_spans):
    if not pred_segments:
        return None, 0.0, None
    # candidate list (by tIoU to GT spans)
    best = (None, -1.0, None)  # (seg, best_tiou, best_gt)
    for (s,e,sc) in pred_segments:
        for gt in gt_spans:
            t = tiou((s,e), gt)
            if t > best[1]:
                best = ((s,e,sc), t, gt)
    # if no overlap at all, choose the longest predicted segment
    if best[1] <= 0.0:
        longest = max(pred_segments, key=lambda x: (x[1]-x[0]))
        return longest, 0.0, None
    return best

def plot_timeline(times, probs, gt_spans, pred_segments, thr, out_path):
    plt.figure(figsize=(12,3))
    plt.plot(times, probs, linewidth=1)
    if thr is not None:
        plt.axhline(thr, linestyle="--")
    # GT spans
    for (a,b) in gt_spans:
        plt.axvspan(a, b, alpha=0.2, color="green", label="GT" if a==gt_spans[0][0] else None)
    # predicted segments
    for i,(s,e,_) in enumerate(pred_segments):
        plt.axvspan(s, e, alpha=0.2, color="tab:blue", label="Pred" if i==0 else None)
    plt.xlabel("Time (s)"); plt.ylabel("Probability")
    plt.title("Timeline: probability, GT (green), predicted segments (blue)")
    handles, labels = plt.gca().get_legend_handles_labels()
    if labels:
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

def plot_pr(y_true, y_prob, out_path):
    if not HAVE_SK:
        return
    p, r, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(4.2,3.5))
    plt.plot(r, p)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR (AP={ap:.3f})")
    plt.tight_layout(); plt.savefig(out_path, dpi=180); plt.close()

def main():
    ap = argparse.ArgumentParser("Make report tables/figures from run directories")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="One or more run directories (each containing scores.csv and pred_segments.csv)")
    ap.add_argument("--gt_span", type=str, action="append", required=True,
                    help="GT spans like '64,1015'. Repeat flag for multiple spans or separate with ';'")
    ap.add_argument("--thr", type=float, default=None,
                    help="Operating threshold for metrics/figures. If None, choose best-F1 per run.")
    args = ap.parse_args()

    gt_spans = parse_spans(args.gt_span)
    rows_out = []
    for run_dir in args.runs:
        name = os.path.basename(run_dir.rstrip("/"))
        out_dir = run_dir  # save figs next to CSVs

        # load
        t, p = load_scores(run_dir)
        y_true = labels_from_spans(t, gt_spans)
        # choose threshold
        if args.thr is None:
            best, sweep = threshold_sweep(y_true, p)
            thr = best["thr"]; m = best["metrics"]; f1 = best["f1"]
        else:
            thr = float(args.thr)
            m = basic_metrics(y_true, (p >= thr).astype(np.int32))
            f1 = m["f1"]

        # AP/AUC
        if HAVE_SK:
            try:
                ap_score = float(average_precision_score(y_true, p))
            except Exception:
                ap_score = float("nan")
            try:
                auc = float(roc_auc_score(y_true, p)) if (y_true.min()!=y_true.max()) else float("nan")
            except Exception:
                auc = float("nan")
        else:
            ap_score, auc = float("nan"), float("nan")

        # segments & best segment
        pred_segments = load_pred_segments(run_dir)
        best_seg, best_tiou, best_gt = pick_best_segment(pred_segments, gt_spans)

        # figures
        timeline_png = os.path.join(out_dir, "timeline.png")
        plot_timeline(t, p, gt_spans, pred_segments, thr, timeline_png)
        if HAVE_SK:
            pr_png = os.path.join(out_dir, "pr_curve.png")
            plot_pr(y_true, p, pr_png)
        # if you saved metrics_over_thresholds.csv earlier, optionally plot F1 vs thr
        mot_path = os.path.join(run_dir, "metrics_over_thresholds.csv")
        if os.path.exists(mot_path):
            xs, f1s = [], []
            with open(mot_path) as f:
                r = csv.DictReader(f)
                for row in r:
                    xs.append(float(row["threshold"]))
                    f1s.append(float(row["f1"]))
            plt.figure(figsize=(4.2,3.5))
            plt.plot(xs, f1s)
            plt.xlabel("Threshold"); plt.ylabel("F1"); plt.title("F1 vs Threshold")
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, "f1_vs_threshold.png"), dpi=180); plt.close()

        # summary row
        pos = int((y_true==1).sum()); neg = int((y_true==0).sum())
        row = dict(
            run=name, thr=thr,
            precision=m["precision"], recall=m["recall"], f1=f1, accuracy=m["accuracy"],
            AP=ap_score, AUC=auc, pos=pos, neg=neg,
            best_tIoU=best_tiou,
            best_seg_start=(best_seg[0] if best_seg else float("nan")),
            best_seg_end=(best_seg[1] if best_seg else float("nan")),
            best_seg_score=(best_seg[2] if best_seg else float("nan"))
        )
        rows_out.append(row)

    # write summary
    hdr = ["run","thr","precision","recall","f1","accuracy","AP","AUC","pos","neg",
           "best_tIoU","best_seg_start","best_seg_end","best_seg_score"]
    summary_path = os.path.join(os.path.dirname(args.runs[0]), "summary.csv") \
                   if len(args.runs)>1 else os.path.join(args.runs[0], "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader(); w.writerows(rows_out)

    print(f"Saved summary to: {summary_path}")
    for r in rows_out:
        print(r)

if __name__ == "__main__":
    main()
