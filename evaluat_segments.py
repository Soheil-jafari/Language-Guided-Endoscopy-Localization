import argparse, csv, json, os
from pathlib import Path
import numpy as np

# Basic IO helpers

def read_segments(csv_path):
    segs = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            # allow optional score column in preds
            s = float(row["start_sec"]); e = float(row["end_sec"])
            sc = float(row.get("score", 1.0)) if "score" in row else 1.0
            segs.append((s, e, sc))
    return segs

# Segment metrics

def tiou(a, b):
    s1, e1, _ = a; s2, e2, _ = b
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return inter / union if union > 0 else 0.0


def pr_f1_at_threshold(preds, gts, thr):
    preds = sorted(preds, key=lambda x: x[2], reverse=True)
    used = [False] * len(gts)
    tp = fp = 0
    for p in preds:
        best_i = -1; best_t = 0.0
        for i, g in enumerate(gts):
            if used[i]:
                continue
            t = tiou(p, g)
            if t > best_t:
                best_t = t; best_i = i
        if best_t >= thr and best_i != -1:
            used[best_i] = True; tp += 1
        else:
            fp += 1
    fn = used.count(False)
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec  = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "precision": prec, "recall": rec, "f1": f1}


def average_precision(preds, gts, thr):
    preds = sorted(preds, key=lambda x: x[2], reverse=True)
    used = [False] * len(gts)
    tps, fps = [], []
    for p in preds:
        best_i = -1; best_t = 0.0
        for i, g in enumerate(gts):
            if used[i]:
                continue
            t = tiou(p, g)
            if t > best_t:
                best_t = t; best_i = i
        if best_t >= thr and best_i != -1:
            used[best_i] = True; tps.append(1); fps.append(0)
        else:
            tps.append(0); fps.append(1)
    cum_tp = 0; cum_fp = 0; prec = []; rec = []
    total = len(gts)
    for tp_i, fp_i in zip(tps, fps):
        cum_tp += tp_i; cum_fp += fp_i
        p = cum_tp / max(1, (cum_tp + cum_fp))
        r = cum_tp / max(1, total)
        prec.append(p); rec.append(r)
    ap = 0.0
    for r_th in [i / 10 for i in range(11)]:
        pmax = max([p for p, r in zip(prec, rec) if r >= r_th], default=0.0)
        ap += pmax
    return ap / 11.0


def write_side_by_side(preds, gts, out_csv, thr):
    used = [False] * len(gts)
    rows = []
    for (ps, pe, sc) in sorted(preds, key=lambda x: x[2], reverse=True):
        best_i = -1; best_t = 0.0
        for i, (gs, ge, _) in enumerate(gts):
            if used[i]:
                continue
            t = tiou((ps, pe, sc), (gs, ge, 1.0))
            if t > best_t:
                best_t = t; best_i = i
        status = "OK" if best_i != -1 and best_t >= thr else "MISS"
        gs = ge = ""
        if best_i != -1:
            gs, ge, _ = gts[best_i]
            if status == "OK":
                used[best_i] = True
        rows.append([ps, pe, sc, gs, ge, best_t, status])
    for i, (gs, ge, _) in enumerate(gts):
        if not used[i]:
            rows.append(["", "", "", gs, ge, "", "FN"])
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pred_start", "pred_end", "pred_score", "gt_start", "gt_end", "tIoU", "status"])
        w.writerows(rows)


# Per-frame metrics

def brier_score(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i < n_bins - 1:
            m = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        else:
            m = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
        if not np.any(m):
            continue
        conf = float(np.mean(y_prob[m]))
        acc  = float(np.mean(y_true[m]))
        w    = float(np.mean(m))
        ece += w * abs(acc - conf)
    return float(ece)


def aurc(y_true, y_prob, uncertainty=None):
    """
    Area Under Risk-Coverage.
    If 'uncertainty' provided: sort ascending by uncertainty (lowest first).
    Otherwise: sort by confidence_desc (highest prob first).
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    if uncertainty is not None:
        order = np.argsort(uncertainty)
    else:
        order = np.argsort(-y_prob)

    y_true_ord = y_true[order]
    y_prob_ord = y_prob[order]

    risks, covs = [], []
    correct = 0
    for k in range(1, len(y_true_ord) + 1):
        pred = (y_prob_ord[:k] >= 0.5).astype(np.int32)
        correct = int(np.sum(pred == y_true_ord[:k]))
        acc = correct / k
        risks.append(1.0 - acc)
        covs.append(k / len(y_true_ord))

    area = 0.0
    for i in range(1, len(covs)):
        area += 0.5 * (risks[i] + risks[i - 1]) * (covs[i] - covs[i - 1])
    return float(area)


# Artifact loading

def _pick_first_exists(base_dir, candidates):
    for name in candidates:
        p = Path(base_dir) / name
        if p.exists():
            return p
    # default to the last candidate
    return Path(base_dir) / candidates[-1]


def load_infer_artifacts(infer_dir, suffix=""):
    """
    Returns:
      preds: list[(start,end,score)]
      pf: dict or None (per-frame arrays)
      chosen_paths: dict with keys used for naming outputs
    """
    if suffix:
        sfx = f"_{suffix.strip()}"
        per_frame_csv = Path(infer_dir) / f"per_frame{sfx}.csv"
        seg_json      = Path(infer_dir) / f"segments{sfx}.json"
        seg_csv       = Path(infer_dir) / f"pred_segments{sfx}.csv"
        # fallback if not found
        if not per_frame_csv.exists():
            per_frame_csv = _pick_first_exists(infer_dir, [f"per_frame{sfx}.csv", "per_frame_improved.csv",
                                                           "per_frame_baseline.csv", "per_frame.csv"])
        if not seg_json.exists():
            seg_json = _pick_first_exists(infer_dir, [f"segments{sfx}.json", "segments_improved.json",
                                                      "segments_baseline.json", "segments.json"])
        if not seg_csv.exists():
            seg_csv = _pick_first_exists(infer_dir, [f"pred_segments{sfx}.csv", "pred_segments_improved.csv",
                                                     "pred_segments_baseline.csv", "pred_segments.csv"])
    else:
        per_frame_csv = _pick_first_exists(infer_dir, ["per_frame_improved.csv",
                                                       "per_frame_baseline.csv", "per_frame.csv"])
        seg_json      = _pick_first_exists(infer_dir, ["segments_improved.json",
                                                       "segments_baseline.json", "segments.json"])
        seg_csv       = _pick_first_exists(infer_dir, ["pred_segments_improved.csv",
                                                       "pred_segments_baseline.csv", "pred_segments.csv"])

    # Load predictions
    preds = None
    if seg_json.exists():
        try:
            with open(seg_json, "r", encoding="utf-8") as f:
                j = json.load(f)
            preds = [(float(x["start_sec"]), float(x["end_sec"]), float(x.get("score", 1.0))) for x in j]
            pred_source = seg_json
        except Exception:
            preds = None

    if preds is None:
        if seg_csv.exists():
            preds = read_segments(seg_csv)
            pred_source = seg_csv
        else:
            preds = []
            pred_source = seg_csv

    # Load per-frame if present
    pf = None
    if per_frame_csv.exists():
        rows = []
        with open(per_frame_csv, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                rows.append(row)
        if len(rows) > 0:
            frame_idx = np.array([int(r["frame_idx"]) for r in rows], dtype=np.int64)
            time_sec  = np.array([float(r["time_sec"]) for r in rows], dtype=np.float32)
            prob      = np.array([float(r["prob"]) for r in rows], dtype=np.float32)
            alpha = beta = unc = None
            if "alpha" in rows[0]:
                try:
                    alpha = np.array([float(r["alpha"]) for r in rows], dtype=np.float32)
                except Exception:
                    alpha = None
            if "beta" in rows[0]:
                try:
                    beta = np.array([float(r["beta"]) for r in rows], dtype=np.float32)
                except Exception:
                    beta = None
            if "uncertainty" in rows[0]:
                try:
                    unc = np.array([float(r["uncertainty"]) for r in rows], dtype=np.float32)
                except Exception:
                    unc = None
            pf = {"frame_idx": frame_idx, "time_sec": time_sec, "prob": prob,
                  "alpha": alpha, "beta": beta, "uncertainty": unc, "per_frame_path": per_frame_csv}

    chosen_paths = {
        "per_frame_csv": per_frame_csv,
        "seg_json": seg_json,
        "seg_csv": seg_csv,
        "pred_source": pred_source
    }
    return preds, pf, chosen_paths


# CLI + main

def main():
    ap = argparse.ArgumentParser()
    # Either use --inference_dir (preferred) or --pred_csv
    ap.add_argument("--inference_dir", type=str, default=None,
                    help="Directory from inference.py containing per_frame*.csv + segments*.json/csv.")
    ap.add_argument("--pred_csv", type=str, default=None,
                    help="Predicted segments CSV (start_sec,end_sec[,score]) if not using --inference_dir.")
    ap.add_argument("--gt_csv", required=True, type=str,
                    help="Ground-truth CSV (start_sec,end_sec).")
    ap.add_argument("--thr", type=float, default=0.5,
                    help="tIoU threshold for segment matching (default 0.5).")
    ap.add_argument("--suffix", type=str, default="",
                    help="If your inference artifacts used a suffix (e.g., 'improved'/'baseline'), provide it.")
    ap.add_argument("--gt_spans", type=str, action="append", default=None,
                    help="Add one or more frame-level GT intervals 'start,end' in seconds; repeatable.")
    ap.add_argument("--out_json", type=str, default=None,
                    help="Where to save metrics JSON. Default: alongside preds as metrics[_{suffix}].json")
    ap.add_argument("--ece_bins", type=int, default=10,
                    help="Number of bins for ECE (default 10).")
    args = ap.parse_args()

    # Load predictions & per-frame artifacts
    pf = None
    pred_source = None
    sfx = f"_{args.suffix.strip()}" if args.suffix else ""

    if args.inference_dir:
        # reuse your existing helper
        _, pf, chosen = load_infer_artifacts(args.inference_dir, args.suffix)
        pred_source = chosen["pred_source"]  # only used for naming outputs if --pred_csv not provided

    # Load segments (preds): prefer --pred_csv (frozen)
    if args.pred_csv:
        preds = read_segments(args.pred_csv)
        pred_source = Path(args.pred_csv)
    else:
        if not args.inference_dir:
            raise SystemExit("Error: provide --pred_csv (frozen) or --inference_dir.")
        preds, _, chosen = load_infer_artifacts(args.inference_dir, args.suffix)
        pred_source = chosen["pred_source"]

    # Load GT segments
    gts = [(s, e, 1.0) for (s, e, _) in read_segments(args.gt_csv)]

    # Segment metrics at multiple IoU thresholds
    report = {}
    for thr in (0.3, 0.5, 0.7):
        report[f"PRF@{thr}"] = pr_f1_at_threshold(preds, gts, thr)
        report[f"mAP@{thr}"] = average_precision(preds, gts, thr)

    # include the exact --thr used for side-by-side
    report[f"PRF@{args.thr}"] = pr_f1_at_threshold(preds, gts, args.thr)

    # Optional: frame-level metrics if we have per_frame.csv and --gt_spans
    if pf is not None and args.gt_spans:
        # parse spans
        spans = []
        for s in args.gt_spans:
            s = s.strip().replace(" ", "")
            if not s:
                continue
            # allow "a,b;c,d"
            parts = s.split(";") if ";" in s else [s]
            for p in parts:
                a, b = p.split(",")
                spans.append((float(a), float(b)))

        times = pf["time_sec"]
        y_true = np.zeros_like(times, dtype=np.int32)
        for (a, b) in spans:
            y_true |= ((times >= a) & (times < b)).astype(np.int32)
        y_prob = pf["prob"]

        # Best-F1 threshold sweep
        thrs = np.linspace(0.05, 0.95, 19, dtype=np.float32)
        best = {"f1": -1, "thr": None, "precision": 0.0, "recall": 0.0}
        for t in thrs:
            y_pred = (y_prob >= t).astype(np.int32)
            tp = int(((y_true == 1) & (y_pred == 1)).sum())
            fp = int(((y_true == 0) & (y_pred == 1)).sum())
            fn = int(((y_true == 1) & (y_pred == 0)).sum())
            prec = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0
            rec  = tp / max(tp + fn, 1) if (tp + fn) > 0 else 0.0
            f1   = (2 * prec * rec / max(prec + rec, 1e-12)) if (prec + rec) > 0 else 0.0
            if f1 > best["f1"] or (f1 == best["f1"] and rec > best["recall"]):
                best = {"f1": f1, "thr": float(t), "precision": prec, "recall": rec}

        # AP / ROC-AUC (optional sklearn)
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score
            ap = float(average_precision_score(y_true, y_prob))
            auc = float(roc_auc_score(y_true, y_prob)) if (y_true.min() == 0 and y_true.max() == 1) else float("nan")
        except Exception:
            ap = float("nan"); auc = float("nan")

        # Calibration & risk-coverage
        ece = expected_calibration_error(y_true, y_prob, n_bins=int(args.ece_bins))
        br  = brier_score(y_true, y_prob)
        unc = pf.get("uncertainty", None)
        aurc_unc = aurc(y_true, y_prob, uncertainty=unc) if unc is not None else float("nan")
        aurc_conf = aurc(y_true, y_prob, uncertainty=None)

        report["frame_metrics"] = {
            "best_f1": best["f1"],
            "best_thr": best["thr"],
            "precision": best["precision"],
            "recall": best["recall"],
            "AP": ap,
            "AUC": auc,
            "Brier": br,
            "ECE_bins": int(args.ece_bins),
            "ECE": ece,
            "AURC_uncertainty": aurc_unc,
            "AURC_confidence": aurc_conf,
        }

    # Side-by-side CSV near the prediction source
    pred_source = Path(pred_source)
    out_csv = pred_source.with_name(f"pred_vs_gt{sfx}.csv")
    write_side_by_side(preds, gts, out_csv, args.thr)

    # Metrics JSON alongside predictions
    if args.out_json:
        out_json = Path(args.out_json)
    else:
        out_json = pred_source.with_name(f"metrics{sfx}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Console summary
    print(json.dumps(report, indent=2))
    print(f"[OK] wrote: {out_csv}")
    print(f"[OK] wrote: {out_json}")


if __name__ == "__main__":
    main()
