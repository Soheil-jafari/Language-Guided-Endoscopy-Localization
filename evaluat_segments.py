# evaluate_segments.py
import argparse, csv, json
from pathlib import Path

def read_segments(csv_path):
    segs=[]
    with open(csv_path,"r",encoding="utf-8") as f:
        r=csv.DictReader(f)
        for row in r:
            # allow optional score column in preds
            s=float(row["start_sec"]); e=float(row["end_sec"])
            sc=float(row.get("score",1.0)) if "score" in row else 1.0
            segs.append((s,e,sc))
    return segs

def tiou(a,b):
    s1,e1,_=a; s2,e2,_=b
    inter=max(0.0, min(e1,e2)-max(s1,s2))
    union=max(e1,e2)-min(s1,s2)
    return inter/union if union>0 else 0.0

def pr_f1_at_threshold(preds,gts,thr):
    preds=sorted(preds,key=lambda x:x[2],reverse=True)
    used=[False]*len(gts)
    tp=fp=0
    for p in preds:
        best_i=-1; best_t=0.0
        for i,g in enumerate(gts):
            if used[i]: continue
            t=tiou(p,g)
            if t>best_t: best_t=t; best_i=i
        if best_t>=thr and best_i!=-1:
            used[best_i]=True; tp+=1
        else:
            fp+=1
    fn=used.count(False)
    prec=tp/(tp+fp) if tp+fp>0 else 0.0
    rec =tp/(tp+fn) if tp+fn>0 else 0.0
    f1  =2*prec*rec/(prec+rec) if prec+rec>0 else 0.0
    return {"TP":tp,"FP":fp,"FN":fn,"precision":prec,"recall":rec,"f1":f1}

def average_precision(preds,gts,thr):
    preds=sorted(preds,key=lambda x:x[2],reverse=True)
    used=[False]*len(gts)
    tps=[]; fps=[]
    for p in preds:
        best_i=-1; best_t=0.0
        for i,g in enumerate(gts):
            if used[i]: continue
            t=tiou(p,g)
            if t>best_t: best_t=t; best_i=i
        if best_t>=thr and best_i!=-1:
            used[best_i]=True; tps.append(1); fps.append(0)
        else:
            tps.append(0); fps.append(1)
    cum_tp=0; cum_fp=0; prec=[]; rec=[]
    total=len(gts)
    for tp_i,fp_i in zip(tps,fps):
        cum_tp+=tp_i; cum_fp+=fp_i
        p=cum_tp/max(1,(cum_tp+cum_fp))
        r=cum_tp/max(1,total)
        prec.append(p); rec.append(r)
    ap=0.0
    for r_th in [i/10 for i in range(11)]:
        pmax=max([p for p,r in zip(prec,rec) if r>=r_th], default=0.0)
        ap+=pmax
    return ap/11.0

def write_side_by_side(preds,gts,out_csv,thr):
    used=[False]*len(gts)
    rows=[]
    for (ps,pe,sc) in sorted(preds,key=lambda x:x[2],reverse=True):
        best_i=-1; best_t=0.0
        for i,(gs,ge,_) in enumerate(gts):
            if used[i]: continue
            t=tiou((ps,pe,sc),(gs,ge,1.0))
            if t>best_t: best_t=t; best_i=i
        status="OK" if best_i!=-1 and best_t>=thr else "MISS"
        gs=ge=""
        if best_i!=-1:
            gs,ge,_=gts[best_i]
            if status=="OK": used[best_i]=True
        rows.append([ps,pe,sc,gs,ge,best_t,status])
    for i,(gs,ge,_) in enumerate(gts):
        if not used[i]:
            rows.append(["","", "", gs,ge, "", "FN"])
    with open(out_csv,"w",encoding="utf-8",newline="") as f:
        w=csv.writer(f); w.writerow(
            ["pred_start","pred_end","pred_score","gt_start","gt_end","tIoU","status"]
        ); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True, help="Predicted segments CSV (start_sec,end_sec[,score])")
    ap.add_argument("--gt_csv", required=True, help="Ground-truth CSV (start_sec,end_sec)")
    ap.add_argument("--thr", type=float, default=0.5)
    args=ap.parse_args()
    preds=read_segments(args.pred_csv)
    gts=[(s,e,1.0) for (s,e,_) in read_segments(args.gt_csv)]
    report={}# evaluate_segments.py
import argparse, csv, json
from pathlib import Path

def read_segments(csv_path):
    segs=[]
    with open(csv_path,"r",encoding="utf-8") as f:
        r=csv.DictReader(f)
        for row in r:
            # allow optional score column in preds
            s=float(row["start_sec"]); e=float(row["end_sec"])
            sc=float(row.get("score",1.0)) if "score" in row else 1.0
            segs.append((s,e,sc))
    return segs

def tiou(a,b):
    s1,e1,_=a; s2,e2,_=b
    inter=max(0.0, min(e1,e2)-max(s1,s2))
    union=max(e1,e2)-min(s1,s2)
    return inter/union if union>0 else 0.0

def pr_f1_at_threshold(preds,gts,thr):
    preds=sorted(preds,key=lambda x:x[2],reverse=True)
    used=[False]*len(gts)
    tp=fp=0
    for p in preds:
        best_i=-1; best_t=0.0
        for i,g in enumerate(gts):
            if used[i]: continue
            t=tiou(p,g)
            if t>best_t: best_t=t; best_i=i
        if best_t>=thr and best_i!=-1:
            used[best_i]=True; tp+=1
        else:
            fp+=1
    fn=used.count(False)
    prec=tp/(tp+fp) if tp+fp>0 else 0.0
    rec =tp/(tp+fn) if tp+fn>0 else 0.0
    f1  =2*prec*rec/(prec+rec) if prec+rec>0 else 0.0
    return {"TP":tp,"FP":fp,"FN":fn,"precision":prec,"recall":rec,"f1":f1}

def average_precision(preds,gts,thr):
    preds=sorted(preds,key=lambda x:x[2],reverse=True)
    used=[False]*len(gts)
    tps=[]; fps=[]
    for p in preds:
        best_i=-1; best_t=0.0
        for i,g in enumerate(gts):
            if used[i]: continue
            t=tiou(p,g)
            if t>best_t: best_t=t; best_i=i
        if best_t>=thr and best_i!=-1:
            used[best_i]=True; tps.append(1); fps.append(0)
        else:
            tps.append(0); fps.append(1)
    cum_tp=0; cum_fp=0; prec=[]; rec=[]
    total=len(gts)
    for tp_i,fp_i in zip(tps,fps):
        cum_tp+=tp_i; cum_fp+=fp_i
        p=cum_tp/max(1,(cum_tp+cum_fp))
        r=cum_tp/max(1,total)
        prec.append(p); rec.append(r)
    ap=0.0
    for r_th in [i/10 for i in range(11)]:
        pmax=max([p for p,r in zip(prec,rec) if r>=r_th], default=0.0)
        ap+=pmax
    return ap/11.0

def write_side_by_side(preds,gts,out_csv,thr):
    used=[False]*len(gts)
    rows=[]
    for (ps,pe,sc) in sorted(preds,key=lambda x:x[2],reverse=True):
        best_i=-1; best_t=0.0
        for i,(gs,ge,_) in enumerate(gts):
            if used[i]: continue
            t=tiou((ps,pe,sc),(gs,ge,1.0))
            if t>best_t: best_t=t; best_i=i
        status="OK" if best_i!=-1 and best_t>=thr else "MISS"
        gs=ge=""
        if best_i!=-1:
            gs,ge,_=gts[best_i]
            if status=="OK": used[best_i]=True
        rows.append([ps,pe,sc,gs,ge,best_t,status])
    for i,(gs,ge,_) in enumerate(gts):
        if not used[i]:
            rows.append(["","", "", gs,ge, "", "FN"])
    with open(out_csv,"w",encoding="utf-8",newline="") as f:
        w=csv.writer(f); w.writerow(
            ["pred_start","pred_end","pred_score","gt_start","gt_end","tIoU","status"]
        ); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True, help="Predicted segments CSV (start_sec,end_sec[,score])")
    ap.add_argument("--gt_csv", required=True, help="Ground-truth CSV (start_sec,end_sec)")
    ap.add_argument("--thr", type=float, default=0.5)
    args=ap.parse_args()
    preds=read_segments(args.pred_csv)
    gts=[(s,e,1.0) for (s,e,_) in read_segments(args.gt_csv)]
    report={}
    for thr in (0.3,0.5,0.7):
        report[f"PRF@{thr}"]=pr_f1_at_threshold(preds,gts,thr)
        report[f"mAP@{thr}"]=average_precision(preds,gts,thr)
    print(json.dumps(report, indent=2))
    out_csv=Path(args.pred_csv).with_name("pred_vs_gt.csv")
    write_side_by_side(preds,gts,out_csv,args.thr)
    print(f"[OK] wrote: {out_csv}")

if __name__=="__main__":
    main()
