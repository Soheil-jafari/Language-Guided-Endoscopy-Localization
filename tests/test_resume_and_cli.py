import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch


def test_epoch_resume_reproduces_next_optimizer_step(tmp_path):
    from train import save_checkpoint,load_checkpoint
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.config=SimpleNamespace(test='resume')
            self.linear=torch.nn.Linear(3,1); self.drop=torch.nn.Dropout(.4)
        def forward(self,x): return self.linear(self.drop(x))
    def setup():
        model=Model(); optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer,1,.9)
        return model,optimizer,scheduler,torch.amp.GradScaler('cuda',enabled=False)
    def step(model,optimizer,scheduler):
        x=torch.randn(4,3); y=torch.tensor(np.random.rand(4,1),dtype=torch.float32)+random.random()
        optimizer.zero_grad(); (model(x)-y).square().mean().backward(); optimizer.step(); scheduler.step()
    a,opt,sched,scaler=setup(); step(a,opt,sched)
    save_checkpoint(a,opt,sched,scaler,1,.25,tmp_path/'latest.pt',dict(provenance={'data':'hash'}))
    step(a,opt,sched)
    b,opt2,sched2,scaler2=setup()
    ckpt=load_checkpoint(tmp_path/'latest.pt',b,opt2,sched2,scaler2,expected_provenance={'data':'hash'})
    assert ckpt['best_val_loss']==.25 and ckpt['epoch']==1
    step(b,opt2,sched2)
    for p,q in zip(a.parameters(),b.parameters()): torch.testing.assert_close(p,q,rtol=0,atol=0)
    assert opt.param_groups[0]['lr']==opt2.param_groups[0]['lr']
    with pytest.raises(ValueError,match='manifests'):
        load_checkpoint(tmp_path/'latest.pt',b,expected_provenance={'data':'changed'})


def test_unsupported_architectures_fail_explicitly(monkeypatch):
    import sys
    from models import LocalizationFramework,TemporalHeadSSM,CustomSSMBlock
    from project_config import Config
    cfg=Config(); cfg.MODEL.VISION_BACKBONE_NAME='EndoMamba'
    with pytest.raises(NotImplementedError): LocalizationFramework(cfg)
    monkeypatch.setitem(sys.modules,'mamba_ssm',None)
    with pytest.raises(RuntimeError,match='Official Mamba'): TemporalHeadSSM(8,1)
    block=CustomSSMBlock(8,d_state=2)
    output=block(torch.randn(2,3,8)); output.sum().backward()
    assert all(p.grad is not None for p in block.parameters())


def test_raw_annotation_merge_does_not_invent_tool_negatives(tmp_path):
    from dataset_preprocessing.build_dataset import parse_annotations
    from data_contract import AnnotationIndex
    phases=tmp_path/'phases'; tools=tmp_path/'tools'; phases.mkdir(); tools.mkdir()
    (phases/'video01-phase.txt').write_text('Frame\tPhase\n0\tPreparation\n1\tPreparation\n25\tCalotTriangleDissection\n')
    (tools/'video01-tool.txt').write_text('Frame\tGrasper\tBipolar\tHook\tScissors\tClipper\tIrrigator\tSpecimenBag\n0\t1\t0\t0\t0\t0\t0\t0\n25\t0\t0\t1\t0\t0\t0\t0\n')
    output=tmp_path/'parsed.csv'; parse_annotations(phases,tools,output)
    ann=AnnotationIndex(output)
    assert ann.label('CHOLEC80__video01',1,('tool',0))==-100
    assert ann.label('CHOLEC80__video01',25,('tool',0))==0


def test_evaluation_cli_validation_calibration_and_test_guard(tmp_path,monkeypatch):
    import sys
    from evaluate import main
    ref=pd.DataFrame(dict(video_id=['v']*3,text_query=['q']*3,frame_idx=[0,25,50],label=[0,1,1]))
    pred=ref.copy(); pred['score']=[.1,.8,.9]
    ref.to_csv(tmp_path/'ref.csv',index=False); pred.to_csv(tmp_path/'pred.csv',index=False)
    base=['evaluate.py','--predictions',str(tmp_path/'pred.csv'),'--reference',str(tmp_path/'ref.csv')]
    monkeypatch.setattr(sys,'argv',base+['--split','validation','--select-threshold','--output',str(tmp_path/'val.json')]); main()
    monkeypatch.setattr(sys,'argv',base+['--split','test','--calibration',str(tmp_path/'val.json'),'--output',str(tmp_path/'test.json')]); main()
    assert json.loads((tmp_path/'test.json').read_text())['f1']==1
    monkeypatch.setattr(sys,'argv',base+['--split','test','--select-threshold','--output',str(tmp_path/'invalid.json')])
    with pytest.raises(ValueError,match='validation'): main()
    pred.iloc[:-1].to_csv(tmp_path/'pred.csv',index=False)
    monkeypatch.setattr(sys,'argv',base+['--split','test','--output',str(tmp_path/'missing.json')])
    with pytest.raises(ValueError,match='keys'): main()


def test_segment_export_round_trip_and_absent_penalty(tmp_path,monkeypatch):
    import sys
    from export_segments import main
    from evaluate import segment_metrics
    metadata={'v':dict(source_fps=25,frame_count=100)}
    (tmp_path/'meta.json').write_text(json.dumps(metadata))
    refs=[dict(video='v',query=q,duration=4,timestamps=spans) for q,spans in [('present',[[1,3]]),('absent',[])]]
    (tmp_path/'ref.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in refs))
    rows=[]
    for q,scores in [('present',[.1,.8,.8,.1]),('absent',[.9,.1,.1,.1])]:
        for idx,score in zip([0,25,50,75],scores): rows.append(dict(video_id='v',text_query=q,frame_idx=idx,score=score))
    pd.DataFrame(rows).to_csv(tmp_path/'pred.csv',index=False)
    monkeypatch.setattr(sys,'argv',['export_segments.py','--predictions',str(tmp_path/'pred.csv'),'--reference',str(tmp_path/'ref.jsonl'),
        '--metadata',str(tmp_path/'meta.json'),'--threshold','.5','--output',str(tmp_path/'segments.jsonl')])
    main(); records=[json.loads(s) for s in (tmp_path/'segments.jsonl').read_text().splitlines()]
    metrics=segment_metrics(records)
    assert metrics['pooled_AP@0.5']==.5
    assert metrics['absent_query_false_positive_rate']==1
    assert metrics['R1@0.5']==1
