import copy
import json
import random
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from data_contract import AnnotationIndex,query_spec,sample_indices,window_positions,FrameStore,validate_triplets
from losses import EvidentialLoss,MasterLoss,backward_warp
from checkpoint_utils import atomic_save,extract_model_state,load_model_state,rng_state,restore_rng
from metrics import binary_metrics,temporal_ap,segments_from_scores,deduplicate_frame_rows
from project_config import Config


def test_aliases_and_explicit_paraphrases():
    assert query_spec('calot')==query_spec('CalotTriangleDissection')==('phase',1)
    assert query_spec('Show the surgeon opening the triangle','phase',1)==('phase',1)
    with pytest.raises(ValueError): query_spec('a wholly unlabelled event')
    df=pd.DataFrame([dict(frame_path='x/v/frame_00001.jpg',text_query=q,relevance_label=y) for q,y in [('calot',1),('CalotTriangleDissection',0)]])
    with pytest.raises(ValueError,match='Conflicting'): validate_triplets(df)


def test_sparse_tools_are_unknown_and_duplicates_conflict():
    df=pd.DataFrame([dict(standardized_video_id='v',frame_idx=0,original_label='Preparation',grasper=1),
                     dict(standardized_video_id='v',frame_idx=25,original_label='CalotTriangleDissection',grasper=0)])
    ann=AnnotationIndex(df)
    assert ann.label('v',1,('tool',0))==-100
    assert ann.label('v',25,('tool',0))==0
    assert ann.label('v',24,('phase',0))==1
    assert ann.label('v',25,('phase',1))==1
    with pytest.raises(ValueError,match='Conflicting'):
        AnnotationIndex(pd.concat([df,pd.DataFrame([dict(standardized_video_id='v',frame_idx=0,original_label='Preparation',grasper=0)])]))


def test_fractional_fps_and_tail_coverage():
    ids=sample_indices(3000,29.97,1.)
    assert abs(ids[-1]/29.97-100)<1/29.97
    starts=window_positions(35,16,16)
    assert set(i for s in starts for i in range(s,min(s+16,35)))==set(range(35))
    with pytest.raises(ValueError): sample_indices(10,25,30)


def test_frame_store_missing_never_pads(tmp_path):
    folder=tmp_path/'v'; folder.mkdir()
    Image.new('RGB',(8,8)).save(folder/'frame_0000025.jpg')
    store=FrameStore(tmp_path)
    assert list(store.names('v'))==[25]
    assert store.read('v',25).size==(8,8)
    with pytest.raises(FileNotFoundError): store.read('v',1)


def test_edl_matches_distribution_kl_and_masks():
    evidence=torch.tensor([[[3.,2.],[9.,4.]]],requires_grad=True)
    target=torch.tensor([[1.,-100.]])
    criterion=EvidentialLoss(.2)
    got=criterion(evidence,target)
    alpha=evidence[0,0]+1
    adjusted=torch.stack((alpha[0]*0+1,alpha[1]))
    expected=torch.digamma(alpha.sum())-torch.digamma(alpha[0]) + .2*torch.distributions.kl_divergence(
        torch.distributions.Dirichlet(adjusted),torch.distributions.Dirichlet(torch.ones(2)))
    torch.testing.assert_close(got,expected)
    got.backward(); assert torch.equal(evidence.grad[0,1],torch.zeros(2))
    assert EvidentialLoss()(torch.zeros(1,1,2),torch.tensor([[0.]])).item()==pytest.approx(1.)


def test_backward_warp_identity_translation_and_gradients():
    source=torch.arange(12,dtype=torch.float32).reshape(1,1,3,4).requires_grad_()
    flow=torch.zeros(1,2,3,4)
    same,mask=backward_warp(source,flow)
    torch.testing.assert_close(same,source); assert mask.all()
    flow[:,0]=-1
    shifted,mask=backward_warp(source,flow)
    torch.testing.assert_close(shifted[:,:,:,1:],source[:,:,:,:-1])
    assert not mask[:,:,:,0].any()
    shifted.sum().backward(); assert torch.isfinite(source.grad).all()


def test_metrics_ties_perfect_detection_empty_events():
    assert binary_metrics([0,1],[.5,.5])['auroc']==.5
    assert binary_metrics([0,1],[.01,.99])['aurc']==0
    assert temporal_ap({'v':[[1,2,.9]]},{'v':[[1,2]]})==1.
    assert temporal_ap({'v':[]},{'v':[[1,2]]})==0.
    assert temporal_ap({'v':[]},{'v':[]}) is None
    assert temporal_ap({'a':[[1,2,.9]],'b':[[1,2,.8]]},{'a':[],'b':[[1,2]]})==.5
    with pytest.raises(ValueError): temporal_ap({}, {'v':[]})
    with pytest.raises(ValueError): binary_metrics([0],[np.nan])


def test_segment_gap_merges_and_duration_clamps():
    assert segments_from_scores([0,1,2],[.9,.1,.8],2.5,merge_gap=1)==[[0.,2.5,.9]]
    assert segments_from_scores([0,1,2],[.1,.1,.1],2.5)==[]


def test_overlapping_metrics_count_frames_once():
    rows=[dict(video_id='v',text_query='q',frame_idx=1,label=1,score=s) for s in [.4,.8]]
    df=deduplicate_frame_rows(rows)
    assert len(df)==1 and df.score.iloc[0]==pytest.approx(.6)
    rows[1]['label']=0
    with pytest.raises(ValueError): deduplicate_frame_rows(rows)


def test_checkpoint_tensor_round_trip_and_prefix(tmp_path):
    first=torch.nn.Linear(3,2); second=torch.nn.Linear(3,2)
    path=tmp_path/'checkpoint.pt'; atomic_save({'model_state_dict':first.state_dict()},path)
    load_model_state(second,torch.load(path,weights_only=True))
    x=torch.randn(4,3); torch.testing.assert_close(first(x),second(x))
    load_model_state(second,{'model':{'module.'+k:v for k,v in first.state_dict().items()}})
    with pytest.raises(RuntimeError): load_model_state(torch.nn.Linear(4,2),torch.load(path,weights_only=True))
    with pytest.raises(ValueError): extract_model_state({'epoch':3})


def test_rng_restoration():
    state=rng_state(); values=(random.random(),np.random.rand(),torch.rand(2))
    restore_rng(state)
    assert random.random()==values[0] and np.random.rand()==values[1]
    torch.testing.assert_close(torch.rand(2),values[2])


def test_actual_partial_accumulation_matches_large_batch():
    from train import train_one_epoch
    class Tiny(torch.nn.Module):
        def __init__(self): super().__init__(); self.w=torch.nn.Parameter(torch.tensor(.2))
        def forward(self,x,*args): return x*self.w
    class Criterion:
        config=SimpleNamespace(TRAIN=SimpleNamespace(GRADIENT_ACCUMULATION_STEPS=4))
        def __call__(self,out,video,target):
            m=target!=-100; loss=((out[m]-target[m])**2).mean(); return loss,loss,loss*0
    a=Tiny(); b=copy.deepcopy(a)
    batches=[]
    for x,y in [([1.,2.],[1.,0.]),([3.],[1.]),([4.,5.],[0.,-100.])]:
        batches.append(dict(video_clip=torch.tensor(x),labels=torch.tensor(y),input_ids=torch.zeros(1),attention_mask=torch.ones(1)))
    optimizer=torch.optim.SGD(a.parameters(),lr=.1)
    train_one_epoch(a,batches,optimizer,Criterion(),None,'cpu',torch.amp.GradScaler('cuda',enabled=False))
    opt=torch.optim.SGD(b.parameters(),lr=.1)
    x=torch.tensor([1.,2.,3.,4.]); y=torch.tensor([1.,0.,1.,0.])
    ((b(x)-y)**2).mean().backward(); opt.step()
    torch.testing.assert_close(a.w,b.w)


def test_lora_freezes_text_and_preserves_zero_init(monkeypatch):
    import models
    class Text(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.config=SimpleNamespace(hidden_size=8)
            self.q_proj=torch.nn.Linear(8,8); self.v_proj=torch.nn.Linear(8,8); self.emb=torch.nn.Embedding(10,8)
    monkeypatch.setattr(models.AutoTokenizer,'from_pretrained',lambda *a,**k:object())
    monkeypatch.setattr(models.CLIPTextModel,'from_pretrained',lambda *a,**k:Text())
    encoder=models.TextEncoder(Config())
    assert all(('lora_A' in n or 'lora_B' in n)==p.requires_grad for n,p in encoder.named_parameters())
    original=torch.nn.Linear(8,8); lora=models.LoRALinear(original)
    x=torch.randn(2,8); torch.testing.assert_close(original(x),lora(x))


def test_confidence_pool_ignores_padding():
    from models import ConfidenceAwareFusion
    net=ConfidenceAwareFusion(8).eval(); visual=torch.randn(2,4,8); text=torch.randn(2,3,8); mask=torch.tensor([[1,1,0],[1,1,0]])
    first=net(visual,text,mask); text[:,2]=10000
    torch.testing.assert_close(first,net(visual,text,mask))


def test_detr_empty_target_backward_and_batch_invariant_spans():
    from moment_detr_module.dataset import collate_fn
    from moment_detr_module.matcher import HungarianMatcher
    from moment_detr_module.loss import SetCriterion
    def item(n,spans): return dict(video_feats=torch.zeros(n,4),query=torch.ones(2,dtype=torch.long),query_mask=torch.ones(2,dtype=torch.long),raw_spans=torch.tensor(spans).reshape(-1,2),duration=10,video_id='v',query_str='q',time_sec=torch.arange(n))
    one=item(3,[[.13,.26]]); empty=item(7,[])
    torch.testing.assert_close(collate_fn([one])['targets'][0]['segments'],collate_fn([one,empty])['targets'][0]['segments'])
    outputs=dict(pred_spans=torch.rand(1,2,2,requires_grad=True),pred_logits=torch.randn(1,2,2,requires_grad=True))
    criterion=SetCriterion(HungarianMatcher(),{'loss_span':1,'loss_giou':1,'loss_ce':1},.1)
    loss=sum(criterion(outputs,collate_fn([empty])['targets']).values())
    assert torch.isfinite(loss); loss.backward(); assert outputs['pred_logits'].grad is not None


def test_xclip_duplicate_query_loss():
    from train_xclip import multi_positive_loss
    logits=torch.tensor([[2.,2.,-2.],[2.,2.,-2.],[-2.,-2.,2.]],requires_grad=True)
    concepts=[('phase',1),('phase',1),('phase',2)]
    loss=multi_positive_loss(logits,concepts); loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    torch.testing.assert_close(logits.grad[0,0],logits.grad[0,1])


def test_actual_generalized_iou_uses_start_end_and_enclosure():
    from moment_detr_module.utils import generalized_temporal_iou
    a=torch.tensor([[0.,1.],[0.,.25]],requires_grad=True)
    b=torch.tensor([[0.,1.],[.75,1.]])
    result=generalized_temporal_iou(a,b)
    assert result[0,0].item()==pytest.approx(1.)
    assert result[1,1].item()==pytest.approx(-.5)
    result.sum().backward(); assert torch.isfinite(a.grad).all()
