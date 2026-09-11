import copy
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image


class Tokenizer:
    def __call__(self,text,**kwargs):
        return dict(input_ids=torch.tensor([[1,2,3,0]]),attention_mask=torch.tensor([[1,1,1,0]]))


def test_dataset_real_files_full_validation_and_leakage(tmp_path):
    from project_config import Config
    from dataset import EndoscopyLocalizationDataset,create_dataloaders
    cfg=Config(); cfg.EXTRACTED_FRAMES_DIR=str(tmp_path/'frames'); cfg.VIDEO_METADATA_PATH=str(tmp_path/'meta.json')
    cfg.CHOLEC80_PARSED_ANNOTATIONS=str(tmp_path/'ann.csv'); cfg.DATA.SAMPLE_FPS=1; cfg.DATA.NUM_WORKERS=0; cfg.DATA.TRAIN_CROP_SIZE=16
    metadata={}; annotations=[]
    for video in ['v1','v2']:
        folder=tmp_path/'frames'/video; folder.mkdir(parents=True)
        metadata[video]=dict(source_fps=25,frame_count=125)
        for i in range(0,125,25): Image.new('RGB',(20,20),(i,0,0)).save(folder/f'frame_{i:07d}.jpg')
        # The final phase interval ends at the last annotated frame (100); frames after it are unknown.
        annotations.extend([dict(standardized_video_id=video,frame_idx=0,original_label='Preparation',grasper=1),
                            dict(standardized_video_id=video,frame_idx=50,original_label='CalotTriangleDissection',grasper=0),
                            dict(standardized_video_id=video,frame_idx=100,original_label='CalotTriangleDissection')])
    (tmp_path/'meta.json').write_text(json.dumps(metadata)); pd.DataFrame(annotations).to_csv(cfg.CHOLEC80_PARSED_ANNOTATIONS,index=False)
    for split,video in [('train','v1'),('val','v2')]:
        pd.DataFrame([dict(frame_path=f'old/{video}/frame_0000000.jpg',text_query='Preparation phase',relevance_label=1)]).to_csv(tmp_path/(split+'.csv'),index=False)
    ds=EndoscopyLocalizationDataset(tmp_path/'val.csv',Tokenizer(),4,False,cfg)
    assert ds[0]['frame_indices'].tolist()==[0,25,50,75]
    assert ds[0]['labels'].tolist()==[1.,1.,0.,0.]
    assert ds[0]['video_clip'].shape==(3,4,16,16)
    train,val=create_dataloaders(tmp_path/'train.csv',tmp_path/'val.csv',Tokenizer(),4,.2,cfg)
    assert len(val.dataset)==2  # includes final tail, not a validation subset
    # Training windows come from a fixed stride over the (video, query) grid, not one per triplet row.
    full=EndoscopyLocalizationDataset(tmp_path/'train.csv',Tokenizer(),4,True,cfg)
    assert [r[2] for r in full.records]==[0,1] and full.positive_windows==[True,True]
    cfg.TRAIN.POSITIVE_WINDOW_WEIGHT=3.
    weighted,_=create_dataloaders(tmp_path/'train.csv',tmp_path/'val.csv',Tokenizer(),4,1.,cfg)
    assert isinstance(weighted.sampler,torch.utils.data.WeightedRandomSampler) and len(list(weighted.sampler))==2
    cfg.TRAIN.POSITIVE_WINDOW_WEIGHT=1.
    with pytest.raises(ValueError,match='leakage'):
        create_dataloaders(tmp_path/'train.csv',tmp_path/'train.csv',Tokenizer(),4,1.,cfg)
    (tmp_path/'frames'/'v2'/'frame_0000025.jpg').unlink()
    with pytest.raises(FileNotFoundError): EndoscopyLocalizationDataset(tmp_path/'val.csv',Tokenizer(),4,False,cfg)


def test_actual_backbone_checkpoint_gradients_and_rectangular_tokens():
    from backbone.vision_transformer import VisionTransformer
    torch.manual_seed(4)
    a=VisionTransformer(img_size=(16,24),patch_size=8,embed_dim=16,depth=2,num_heads=4,num_frames=3,attention_type='divided_space_time',use_checkpoint=False)
    b=copy.deepcopy(a); b.use_checkpoint=True
    a.train(); b.train(); x=torch.randn(2,3,3,16,24)
    one=a.forward_features(x,get_all=True); two=b.forward_features(x,get_all=True)
    assert one.shape==(2,19,16)
    torch.testing.assert_close(one,two)
    one.square().sum().backward(); two.square().sum().backward()
    for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
        if p.grad is not None: torch.testing.assert_close(p.grad,q.grad,rtol=1e-4,atol=1e-5,msg=name)


def test_tiny_full_framework_forward_backward_reload(monkeypatch):
    import models
    from project_config import Config
    from backbone.vision_transformer import VisionTransformer
    from transformers import CLIPTextModel,CLIPTextConfig
    from losses import MasterLoss
    from checkpoint_utils import load_model_state
    cfg=Config(); cfg.DATA.TRAIN_CROP_SIZE=16; cfg.DATA.NUM_FRAMES=3; cfg.DATA.CLIP_LENGTH=3
    cfg.MODEL.HEAD_NUM_ATTENTION_HEADS=4; cfg.MODEL.HEAD_NUM_LAYERS=1; cfg.TRAIN.DEVICE='cpu'; cfg.TRAIN.LORA_DROPOUT=0
    monkeypatch.setattr(models,'VisionTransformer',lambda **kw:VisionTransformer(img_size=16,patch_size=8,embed_dim=16,depth=1,num_heads=4,num_frames=3))
    monkeypatch.setattr(models.AutoTokenizer,'from_pretrained',lambda *a,**k:Tokenizer())
    monkeypatch.setattr(models.CLIPTextModel,'from_pretrained',lambda *a,**k:CLIPTextModel(CLIPTextConfig(vocab_size=10,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=4,max_position_embeddings=8)))
    for uncertainty in [False,True]:
        cfg.MODEL.USE_UNCERTAINTY=uncertainty
        model=models.LocalizationFramework(cfg,initialize_backbone=False)
        model.language_guided_head.return_xai_map=True
        video=torch.randn(2,3,3,16,16); ids=torch.tensor([[1,2,3,0],[1,2,3,0]]); mask=(ids!=0).long()
        out=model(video,ids,mask); loss,*_=MasterLoss(cfg)(out,video,torch.tensor([[1.,0.,-100.],[0.,1.,1.]]))
        torch.testing.assert_close(out[2].mean((-1,-2)),out[1].squeeze(-1))
        assert torch.isfinite(loss); loss.backward()
        assert model.temporal_head.fc_output.weight.grad is not None
        model.eval(); before=model(video,ids,mask)[0]
        second=models.LocalizationFramework(copy.deepcopy(cfg),initialize_backbone=False).eval()
        load_model_state(second,{'model_state_dict':model.state_dict()})
        torch.testing.assert_close(before,second(video,ids,mask)[0])


def test_huggingface_xclip_logits_axes_offline():
    from transformers import XCLIPConfig,XCLIPTextConfig,XCLIPVisionConfig,XCLIPModel
    tc=XCLIPTextConfig(vocab_size=20,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=4,max_position_embeddings=8)
    vc=XCLIPVisionConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=4,image_size=16,patch_size=8,num_frames=2,mit_hidden_size=16,mit_intermediate_size=32,mit_num_hidden_layers=1,mit_num_attention_heads=4)
    cfg=XCLIPConfig.from_text_vision_configs(tc,vc,projection_dim=16,prompt_layers=1,prompt_alpha=0.1,prompt_hidden_act='quick_gelu',prompt_num_attention_heads=4)
    model=XCLIPModel(cfg)
    out=model(pixel_values=torch.randn(2,2,3,16,16),input_ids=torch.tensor([[1,2,3],[1,4,3]]),attention_mask=torch.ones(2,3,dtype=torch.long))
    assert out.logits_per_video.shape==(2,2)
    assert out.text_embeds.shape==(2,2,16)  # video,query,embedding; no token axis
    assert torch.isfinite(out.logits_per_video).all()


def test_adapted_detr_full_forward_with_mixed_empty_targets(monkeypatch):
    from transformers import RobertaConfig,RobertaModel
    from moment_detr_module import modeling
    from moment_detr_module.configs import Config
    monkeypatch.setattr(modeling.RobertaModel,'from_pretrained',lambda *a,**k:RobertaModel(RobertaConfig(vocab_size=20,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=4,max_position_embeddings=32)))
    cfg=Config(); cfg.v_feat_dim=8; cfg.t_feat_dim=16; cfg.hidden_dim=16; cfg.nheads=4
    cfg.enc_layer=1; cfg.dec_layer=1; cfg.dim_feedforward=32; cfg.max_v_len=5; cfg.dropout=0.
    model=modeling.MomentDETR(cfg)
    targets=[dict(spans=torch.tensor([[.5,.3]]),labels=torch.tensor([0])),dict(spans=torch.empty(0,2),labels=torch.empty(0,dtype=torch.long))]
    video=torch.randn(2,5,8); valid=torch.tensor([[1,1,1,0,0],[1,1,1,1,1]],dtype=torch.bool)
    query=torch.tensor([[0,2,3],[0,2,4]]); mask=torch.ones_like(query)
    output=model(video,valid,query,mask,targets)
    loss=sum(output['loss_dict'].values()); assert torch.isfinite(loss); loss.backward()
    assert model.span_start_head.weight.grad is not None
    model.eval(); assert model(video,valid,query,mask)['pred_logits'].shape==(2,5,2)


def test_real_video_extraction_and_inference_share_time_grid(tmp_path):
    import cv2
    from dataset_preprocessing.build_dataset import extract
    from inference import read_sampled_video
    source=tmp_path/'videos'; source.mkdir(); video=source/'video01.mp4'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),29.97,(32,32))
    assert writer.isOpened()
    for i in range(61): writer.write(np.full((32,32,3),i,dtype=np.uint8))
    writer.release()
    extract(source,tmp_path/'frames',tmp_path/'meta.json',1.)
    meta=json.loads((tmp_path/'meta.json').read_text())['CHOLEC80__video01']
    images,ids,fps,duration,info=read_sampled_video(video,1.,16)
    assert ids.tolist()==[0,30,60]
    assert meta['frame_count']==61 and meta['source_fps']==pytest.approx(fps)
    assert meta['reported_frame_count']==info['reported_frame_count'] and info['decoded_frame_count']==61
    assert duration==pytest.approx(61/fps) and len(images)==3
    assert sorted(p.name for p in (tmp_path/'frames'/'CHOLEC80__video01').glob('*.jpg'))==[f'frame_{i:07d}.jpg' for i in ids]
    with pytest.raises(FileExistsError): extract(source,tmp_path/'frames',tmp_path/'meta.json',1.)


def test_zip_frame_store_reopens_for_workers(tmp_path):
    import io
    import pickle
    import zipfile
    from data_contract import FrameStore
    data=io.BytesIO(); Image.new('RGB',(8,8),(30,40,50)).save(data,format='JPEG')
    with zipfile.ZipFile(tmp_path/'v.zip','w') as z: z.writestr('frame_0000025.jpg',data.getvalue())
    store=FrameStore(tmp_path); before=np.asarray(store.read('v',25))
    clone=pickle.loads(pickle.dumps(store)); np.testing.assert_array_equal(before,np.asarray(clone.read('v',25)))
    original=store._zips['v']; store._pid=-1
    np.testing.assert_array_equal(before,np.asarray(store.read('v',25)))
    assert store._zips['v'] is not original
    clone.close(); store.close()
