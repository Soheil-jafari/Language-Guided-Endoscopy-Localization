# This file contains all hyperparameters for the Moment-DETR model.
class Config:
    def __init__(self):
        self.seed = 42; self.dataset_name = "cholec80"; self.num_workers = 4
        self.feature_path = "extracted_features_resnet50"; self.ann_path = "preprocessed_data"
        self.v_feat_dim = 2048; self.t_feat_dim = 768
        self.max_v_len = 256; self.max_q_len = 32
        self.enc_layer = 2; self.dec_layer = 2
        self.dim_feedforward = 1024; self.hidden_dim = 256
        self.nheads = 8; self.dropout = 0.1; self.pre_norm = False
        self.position_embedding = 'sine'
        self.span_loss_type = 'l1'; self.ce_loss = True
        self.giou_loss_coef = 5.0; self.l1_loss_coef = 2.0
        self.eos_coef = 0.1; self.no_object_weight = 0.1
        self.epochs = 10; self.lr = 1e-4; self.lr_backbone = 1e-5
        self.lr_drop = 40; self.batch_size = 24
        self.weight_decay = 1e-4; self.clip_max_norm = 0.1


config = Config()