import torch, random, numpy as np, logging

def setup_seed(seed): torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed); random.seed(seed); torch.backends.cudnn.deterministic = True
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s")
    logger = logging.getLogger(name); logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w"); fh.setFormatter(formatter); logger.addHandler(fh)
    sh = logging.StreamHandler(); sh.setFormatter(formatter); logger.addHandler(sh)
    return logger
class RunningMeter:
    def __init__(self): self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1): self.sum += val * n; self.count += n; self.avg = self.sum / self.count
def span_cxw_to_xx(span): c, w = span.unbind(-1); return torch.stack([c - 0.5 * w, c + 0.5 * w], dim=-1).clamp(0, 1)
def generalized_temporal_iou(spans1, spans2):
    l1, r1 = span_cxw_to_xx(spans1).unbind(-1); l2, r2 = span_cxw_to_xx(spans2).unbind(-1)
    intersection = (torch.min(r1.unsqueeze(1), r2) - torch.max(l1.unsqueeze(1), l2)).clamp(0)
    union = (r1 - l1).unsqueeze(1) + (r2 - l2) - intersection
    return intersection / union.clamp(1e-7)
def get_iou(pred, gt):
    pred, gt = torch.from_numpy(pred), torch.from_numpy(gt)
    inter = (torch.min(pred[:, None, 1], gt[None, :, 1]) - torch.max(pred[:, None, 0], gt[None, :, 0])).clamp(0)
    union = (pred[:, None, 1] - pred[:, None, 0]) + (gt[None, :, 1] - gt[None, :, 0]) - inter
    return (inter / union.clamp(1e-7)).numpy()

