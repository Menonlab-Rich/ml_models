from torchmetrics import Metric
import torch

class ClassSpecificAccuracy(Metric):
    def __init__(self, ignore_indices=[0], dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.ignore_indices = ignore_indices
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds and target should be long tensors
        preds = preds.argmax(dim=1)
        mask = target not in self.ignore_indices
        preds = preds[mask]
        target = target[mask]
        self.correct += (preds == target).sum()
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total
