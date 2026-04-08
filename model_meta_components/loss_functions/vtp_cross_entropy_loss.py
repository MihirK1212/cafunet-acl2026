import torch
from typing import Dict

from data.schemas import PytorchModelOutputSchema

class VisionTextPipelineCrossEntropyLoss(torch.nn.Module):
    def __init__(self) -> None:     
        super(VisionTextPipelineCrossEntropyLoss, self).__init__()
        self.classification_loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, model_output: PytorchModelOutputSchema, target: Dict[str, torch.Tensor]) -> torch.Tensor:
        pred_logits = model_output.pred_logits
        assert pred_logits.keys() == target.keys()
        loss = 0
        for k in pred_logits.keys():
            loss+=self.classification_loss_function(pred_logits[k], target[k])
        return loss