from abc import ABC, abstractmethod
from typing import List, Any, Dict, Tuple
import torch
import torch.distributed
from tqdm import tqdm

import constants
from model_interfaces import ModelInterface, PytorchModelInterface
from data.schemas import DataItemSchema, PytorchModelOutputSchema
from enums import SplitRunType
from runner_interfaces import RunnerInterface
from utils import gpu_utils, data_utils, config_utils, metric_utils

config = config_utils.load_config()

class ParallelPytorchRunnerInterface(RunnerInterface):
    def __init__(self, model_interfaces: Dict[str, PytorchModelInterface], dataloader: Any, sampler: Any, split_run_type: SplitRunType, rank: int) -> None:
        super().__init__(model_interfaces, dataloader, split_run_type, rank)
        assert len(model_interfaces) == 1
        assert constants.FIELD_PYTROCH_MODEL_INTERFACE in model_interfaces
        self.model_interface = self._load_model_interface()
        self.sampler = sampler 

    def _load_model_interface(self) -> PytorchModelInterface:
        return self.model_interfaces[constants.FIELD_PYTROCH_MODEL_INTERFACE]

    def run_epoch(self, epoch: int) -> None:
        torch.cuda.empty_cache()

        if self.split_run_type == SplitRunType.TRAIN:
            assert self.sampler is not None
            self.sampler.set_epoch(epoch)
        else:
            assert self.sampler is None

        self.model_interface.prepare_model(self.split_run_type)

        num_steps = 0
        total_loss = 0
        batch_probabilities: List[Dict[str, torch.Tensor]] = []
        batch_predictions: List[Dict[str, torch.Tensor]] = []
        batch_targets: List[Dict[str, torch.Tensor]] = []

        for _, batch_item in tqdm(enumerate(self.dataloader, 0)):
            gpu_utils.clear_cuda_cache()
            
            data_item = DataItemSchema(**batch_item)
            data_item = data_utils.load_data_item_to_device(data_item, device=self.rank)

            model_output, prediction = self.model_interface.predict(data_item)
            target = data_item.label

            loss = self.model_interface.get_loss(model_output, target)

            if self.split_run_type == SplitRunType.TRAIN:
                self.model_interface.fit(loss)

            probabilities = data_utils.load_tensor_dict_to_device(model_output.pred_logits, device=constants.DEVICE_CPU)
            prediction = data_utils.load_tensor_dict_to_device(prediction, device=constants.DEVICE_CPU, data_type=torch.long)
            target = data_utils.load_tensor_dict_to_device(target, device=constants.DEVICE_CPU, data_type=torch.long)

            num_steps+=1
            total_loss+=loss.item()
            batch_probabilities.append(probabilities)
            batch_predictions.append(prediction)
            batch_targets.append(target)

        if config[constants.FIELD_USE_LR_SCHEDULER]:
            self.model_interface.scheduler_step()
        
        epoch_loss = total_loss / num_steps
        probabilities = data_utils.concatenate_tensor_dict_list(batch_probabilities)
        predictions = data_utils.concatenate_tensor_dict_list(batch_predictions)
        targets = data_utils.concatenate_tensor_dict_list(batch_targets)

        torch.distributed.barrier()

        if self.rank == config.get(constants.FIELD_READ_WRITE_GPU_RANK):
            self.epoch_analysis(epoch=epoch, probabilities=probabilities, predictions=predictions, targets=targets, other_values={'epoch_loss': epoch_loss})
        
        del epoch_loss
        torch.cuda.empty_cache()

    
