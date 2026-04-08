from runner_interfaces import RunnerInterface
from data.schemas import DataItemSchema
from abc import ABC, abstractmethod
from typing import List, Any, Dict, Tuple
import torch
from tqdm import tqdm

import constants
from model_interfaces import ClassicalModelInteface, PytorchModelInterface
from data.schemas import DataItemSchema, PytorchModelOutputSchema
from enums import SplitRunType
from runner_interfaces import RunnerInterface
from utils import gpu_utils, data_utils, config_utils, metric_utils


class ClassicalPytorchHybridRunnerInterface(RunnerInterface):
    def __init__(self, model_interfaces, dataloader, split_run_type):
        super().__init__(model_interfaces, dataloader, split_run_type)
        assert len(model_interfaces) == 2
        assert constants.FIELD_CLASSICAL_MODEL_INTERFACE in model_interfaces
        assert constants.FIELD_PYTROCH_MODEL_INTERFACE in model_interfaces
        self.pytorch_model_interface = self._load_pytorch_model_interface()
        self.clasical_model_interface = self._load_classical_model_interface()

    def _load_pytorch_model_interface(self) -> PytorchModelInterface:
        return self.model_interfaces[constants.FIELD_PYTROCH_MODEL_INTERFACE]

    def _load_classical_model_interface(self) -> ClassicalModelInteface:
        return self.model_interfaces[constants.FIELD_CLASSICAL_MODEL_INTERFACE]
        
    def run_epoch(self, epoch: int) -> None:
        torch.cuda.empty_cache()

        self.pytorch_model_interface.prepare_model(SplitRunType.TEST)

        batch_feature_vectors: List[torch.Tensor] = []
        batch_targets: List[Dict[str, torch.Tensor]] = []

        for _, batch_item in tqdm(enumerate(self.dataloader, 0)):
            gpu_utils.clear_cuda_cache()
            
            data_item = DataItemSchema(**batch_item)
            data_item = data_utils.load_data_item_to_device(data_item, device=gpu_utils.get_device())

            with torch.no_grad():
                model_output, _ = self.pytorch_model_interface.predict(data_item)
            target = data_item.label

            feature_vector = model_output.metadata['feature_vector'].to(constants.DEVICE_CPU)
            target = data_utils.load_tensor_dict_to_device(target, device=constants.DEVICE_CPU, data_type=torch.long)

            batch_feature_vectors.append(feature_vector)
            batch_targets.append(target)
        
        feature_vectors = torch.cat(batch_feature_vectors).to(constants.DEVICE_CPU)
        targets = data_utils.concatenate_tensor_dict_list(batch_targets)

        classical_data_item = DataItemSchema(
            image_tensors={},
            tokenized_text_inputs={},
            label=targets,
            topics_tokenized_inputs=[],
            metadata={
                'feature_vector': feature_vectors
            }
        )

        if self.split_run_type == SplitRunType.TRAIN:
            self.clasical_model_interface.fit(classical_data_item)

        _, predictions = self.clasical_model_interface.predict(classical_data_item)

        predictions = data_utils.load_tensor_dict_to_device(predictions, device=constants.DEVICE_CPU, data_type=torch.long)
        targets = data_utils.load_tensor_dict_to_device(targets, device=constants.DEVICE_CPU, data_type=torch.long)

        self.epoch_analysis(epoch=epoch, probabilities={}, predictions=predictions, targets=targets)