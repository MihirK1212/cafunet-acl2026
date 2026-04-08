from runner_interfaces import RunnerInterface
from data.schemas import DataItemSchema
from abc import ABC, abstractmethod
from typing import List, Any, Dict, Tuple
import torch
from tqdm import tqdm

import constants
from model_interfaces import ClassicalModelInteface
from data.schemas import DataItemSchema, PytorchModelOutputSchema
from enums import SplitRunType
from runner_interfaces import RunnerInterface
from utils import gpu_utils, data_utils, config_utils, metric_utils
from torch.utils.data._utils.collate import default_collate


class ClassicalRunnerInterface(RunnerInterface):
    def __init__(self, model_interfaces, dataloader, split_run_type):
        super().__init__(model_interfaces, dataloader, split_run_type)
        assert len(model_interfaces) == 1
        assert constants.FIELD_CLASSICAL_MODEL_INTERFACE in model_interfaces
        self.model_interface = self._load_model_interface()

    def _load_model_interface(self) -> ClassicalModelInteface:
        return self.model_interfaces[constants.FIELD_CLASSICAL_MODEL_INTERFACE]

    def run_epoch(self, epoch: int) -> None:
        all_batch_items = []
        for _, batch_item in tqdm(enumerate(self.dataloader, 0)):
            all_batch_items.append(batch_item)
        collated_dict = default_collate(all_batch_items)
        data_item = DataItemSchema(**collated_dict)
        data_item = data_utils.load_data_item_to_device(
            data_item, device=gpu_utils.get_device()
        )

        if self.split_run_type == SplitRunType.TRAIN:
            self.model_interface.fit(data_item)

        _, predictions = self.model_interface.predict(data_item)
        targets = data_item.label

        predictions = data_utils.load_tensor_dict_to_device(
            predictions, device=constants.DEVICE_CPU, data_type=torch.long
        )
        targets = data_utils.load_tensor_dict_to_device(
            targets, device=constants.DEVICE_CPU, data_type=torch.long
        )

        self.epoch_analysis(
            epoch=epoch, probabilities={}, predictions=predictions, targets=targets
        )
