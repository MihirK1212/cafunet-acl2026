import torch

from data.datasets import CrisisMMDDataset, TSEQDDataset
from models import (
    CrisisMMDVisionTextPipelineModel,
)
from model_interfaces import PytorchModelInterface
from model_meta_components.loss_functions import (
    VisionTextPipelineLoss,
)
from runner_interfaces import (
    RunnerInterface,
    PytorchRunnerInterface,
    ParallelPytorchRunnerInterface,
)
from torch.nn.parallel import DistributedDataParallel as DDP

import constants
from data.dataloader_interface import get_dataloader
from enums import SplitRunType
from typing import Dict
from utils import config_utils, gpu_utils

config = config_utils.load_config()


def _get_dataset_cls():
    if config.get('crisis_mmd_like_dataset_to_use') == constants.FIELD_CRISIS_MMD_DATASET:
        return CrisisMMDDataset
    elif config.get('crisis_mmd_like_dataset_to_use') == constants.FIELD_TSEQD_DATASET:
        return TSEQDDataset
    return None

def _get_model_save_path():
    if config.get('crisis_mmd_like_dataset_to_use') == constants.FIELD_CRISIS_MMD_DATASET:
        return "/home/shahid/1.Mihir/cafunet-acl2026/models/cirsis_mmd_vision_text_pipeline/saved_model/crisis_mmd_model.pth"
    elif config.get('crisis_mmd_like_dataset_to_use') == constants.FIELD_TSEQD_DATASET:
        return "/home/shahid/1.Mihir/cafunet-acl2026/models/cirsis_mmd_vision_text_pipeline/saved_model/tseqd_model.pth"
    return None

def get_crisis_mmd_vision_text_pipline_model_runners(
    rank: int,
    use_pretrained: bool = False
) -> Dict[SplitRunType, RunnerInterface]:
    dataset_cls = _get_dataset_cls()
    if config["run_parallel"]:
        train_dataloader, train_sampler = get_dataloader(
            dataset_cls=dataset_cls, split_run_type=SplitRunType.TRAIN, rank=rank
        )
    else:
        train_dataloader = get_dataloader(
            dataset_cls=dataset_cls, split_run_type=SplitRunType.TRAIN
        )
    validation_dataloader = get_dataloader(
        dataset_cls=dataset_cls, split_run_type=SplitRunType.VALIDATION
    )
    test_dataloader = get_dataloader(
        dataset_cls=dataset_cls, split_run_type=SplitRunType.TEST
    )

    model = CrisisMMDVisionTextPipelineModel(rank=rank)
    model = model.to(rank)
    if config[constants.FIELD_RUN_PARALLEL]:
        model = DDP(model, device_ids=[rank])
    if use_pretrained:
        model.load_state_dict(
            torch.load(
                _get_model_save_path(),
                map_location=torch.device(gpu_utils.get_device()),
            )
        )

    class_weights_crisis_mmd = torch.tensor([0.4692, 1.0064, 2.9198, 1.8770])
    class_weights_crisis_tseqd = torch.tensor([2.2736, 8.1695, 9.4510, 3.0125])

    class_weights = (
        class_weights_crisis_mmd if config.get('crisis_mmd_like_dataset_to_use') == constants.FIELD_CRISIS_MMD_DATASET
        else class_weights_crisis_tseqd
    )

    loss_function = VisionTextPipelineLoss(
        rank=rank, 
        class_weights=class_weights
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.00007)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    model_interface = PytorchModelInterface(
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        scheduler=scheduler,
        rank=rank,
        model_save_path=_get_model_save_path(),
    )

    model_interfaces = {constants.FIELD_PYTROCH_MODEL_INTERFACE: model_interface}

    if config["run_parallel"]:
        return {
            SplitRunType.TRAIN: ParallelPytorchRunnerInterface(
                model_interfaces=model_interfaces,
                dataloader=train_dataloader,
                sampler=train_sampler,
                split_run_type=SplitRunType.TRAIN,
                rank=rank,
            ),
            SplitRunType.VALIDATION: ParallelPytorchRunnerInterface(
                model_interfaces=model_interfaces,
                dataloader=validation_dataloader,
                sampler=None,
                split_run_type=SplitRunType.VALIDATION,
                rank=rank,
            ),
            SplitRunType.TEST: ParallelPytorchRunnerInterface(
                model_interfaces=model_interfaces,
                dataloader=test_dataloader,
                sampler=None,
                split_run_type=SplitRunType.TEST,
                rank=rank,
            ),
        }
    else:
        return {
            SplitRunType.TRAIN: PytorchRunnerInterface(
                model_interfaces=model_interfaces,
                dataloader=train_dataloader,
                split_run_type=SplitRunType.TRAIN,
            ),
            SplitRunType.VALIDATION: PytorchRunnerInterface(
                model_interfaces=model_interfaces,
                dataloader=validation_dataloader,
                split_run_type=SplitRunType.VALIDATION,
            ),
            SplitRunType.TEST: PytorchRunnerInterface(
                model_interfaces=model_interfaces,
                dataloader=test_dataloader,
                split_run_type=SplitRunType.TEST,
            ),
        }

