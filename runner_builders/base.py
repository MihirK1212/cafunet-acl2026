from runner_interfaces import RunnerInterface

import constants
from enums import SplitRunType
from typing import Dict
from utils import config_utils

from .crisis_mmd_runners import *

config = config_utils.load_config()

def get_runners(**kwargs) -> Dict[SplitRunType, RunnerInterface]:
    model_name_to_runner_func_map: Dict[str, function] = {
        constants.MODEL_CRISIS_MMD_VISION_TEXT_PIPELINE: get_crisis_mmd_vision_text_pipline_model_runners,
    }
    return model_name_to_runner_func_map[config[constants.FIELD_MODEL_TO_USE]](**kwargs)
