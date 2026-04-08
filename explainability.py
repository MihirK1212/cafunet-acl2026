from runner_builders.crisis_mmd_runners import get_crisis_mmd_vision_text_pipline_model_runners
from enums import SplitRunType
from utils import metric_utils
# from lime.lime_text import LimeTextExplainer
from data.schemas import DataItemSchema

if __name__ == "__main__":
    test_model_runner_interface = get_crisis_mmd_vision_text_pipline_model_runners(
        0, True
    )[SplitRunType.TEST]

    captions = []
    data_loader = test_model_runner_interface.dataloader
    i = 0
    for batch_item in data_loader:
        i+=1
        if i==50:
            break

        data_item = DataItemSchema(**batch_item)
        batch_captions = list(data_item.metadata['caption'])
        captions.extend(batch_captions)

    print('captions', captions)


    epoch_result = test_model_runner_interface.run_epoch(0)

    probabilities = epoch_result['probabilities']
    predictions = epoch_result['predictions']
    targets = epoch_result['targets']

    assert predictions.keys() == targets.keys()
    for k in predictions.keys():
        metrics_dict = metric_utils.get_metrics(
            probabilities.get(k), predictions[k], targets[k]
        )
        print('metrics_dict', metrics_dict)